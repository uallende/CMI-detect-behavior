import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as k
from tensorflow import argmax, minimum, shape
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras import Layer, Sequential
from tensorflow.keras.models import Model 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import pad_sequences, Sequence 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import (
    Dense, 
    Input, 
    Conv1D, 
    MaxPooling1D, 
    GlobalMaxPooling1D, 
    Concatenate, 
    GlobalAveragePooling1D,
    BatchNormalization,
    GRU,
    Dropout,
    add,
    Activation,
    Multiply, 
    Reshape,
    LayerNormalization,
    Add,
    MultiHeadAttention,
    Bidirectional,
    LSTM,
    UpSampling1D,
    Lambda,
    GaussianNoise
    )

def wave_block(x, filters, kernel_size, n, dropout_rate=0.3):
    dilation_rates = [2**i for i in range(n)]

    # Initial 1x1 projection
    x_in = Conv1D(filters=filters, kernel_size=1, padding='same')(x)
    res_x = x_in

    for dilation_rate in dilation_rates:
        # Gated activation unit
        tanh_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='tanh',
                          dilation_rate=dilation_rate)(res_x)

        sigm_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='sigmoid',
                          dilation_rate=dilation_rate)(res_x)

        gated = Multiply()([tanh_out, sigm_out])
        gated = Dropout(dropout_rate)(gated)

        # 1x1 projection after gated activation
        transformed = Conv1D(filters=filters,
                             kernel_size=1,
                             padding='same')(gated)

        # Residual connection
        res_x = Add()([res_x, transformed])
        res_x = LayerNormalization()(res_x)

    return res_x

def crop_or_pad(inputs):
    x, skip = inputs
    x_len = shape(x)[1]
    skip_len = shape(skip)[1]
    min_len = minimum(x_len, skip_len)
    return x[:, :min_len, :], skip[:, :min_len, :]

def crop_or_pad_output_shape(input_shapes):
    shape1, shape2 = input_shapes
    min_time_steps = min(shape1[1], shape2[1])
    num_features = shape1[2]
    output_shape = (None, min_time_steps, num_features)
    return [output_shape, output_shape]

def match_time_steps(x, skip):    
    x, skip = Lambda(
        crop_or_pad, 
        output_shape=crop_or_pad_output_shape 
    )([x, skip])
    return x, skip

def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    """
    Output: (B, T, # filters)
    """
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = LayerNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=l2(wd))(shortcut)
        shortcut = LayerNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def res_se_cnn_decoder_block(x, filters, kernel_size, drop=0.3, wd=1e-4, skip_connection=None):
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
               kernel_regularizer=l2(wd))(x)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)

    if skip_connection is not None:
        x, skip_connection = match_time_steps(x, skip_connection)
        x = Concatenate()([x, skip_connection])

    x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
               kernel_regularizer=l2(wd))(x)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)

    x = se_block(x)
    x = Dropout(drop)(x)
    return x

def unet_se_cnn(x, unet_depth=3, base_filters=64, kernel_size=3, drop=0.3):
    filters = base_filters
    skips = []
    
    # Encoder
    for _ in range(unet_depth):
        x = residual_se_cnn_block(x, filters, kernel_size, drop=drop)
        skips.append(x)
        filters *= 2
    
    # Bottleneck
    c_shape = x.shape[-1]
    x = Dense(128)(x)
    x = Dense(c_shape)(x)
    
    # Decoder 
    for skip in reversed(skips):
        filters //= 2
        x = res_se_cnn_decoder_block(x, filters, kernel_size, drop=drop, skip_connection=skip)
    
    return x

def res_se_cnn_wave_gru_block(x, filters, kernel_size, dilation_depth, dropout_rate=0.3):
    x1 = residual_se_cnn_block(x, filters, kernel_size)
    x2 = wave_block(x, filters, kernel_size, dilation_depth)
    x2 = MaxPooling1D(2)(x2)
    
    x = Concatenate()([x1, x2])
    skip = x
    gru_params = filters*2

    x = Bidirectional(GRU(gru_params, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x) 
    x = Dense(gru_params, activation="relu")(x)
    return Add()([x, skip])
        
class GatedMixupGenerator(Sequence):
    def __init__(self, X, y, gate_targets, batch_size, imu_dim, class_weight=None, alpha=0.2, masking_prob=0.0):
        self.X, self.y = X, y
        self.gate_targets = gate_targets  
        self.batch = batch_size
        self.imu_dim = imu_dim
        self.class_weight = class_weight
        self.alpha = alpha
        self.masking_prob = masking_prob
        self.indices = np.arange(len(X))
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx].copy(), self.y[idx].copy()
        
        gate_target = self.gate_targets[idx].copy()

        if self.masking_prob > 0:
            for i in range(len(Xb)):
                # If the gate is 1.0 (has ToF) AND we hit the random chance...
                if gate_target[i] == 1.0 and np.random.rand() < self.masking_prob:
                    Xb[i, :, self.imu_dim:] = 0  # Zero out the ToF features
                    gate_target[i] = 0.0         # Set the gate to 0 for this augmented sample

        # The rest of the logic (class weights, mixup) can remain the same
        sample_weights = np.ones(len(Xb), dtype='float32')
        if self.class_weight:
            y_integers = yb.argmax(axis=1)
            sample_weights = np.array([self.class_weight[i] for i in y_integers])

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            perm = np.random.permutation(len(Xb))
            X_mix = lam * Xb + (1 - lam) * Xb[perm]
            y_mix = lam * yb + (1 - lam) * yb[perm]
            gate_target_mix = lam * gate_target + (1 - lam) * gate_target[perm]
            sample_weights_mix = lam * sample_weights + (1 - lam) * sample_weights[perm]
            return X_mix, {'main_output': y_mix, 'tof_gate': gate_target_mix[:, np.newaxis]}, sample_weights_mix

        return Xb, {'main_output': yb, 'tof_gate': gate_target[:, np.newaxis]}, sample_weights    

def on_epoch_end(self):
    np.random.shuffle(self.indices)    

def time_sum(x):
    return k.sum(x, axis=1)

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def tof_block_2(tof_inputs, wd=1e-4):
    # TOF/Thermal lighter branch
    x2 = tf.keras.layers.Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(wd))(tof_inputs)
    x2 = tf.keras.layers.BatchNormalization()(x2); x2 = tf.keras.layers.Activation('relu')(x2)
    x2 = tf.keras.layers.MaxPooling1D(2)(x2); x2 = tf.keras.layers.Dropout(0.2)(x2)
    x2 = tf.keras.layers.Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(wd))(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2); x2 = tf.keras.layers.Activation('relu')(x2)
    x2 = tf.keras.layers.MaxPooling1D(2)(x2); x2 = tf.keras.layers.Dropout(0.2)(x2)
    return x2

def tof_block(tof_inputs, wd=1e-4):
    x2_base = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof_inputs)
    x2_base = BatchNormalization()(x2_base); x2_base = Activation('relu')(x2_base)
    x2_base = MaxPooling1D(2)(x2_base); x2_base = Dropout(0.2)(x2_base)
    x2_base = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2_base)
    x2_base = BatchNormalization()(x2_base); x2_base = Activation('relu')(x2_base)

    gate_input = GlobalAveragePooling1D()(tof_inputs)
    gate_input = Dense(16, activation='relu')(gate_input)

    gate = Dense(1, activation='sigmoid', name='tof_gate_dense')(gate_input)
    return Multiply()([x2_base, gate])

def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context    

def features_processing_old(x1, x2, wd=1e-4):
    merged = Concatenate()([x1, x2])
    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation='elu')(xc)
    
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)
    return x

def features_processing(x1, x2, wd=1e-4):
    # --- THIS IS THE FIX ---
    # Match the time dimensions of the two input tensors before concatenating.
    # This will crop the longer tensor to match the shorter one.
    x1_matched, x2_matched = match_time_steps(x1, x2)
    
    # Now, concatenation will work correctly.
    merged = Concatenate()([x1_matched, x2_matched])
    
    # The rest of the function remains the same
    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation='elu')(xc)
    
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)

    return x

def kaggle_082(dataset, imu_dim, wd=1e-4):
    sample_batch = next(iter(dataset))
    input_shape = sample_batch[0].shape[1:]
    inp = tf.keras.layers.Input(shape=input_shape)
    imu = tf.keras.layers.Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = tf.keras.layers.Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMU deep branch
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    # TOF/Thermal lighter branch
    x2 = tf.keras.layers.Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(wd))(tof)
    x2 = tf.keras.layers.BatchNormalization()(x2); x2 = tf.keras.layers.Activation('relu')(x2)
    x2 = tf.keras.layers.MaxPooling1D(2)(x2); x2 = tf.keras.layers.Dropout(0.2)(x2)
    x2 = tf.keras.layers.Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(wd))(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2); x2 = tf.keras.layers.Activation('relu')(x2)
    x2 = tf.keras.layers.MaxPooling1D(2)(x2); x2 = tf.keras.layers.Dropout(0.2)(x2)

    merged = tf.keras.layers.Concatenate()([x1, x2])

    xa = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(wd)))(merged)
    xb = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(wd)))(merged)
    xc = tf.keras.layers.GaussianNoise(0.09)(merged)
    xc = tf.keras.layers.Dense(16, activation='elu')(xc)
    
    x = tf.keras.layers.Concatenate()([xa, xb, xc])
    x = tf.keras.layers.Dropout(0.4)(x)
    x = attention_layer(x)

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = tf.keras.layers.Dense(units, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(wd))(x)
        x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(drop)(x)

    main_out = tf.keras.layers.Dense(18, activation="softmax", name="main_output")(x)
    gate_out = tf.keras.layers.Dense(1, activation="sigmoid", name="tof_gate")(x) # Renamed layer
    
    return tf.keras.models.Model(inputs=inp, outputs={"main_output": main_out, "tof_gate": gate_out})

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):  
        attn_output = self.att(inputs, inputs, training=training)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# SOME TESTS

# def kaggle_solution_model(pad_len, n_classes, wd=1e-4):
#     inp = Input(shape=(pad_len, 16))
#     imu = Lambda(lambda t: t[:, :, :])(inp)
#     # tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

#     # IMU deep branch
#     x = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
#     x = residual_se_cnn_block(x, 128, 5, drop=0.1, wd=wd)

#     xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(x)
#     xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(x)
#     xc = GaussianNoise(0.09)(x)
#     xc = Dense(16, activation='elu')(xc)
    
#     x = Concatenate()([xa, xb, xc])
#     x = Dropout(0.4)(x)
#     x = attention_layer(x)

#     for units, drop in [(256, 0.5), (128, 0.3)]:
#         x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
#         x = BatchNormalization()(x); x = Activation('relu')(x)
#         x = Dropout(drop)(x)

#     out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd), name="main_output")(x)
#     return Model(inp, out)

def unet_se_cnn_bilstm(x, base_filters=32, kernel_size=3, drop=0.3):
    filters = base_filters
    skips = []
    
    # Encoder
    for _ in range(3):
        x = residual_se_cnn_block(x, filters, kernel_size, drop=drop)
        skips.append(x)
        filters *= 2
    
    # Bottleneck
    c_shape = x.shape[-1] * 2
    x = Bidirectional(LSTM(c_shape, return_sequences=True))(x)
    x = Dense(c_shape)(x)
    
    # Decoder 
    for skip in reversed(skips):
        filters //= 2
        x = res_se_cnn_decoder_block(x, filters, kernel_size, drop=drop, skip_connection=skip)
    
    return x    

# def res_se_cnn_wave_lstm_block(x, filters, kernel_size, dilation_depth, dropout_rate=0.3):
#     x1 = residual_se_cnn_block(x, filters, kernel_size)
#     x2 = wave_block(x, filters, kernel_size, dilation_depth)
#     x2 = MaxPooling1D(2)(x2)
    
#     x = Concatenate()([x1, x2])
#     skip = x
#     lstm_params = filters

#     x = Bidirectional(LSTM(lstm_params, return_sequences=True))(x)
#     x = Dropout(dropout_rate)(x) 
#     x = Dense(lstm_params, activation="relu")(x)
#     return Add()([x, skip])



# class GatedMixupGenerator(Sequence):
#     def __init__(self, X, y, batch_size, imu_dim, class_weight=None, alpha=0.2, masking_prob=0.0):
#         self.X, self.y = X, y
#         self.batch = batch_size
#         self.imu_dim = imu_dim
#         self.class_weight = class_weight
#         self.alpha = alpha
#         self.masking_prob = masking_prob
#         self.indices = np.arange(len(X))
        
#     def __len__(self):
#         return int(np.ceil(len(self.X) / self.batch))

#     def __getitem__(self, i):
#         idx = self.indices[i*self.batch:(i+1)*self.batch]
#         Xb, yb = self.X[idx].copy(), self.y[idx].copy()
        
#         # サンプルごとの重みを計算
#         sample_weights = np.ones(len(Xb), dtype='float32')
#         if self.class_weight:
#             y_integers = yb.argmax(axis=1)
#             sample_weights = np.array([self.class_weight[i] for i in y_integers])
        
#         gate_target = np.ones(len(Xb), dtype='float32')
#         if self.masking_prob > 0:
#             for i in range(len(Xb)):
#                 if np.random.rand() < self.masking_prob:
#                     Xb[i, :, self.imu_dim:] = 0
#                     gate_target[i] = 0.0

#         if self.alpha > 0:
#             lam = np.random.beta(self.alpha, self.alpha)
#             perm = np.random.permutation(len(Xb))
#             X_mix = lam * Xb + (1 - lam) * Xb[perm]
#             y_mix = lam * yb + (1 - lam) * yb[perm]
#             gate_target_mix = lam * gate_target + (1 - lam) * gate_target[perm]
#             sample_weights_mix = lam * sample_weights + (1 - lam) * sample_weights[perm]
#             return X_mix, {'main_output': y_mix, 'tof_gate': gate_target_mix}, sample_weights_mix

#         return Xb, {'main_output': yb, 'tof_gate': gate_target}, sample_weights