==== full dataset ===

def create_model_definition(dataset):
    sample_batch = next(iter(dataset))
    input_shape = sample_batch[0].shape[1:]
    inputs = Input(shape=input_shape)

    x = inputs
    x = res_se_cnn_wave_gru_block(x, filters=32, kernel_size=3, dilation_depth=4)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(18, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model 

Per-fold Accuracies: [0.5879828326180258, 0.6202453987730061, 0.6, 0.5914110429447853, 0.5969325153374233]
Mean Accuracy: 0.5993 ± 0.0113

=== Full dataset Kaggle 0.8 notebook features ===

def create_model_definition(dataset, imu_dim, wd=1e-4):

    x1 = unet_se_cnn(imu, 3, base_filters=64, kernel_size=3)
    x2 = tof_block(tof, wd)

    x = features_processing(x1, x2)
    x = Dropout(0.3)(x) 
    main_out = Dense(18, activation="softmax", name="main_output")(x)
    gate_out = Dense(1, activation="sigmoid", name="tof_gate")(x)
    
    return Model(inputs=inp, outputs={"main_output": main_out, "tof_gate": gate_out})


Per-fold Accuracies: [0.7546614327772326, 0.7507360157016683, 0.767909715407262, 0.7741777123220422]
Mean Accuracy: 0.7619 ± 0.0095 

0.795 in PLB

In Kaggle
=== Cross-validation Summary ===
Per-fold Accuracies: [0.7419038272816487, 0.7708537782139352, 0.7737978410206084, 0.7947962690230731]
Mean Accuracy: 0.7703 ± 0.0188


=== new tof and imu features ===
Fold 4 Accuracy: 0.7727
Per-fold Accuracies: [0.7649656526005888, 0.7850834151128557, 0.7443572129538764, 0.7727049582719686]
Mean Accuracy: 0.7668 ± 0.0148


2 unet blocks 128 base filters
Per-fold Accuracies: [0.7998037291462218, 0.7787046123650638, 0.788027477919529, 0.7677957781050565]
Mean Accuracy: 0.7836 ± 0.0118


0.8 notebook featuresand architecture
Per-fold Accuracies: [0.7595682041216879, 0.7340529931305201, 0.7526987242394504, 0.7550319096710849]
Mean Accuracy: 0.7503 ± 0.0097

0.8 notebook my tof and imu features sand architecture
Per-fold Accuracies: [0.7522080471050049, 0.7713444553483808, 0.7487733071638861, 0.7722140402552774]
Mean Accuracy: 0.7611 ± 0.0107

=============================================================
    xa = unet_se_cnn(imu, 3, base_filters=128, kernel_size=3) # 64,128
    xa = unet_se_cnn(xa, 3, base_filters=128, kernel_size=5)
    # xb = tf.keras.layers.MaxPool1D(2)(xb) # 64,128
    # x1 = tf.keras.layers.Concatenate()([xa, xb])
    # x1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(x1)

    # input_shape=[(None, 64, 256), (None, 32, 128)
    x2 = tof_block_2(tof, wd)
Per-fold Accuracies: [0.7723258096172718, 0.7644749754661433, 0.7811579980372915, 0.7673048600883653]
Mean Accuracy: 0.7713 ± 0.0063












================================= OLD =================================

fraction = 1/6

=== Cross-validation Summary ===

imu_cols

=== ===

Simple CNN

Per-fold Accuracies: [0.3463687150837989, 0.30726256983240224, 0.35195530726256985, 0.31843575418994413, 0.3128491620111732]
Mean Accuracy: 0.3274 ± 0.0182

===  ===

residual_se_cnn_block

Per-fold Accuracies: [0.35447761194029853, 0.30970149253731344, 0.4044943820224719, 0.3408239700374532, 0.3970037453183521]
Mean Accuracy: 0.3613 ± 0.0354

=== ===

wavenet

Per-fold Accuracies: [0.332089552238806, 0.3805970149253731, 0.30711610486891383, 0.299625468164794, 0.3445692883895131]
Mean Accuracy: 0.3328 ± 0.0289

=== ===

simple CNN - self-attention
Per-fold Accuracies: [0.3582089552238806, 0.332089552238806, 0.3146067415730337, 0.3333333333333333, 0.40074906367041196]
Mean Accuracy: 0.3478 ± 0.0299

=== ===

simple CNN - self-attention - gru
Per-fold Accuracies: [0.3656716417910448, 0.3805970149253731, 0.30711610486891383, 0.31086142322097376, 0.3782771535580524]
Mean Accuracy: 0.3485 ± 0.0327

=== ===

simple CNN - gru
Per-fold Accuracies: [0.34328358208955223, 0.35447761194029853, 0.38202247191011235, 0.40074906367041196, 0.3408239700374532]
Mean Accuracy: 0.3643 ± 0.0234

=== ===

    x1 = residual_se_cnn_block(inputs, 32, 3)
    x2 = wave_block(inputs, 32, 3, 4)

Per-fold Accuracies: [0.3246268656716418, 0.39552238805970147, 0.30711610486891383, 0.37453183520599254, 0.3595505617977528]
Mean Accuracy: 0.3523 ± 0.0323

=== ===

res_se_cnn_wave_gru_block

Per-fold Accuracies: [0.3805970149253731, 0.417910447761194, 0.352059925093633, 0.3970037453183521, 0.40823970037453183]
Mean Accuracy: 0.3912 ± 0.0232

=== ===

2 blocks or 4 blocks worse
Per-fold Accuracies: [0.4216417910447761, 0.44402985074626866, 0.29213483146067415, 0.3895131086142322, 0.3258426966292135]
Mean Accuracy: 0.3746 ± 0.0573

