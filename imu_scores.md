Architecture:

def create_model(dataset, imu_dim, wd=1e-4):
    sample_batch = next(iter(dataset))
    input_shape = sample_batch[0].shape[1:]
    inp = tf.keras.layers.Input(shape=input_shape)
    imu = tf.keras.layers.Lambda(lambda t: t[:, :, :imu_dim])(inp)

    x = unet_se_cnn(imu, 3, base_filters=128, kernel_size=3)
    x = attention_layer(x) 
    x = tf.keras.layers.Dropout(0.3)(x) 

    main_out = tf.keras.layers.Dense(18, activation="softmax", name="main_output")(x)
    return tf.keras.models.Model(inputs=inp, outputs=main_out)


physics feats only
Per-fold Accuracies: [0.6545632973503435, 0.6717369970559371, 0.6683022571148185, 0.6828669612174767]
Mean Accuracy: 0.6694 ± 0.0101


physics feats + cross-modal feats
Per-fold Accuracies: [0.6776251226692837, 0.662414131501472, 0.6678115799803729, 0.6755031909671085]
Mean Accuracy: 0.6708 ± 0.0061

physics feats + cross-modal feats + rolling stats (10, 50, 100)
Per-fold Accuracies: [0.60...]

physics feats + cross-modal feats + rolling stats (10)
Per-fold Accuracies: [0.64, 0.65, 0.63...]

====================================================================================

def create_model(dataset, imu_dim, wd=1e-4):
    sample_batch = next(iter(dataset))
    input_shape = sample_batch[0].shape[1:]
    inp = tf.keras.layers.Input(shape=input_shape)
    imu = tf.keras.layers.Lambda(lambda t: t[:, :, :imu_dim])(inp)

    x = unet_se_cnn(imu, 5, base_filters=128, kernel_size=3)
    x = attention_layer(x) 
    x = tf.keras.layers.Dropout(0.3)(x) 

    main_out = tf.keras.layers.Dense(18, activation="softmax", name="main_output")(x)
    return tf.keras.models.Model(inputs=inp, outputs=main_out)

physics feats + cross-modal feats
Fold 1 Accuracy: 0.6693
Fold 2 Accuracy: 0.6742

----------------------------------

physics feats + cross-modal feats = no attention layer but GlobalAveragePooling1D
Per-fold Accuracies: [0.6575073601570167, 0.6496565260058881, 0.6496565260058881, 0.6696121747668139]
Mean Accuracy: 0.6566 ± 0.0082

----------------------------------

physics feats + cross-modal feats
    x = unet_se_cnn(imu, 3, base_filters=128, kernel_size=5)
    x = attention_layer(x) 

Per-fold Accuracies: [0.6658488714425908, 0.6746810598626104, 0.6511285574092247, 0.658321060382916]
Mean Accuracy: 0.6625 ± 0.0088

-----------------------------------

physics feats + cross-modal feats
    x = unet_se_cnn(imu, 3, base_filters=172, kernel_size=3)
    x = attention_layer(x) 
Per-fold Accuracies: [0.6658488714425908, 0.6815505397448479, 0.662414131501472, 0.6735395189003437]
Mean Accuracy: 0.6708 ± 0.0074

------------------------------------
physics feats + cross-modal feats
    x = wave_block(imu, 64, 4, 4)
    x = attention_layer(x)
Per-fold Accuracies: [0.64]

-------------------------------------
physics feats + cross-modal feats
    x = wave_block(imu, 64, 3, 8)
    x = attention_layer(x)
Per-fold Accuracies: [0.62]