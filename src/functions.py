import os
import tensorflow as tf
import numpy as np
import polars as pl
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import argmax
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras.models import Model 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
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
    )

TARGET = ['gesture']

def get_fold(df, n_splits=5):

    unique_sequences_series = df.get_column('sequence_id').unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_sequences_series)):
        train_seqs = unique_sequences_series[train_idx]
        val_seqs = unique_sequences_series[val_idx]
        train_df = df.filter(pl.col('sequence_id').is_in(pl.Series(train_seqs).implode()))
        val_df   = df.filter(pl.col('sequence_id').is_in(pl.Series(val_seqs).implode()))
        yield train_df, val_df

def pl_standard_scaling(train_df, val_df, cols):

    sc = StandardScaler()
    scaled_df = sc.fit_transform(train_df[cols])
    scaled_cols_df = pl.DataFrame(scaled_df, schema=cols)
    train_df_scaled = train_df.drop(cols).hstack(scaled_cols_df)

    scaled_df = sc.transform(val_df[cols])
    scaled_cols_df = pl.DataFrame(scaled_df, schema=cols)
    val_df_scaled = val_df.drop(cols).hstack(scaled_cols_df)

    return train_df_scaled, val_df_scaled        

def perform_label_encoding(cols, df):
    return df.with_columns([pl.col(col).cast(pl.Categorical).to_physical() for col in cols])
    
def perform_target_encoding(train_df, val_df, target=TARGET):
    le = LabelEncoder()
    train_df = train_df.with_columns(pl.col(target).map_batches(le.fit_transform))   
    val_df = val_df.with_columns(pl.col(target).map_batches(le.transform))

    return train_df, val_df, le

def create_sequence_dataset(df: pl.DataFrame, feature_cols: list, gate_df: pl.DataFrame):
    """
    Creates sequences from the DataFrame and aligns them with their labels and TOF gate targets.

    Args:
        df: The main DataFrame containing scaled feature data.
        feature_cols: A list of column names to be used as features for the sequences.
        gate_df: A DataFrame with ['sequence_id', 'has_tof'] mapping.

    Returns:
        A tuple of three NumPy arrays: (sequences, labels, gate_targets).
    """
    sequences = []
    labels = []
    gate_targets = [] 

    df_with_gate = df.join(gate_df, on='sequence_id', how='left')

    for seq_id, group in df_with_gate.group_by('sequence_id', maintain_order=True):
        sequences.append(group.select(feature_cols).to_numpy())
        labels.append(group.select('gesture_int').item(0, 0))
        gate_targets.append(group.select('has_tof').item(0, 0))

    return np.array(sequences, dtype=object), np.array(labels), np.array(gate_targets)


def perform_padding(X, pad_len):
    return pad_sequences(
        X,
        maxlen=pad_len,
        padding='post', # Pad at the end of the sequence
        dtype='float32',# Data type of the output tensor
        truncating='post',
        value=0.0,      # Value to use for padding (e.g., 0 for numerical data)
    )

def smooth_labels(y, num_classes, smoothing=0.1):
    """
    y: (N,) array of integer class labels
    Returns: (N, num_classes) array with smoothed one-hot labels
    """
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (num_classes - 1)
    one_hot = np.full((len(y), num_classes), smoothing_value, dtype=np.float32)
    one_hot[np.arange(len(y)), y] = confidence
    return one_hot


def build_dataset(X, y, pad_len, batch_size=64, num_classes=18, label_smoothing=0.1):
    X = perform_padding(X, pad_len)
    y = smooth_labels(y, num_classes, smoothing=label_smoothing)
    
    ds = Dataset.from_tensor_slices((X, y))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def generate_gate_targets(df: pl.DataFrame, tof_cols: list) -> pl.DataFrame:
    gate_df = df.group_by("sequence_id").agg(
        pl.any_horizontal(pl.col(tof_cols).is_not_null().any()).alias("has_tof")
    )
    return gate_df.with_columns(pl.col("has_tof").cast(pl.Float32))

def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs=50,
    initial_learning_rate=1e-3, # The starting LR for the schedule
    weight_decay=1e-4           # The strength of the L2 regularization
):

    try:
        steps_per_epoch = len(train_dataset)
        total_decay_steps = steps_per_epoch * epochs
        print(f"LR Scheduler: {steps_per_epoch} steps per epoch, {total_decay_steps} total decay steps.")
    except TypeError:
        total_decay_steps = 10000 # A reasonable fallback
        print("Warning: Could not determine dataset length. Using default decay steps.")

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=total_decay_steps,
        alpha=0.0 # The minimum learning rate as a fraction of the initial one. 0.0 means it decays to zero.
    )

    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay
    )

    early_stopping = EarlyStopping(
        monitor='val_main_output_accuracy',
        mode='max',
        patience=20,
        restore_best_weights=True
    )

    # --- 4. Compile the Model ---
    model.compile(
        optimizer=optimizer,
        loss={
        "main_output": CategoricalCrossentropy(label_smoothing=0.1),
        "tof_gate": BinaryCrossentropy()
        },
        loss_weights={
            "main_output": 1.0,
            "tof_gate": 0.2  # tune this
        },
        metrics={"main_output": "accuracy"}
        )

    # --- 5. Fit the Model ---
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping]
    )    

if __name__ == "__main__":
    print("done")    



# def create_sequence_dataset(df, cols, target_col='gesture'):

#     X, y, lengths = [], [], []

#     for _, seq in df.group_by('sequence_id'):
#         data = seq[cols].to_numpy().astype(np.float32)
#         label = seq[target_col].max()

#         X.append(data)
#         y.append(label)
#         lengths.append(len(data))

#     y = np.array(y, dtype=np.int16)

#     return X, y    