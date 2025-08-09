import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import gc
from scipy.stats import skew, kurtosis

# --- Configuration ---
RAW_DIR = Path("input/cmi-detect-behavior-with-sensor-data")
EXPORT_DIR = Path("output")
EXPORT_DIR.mkdir(parents=True, exist_ok=True) # Ensure export directory exists

# --- Main Processing ---

print("▶ TRAIN MODE – loading dataset …")
df = pd.read_csv(RAW_DIR / "train.csv")
train_dem_df = pd.read_csv(RAW_DIR / "train_demographics.csv")
df = pd.merge(df, train_dem_df, on='subject', how='left')

# Encode the gesture label
le = LabelEncoder()
df['gesture_int'] = le.fit_transform(df['gesture'])
np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

# --- Define Feature and Metadata Columns ---
tof_aggregated_cols_template = []
for i in range(1, 6): 
    tof_aggregated_cols_template.extend([
        f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max',
        f'tof_{i}_median', f'tof_{i}_diff_mean', f'tof_{i}_mean_decay',
        f'tof_{i}_skew', f'tof_{i}_kurtosis', f'tof_{i}_active_pixels',
        f'tof_{i}_centroid_x', f'tof_{i}_centroid_y'
    ])

final_feature_cols = tof_aggregated_cols_template
metadata_cols = ['sequence_id', 'subject', 'gesture', 'gesture_int']

print(f"  Total {len(final_feature_cols)} ToF statistical features will be engineered.")
np.save(EXPORT_DIR / "tof_feature_cols_advanced.npy", np.array(final_feature_cols))

# --- Pre-calculate weights and coordinates for efficiency ---
# For mean_decay (based on 64 pixels)
decay_weights = np.power(0.9, np.arange(64))
# For centroid calculation (8x8 grid)
x_coords, y_coords = np.meshgrid(np.arange(8), np.arange(8))

# --- Build a list of processed DataFrames ---
print("  Building list of processed sequences...")
seq_gp = df.groupby('sequence_id') 
processed_sequences_dfs = []

for seq_id, seq_df in seq_gp:
    # Make a copy to avoid SettingWithCopyWarning
    seq_df_copy = seq_df.copy()
    
    # Calculate aggregated ToF features for the current sequence
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        # Replace -1 with NaN to correctly calculate stats
        tof_data = seq_df_copy[pixel_cols].replace(-1, np.nan)
        
        # --- Calculate all desired statistics ---
        # Original set
        seq_df_copy[f'tof_{i}_mean'] = tof_data.mean(axis=1)
        seq_df_copy[f'tof_{i}_std'] = tof_data.std(axis=1)
        seq_df_copy[f'tof_{i}_min'] = tof_data.min(axis=1)
        seq_df_copy[f'tof_{i}_max'] = tof_data.max(axis=1)
        seq_df_copy[f'tof_{i}_median'] = tof_data.median(axis=1)
        seq_df_copy[f'tof_{i}_diff_mean'] = tof_data.diff(axis=1).mean(axis=1)
        seq_df_copy[f'tof_{i}_mean_decay'] = (tof_data * decay_weights).sum(axis=1)
        
        # --- New suggested set ---
        # Note: .values is needed to pass a clean NumPy array to scipy functions
        seq_df_copy[f'tof_{i}_skew'] = skew(tof_data.values, axis=1, nan_policy='omit')
        seq_df_copy[f'tof_{i}_kurtosis'] = kurtosis(tof_data.values, axis=1, nan_policy='omit')
        seq_df_copy[f'tof_{i}_active_pixels'] = tof_data.notna().sum(axis=1)
        
        # Centroid calculation
        # We invert the distance so that closer objects have higher weight
        # Adding a small epsilon to avoid division by zero
        weights = 1 / (tof_data + 1e-6) 
        total_weight = weights.sum(axis=1)
        
        # Calculate weighted average of x and y coordinates
        centroid_x = (weights * x_coords.ravel()).sum(axis=1) / total_weight
        centroid_y = (weights * y_coords.ravel()).sum(axis=1) / total_weight
        
        seq_df_copy[f'tof_{i}_centroid_x'] = centroid_x
        seq_df_copy[f'tof_{i}_centroid_y'] = centroid_y
        
    seq_df_copy[final_feature_cols] = seq_df_copy[final_feature_cols].ffill().bfill().fillna(0)
    
    cols_to_keep = metadata_cols + final_feature_cols
    processed_seq = seq_df_copy[cols_to_keep]
    processed_sequences_dfs.append(processed_seq)

# --- Concatenate all processed sequences into a single DataFrame ---
print("  Concatenating all processed sequences into a single DataFrame...")
final_df = pd.concat(processed_sequences_dfs, ignore_index=True)
gc.collect()

# --- Verification and Saving ---
print("\nProcessing complete. Final advanced ToF feature DataFrame created.")
print("Final DataFrame Info:")
final_df.info(verbose=False, show_counts=True) # Concise info

print("\nFinal DataFrame Head:")
print(final_df.head())

output_path = EXPORT_DIR / "tof_features_advanced_train.parquet"
print(f"\nSaving final DataFrame to '{output_path}'...")
final_df.to_parquet(output_path)
print("Save complete.")