import polars as pl
from pathlib import Path
import gc

# --- Configuration ---
FEATURE_DIR = Path("output")
IMU_FEATURES_FILE = FEATURE_DIR / "full_features_df.parquet" 
TOF_FEATURES_FILE = FEATURE_DIR / "tof_features_advanced_train.parquet"
FINAL_DATASET_FILE = FEATURE_DIR / "final_model_input_train.parquet"

# --- Main Merging Logic ---

print("▶ Loading pre-computed feature sets...")

# 1. Load the IMU features DataFrame
# This file contains all original columns plus the new IMU features.
print(f"Loading IMU features from '{IMU_FEATURES_FILE}'...")
imu_df = pl.read_parquet(IMU_FEATURES_FILE)
print("IMU features loaded.")

# 2. Load the advanced ToF features DataFrame
# This file contains metadata and the new ToF features.
print(f"Loading ToF features from '{TOF_FEATURES_FILE}'...")
tof_df = pl.read_parquet(TOF_FEATURES_FILE)
print("ToF features loaded.")

# --- Identify columns for the merge ---
# The ToF dataframe has the metadata and the ToF features.
# We need to get ONLY the new IMU features from the imu_df to avoid duplicating columns.
tof_cols = [c for c in tof_df.columns if c.startswith('tof_')]
imu_feature_cols = [
    'acc_mag', 'acc_mag_jerk', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_accel_x',
    'angular_accel_y', 'angular_accel_z', 'grav_orient_x', 'grav_orient_y',
    'grav_orient_z', 'linear_acc_mag', 'angular_vel_mag', 'angular_accel_mag',
    'linear_acc_mag_jerk', 'angular_vel_mag_jerk', 'angular_accel_mag_jerk'
]
# Also keep the original IMU columns if you need them
original_imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']

# Define the key columns for joining
key_cols = ['sequence_id', 'sequence_counter']

# Select only the keys and the desired feature columns from the IMU dataframe
imu_features_to_join = imu_df.select(key_cols + original_imu_cols + imu_feature_cols)


print("\nVerifying shapes before merge:")
print(f"IMU features to join shape: {imu_features_to_join.shape}")
print(f"ToF DataFrame shape:        {tof_df.shape}")

# --- The Merge Operation ---
# We use an 'inner' merge on both sequence_id and sequence_counter.
# This ensures we only keep rows that exist in both datasets and are perfectly aligned.
# The `tof_df` has the primary metadata, so we use it as the "left" DataFrame.
print("\nMerging the two feature sets...")
final_df = tof_df.join(
    imu_features_to_join, 
    on=key_cols,
    how='inner'
)
gc.collect()

print("Merge complete.")

# --- Verification and Saving ---
print("\nFinal Combined DataFrame Info:")
final_df.info(verbose=False, show_counts=True)

print(f"\nFinal Combined DataFrame shape: {final_df.shape}")
print("\nFinal Combined DataFrame Head:")
print(final_df.head())

expected_cols = len(tof_df.columns) + len(imu_features_to_join.columns) - len(key_cols)
print(f"\nExpected column count: {expected_cols}. Actual: {len(final_df.columns)}")

print(f"\nSaving final combined DataFrame to '{FINAL_DATASET_FILE}'...")
final_df.write_parquet(FINAL_DATASET_FILE)
print("Save complete. This file is now ready for your modeling pipeline.")