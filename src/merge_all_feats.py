import polars as pl
from pathlib import Path
import gc
import numpy as np

# =====================================================================================
# CONFIGURATION
# =====================================================================================
# --- Input Paths ---
# Assumes your feature scripts save to the 'output' directory
FEATURE_DIR = Path("output")

IMU_FEATURES_FILE = FEATURE_DIR / "imu_features_plus_raw_others.parquet" 
TOF_FEATURES_FILE = FEATURE_DIR / "tof_features_advanced_train_polars.parquet"
FINAL_DATASET_FILE = FEATURE_DIR / "final_model_input_dataset.parquet"

# =====================================================================================
# MAIN MERGING LOGIC
# =====================================================================================
if __name__ == "__main__":
    print("▶ Starting Feature Merging Script...")

    # --- Step 1: Load Both Feature Sets ---
    print(f"  Loading IMU features from: {IMU_FEATURES_FILE}")
    df_imu = pl.read_parquet(IMU_FEATURES_FILE)
    
    print(f"  Loading ToF features from: {TOF_FEATURES_FILE}")
    df_tof = pl.read_parquet(TOF_FEATURES_FILE)

    print("\nInitial DataFrame Shapes:")
    print(f"  IMU DataFrame: {df_imu.shape}")
    print(f"  ToF DataFrame: {df_tof.shape}")

    # --- Step 2: Select the Columns to Keep from Each DataFrame ---
    
    tof_feature_cols = np.load(FEATURE_DIR / "tof_feature_cols_advanced.npy", allow_pickle=True).tolist()
    metadata_cols = ['sequence_id', 'sequence_counter', 'subject', 'gesture', 'gesture_int']
    
    # Ensure sequence_counter is present for a robust join
    if 'sequence_counter' not in df_tof.columns:
        df_tof = df_tof.with_columns(pl.int_range(0, pl.count()).over('sequence_id').alias('sequence_counter'))
        
    df_tof_selected = df_tof.select(metadata_cols + tof_feature_cols)

    # From the IMU DataFrame, we ONLY want the new, engineered physics features and the keys.
    key_cols = ['sequence_id', 'sequence_counter']
    imu_engineered_cols = [
        'acc_mag', 'acc_mag_jerk', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_accel_x',
        'angular_accel_y', 'angular_accel_z', 'grav_orient_x', 'grav_orient_y',
        'grav_orient_z', 'linear_acc_mag', 'angular_vel_mag', 'angular_accel_mag',
        'linear_acc_mag_jerk', 'angular_vel_mag_jerk', 'angular_accel_mag_jerk'
    ]
    
    # Select only the columns we need to join
    df_imu_selected = df_imu.select(key_cols + imu_engineered_cols)

    print("\nDataFrame Shapes After Column Selection:")
    print(f"  Selected IMU DataFrame: {df_imu_selected.shape}")
    print(f"  Selected ToF DataFrame: {df_tof_selected.shape}")

    # --- Step 3: Merge the Two DataFrames ---
    print("\n  Merging the two feature sets on 'sequence_id' and 'sequence_counter'...")
    
    # Use an inner join to ensure perfect alignment
    final_df = df_tof_selected.join(
        df_imu_selected,
        on=key_cols,
        how='inner'
    )
    
    gc.collect()
    print("  Merge complete.")

    # --- Step 4: Verification and Saving ---
    print("\n--- Final Combined DataFrame ---")
    print(f"  Shape: {final_df.shape}")
    print("  Head:")
    print(final_df.head())

    # Verify column count
    expected_cols = len(df_tof_selected.columns) + len(df_imu_selected.columns) - len(key_cols)
    print(f"\n  Expected column count: {expected_cols}. Actual: {len(final_df.columns)}")
    if len(final_df.columns) != expected_cols:
        print("  WARNING: Column count mismatch. Check for duplicate column names.")

    print(f"\n  Saving final dataset to '{FINAL_DATASET_FILE}'...")
    final_df.write_parquet(FINAL_DATASET_FILE)
    print("  Save complete.")