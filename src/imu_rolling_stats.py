# src/imu_rolling_stats.py

import polars as pl
from pathlib import Path
import gc
import numpy as np
from scipy.spatial.transform import Rotation as R

# =====================================================================================
# CONFIGURATION
# =====================================================================================
INPUT_DIR = Path('Output')
EXPORT_DIR = Path("output")
CLEAN_DATA_FILE = INPUT_DIR / "cleaned_base_train_data.parquet"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLING_RATE_HZ = 200 

# =====================================================================================
# HELPER FUNCTIONS (for base physics features)
# =====================================================================================
# (Your functions: remove_gravity_polars, calculate_angular_velocity, etc. go here)
# ... (I will include them here for a complete, runnable script)

def remove_gravity_polars(acc_df: pl.DataFrame, rot_df: pl.DataFrame) -> np.ndarray:
    acc_values = acc_df.select(['acc_x', 'acc_y', 'acc_z']).to_numpy()
    quat_values = rot_df.select(['rot_x', 'rot_y', 'rot_z', 'rot_w']).to_numpy()
    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i] = acc_values[i]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i] = acc_values[i] - gravity_sensor_frame
        except ValueError:
            linear_accel[i] = acc_values[i]
    return linear_accel

def calculate_angular_velocity(rot_df: pl.DataFrame, sampling_rate_hz: int) -> np.ndarray:
    quats = rot_df.select(['rot_x', 'rot_y', 'rot_z', 'rot_w']).to_numpy()
    angular_velocity = np.zeros_like(quats[:, :3])
    dt = 1.0 / sampling_rate_hz
    for i in range(1, len(quats)):
        try:
            q1 = R.from_quat(quats[i - 1])
            q2 = R.from_quat(quats[i])
            q_delta = q2 * q1.inv()
            rot_vec = q_delta.as_rotvec()
            angular_velocity[i] = rot_vec / dt
        except ValueError:
            angular_velocity[i] = 0
    return angular_velocity

# =====================================================================================
# MAIN PROCESSING
# =====================================================================================
if __name__ == "__main__":
    print("▶ Starting IMU Rolling Stats Feature Engineering...")

    print(f"  Loading clean base data from '{CLEAN_DATA_FILE}'...")
    df = pl.read_parquet(CLEAN_DATA_FILE)
    
    # Add a sequence counter if it doesn't exist
    if 'sequence_counter' not in df.columns:
        df = df.with_columns(pl.int_range(0, pl.count()).over('sequence_id').alias('sequence_counter'))

    print("  Generating base physics features (linear acc, angular vel)...")
    
    grouped = df.partition_by("sequence_id", maintain_order=True)
    all_feature_dfs = []
    for group in grouped:
        acc_df = group.select(["acc_x", "acc_y", "acc_z"])
        rot_df = group.select(["rot_x", "rot_y", "rot_z", "rot_w"])
        
        feature_df_group = pl.DataFrame({
            "sequence_counter": group["sequence_counter"],
            "sequence_id": group["sequence_id"]
        })
        
        linear_acc = remove_gravity_polars(acc_df, rot_df)
        feature_df_group = feature_df_group.with_columns(
            pl.DataFrame(linear_acc, schema=["linear_acc_x", "linear_acc_y", "linear_acc_z"])
        )
        
        angular_vel = calculate_angular_velocity(rot_df, SAMPLING_RATE_HZ)
        feature_df_group = feature_df_group.with_columns(
            pl.DataFrame(angular_vel, schema=["angular_vel_x", "angular_vel_y", "angular_vel_z"])
        )
        all_feature_dfs.append(feature_df_group)

    features_to_add = pl.concat(all_feature_dfs)
    df = df.join(features_to_add, on=["sequence_id", "sequence_counter"], how="left")
    
    # Calculate magnitudes of the base signals
    df = df.with_columns([
        (pl.col("linear_acc_x")**2 + pl.col("linear_acc_y")**2 + pl.col("linear_acc_z")**2).sqrt().alias("linear_acc_mag"),
        (pl.col("angular_vel_x")**2 + pl.col("angular_vel_y")**2 + pl.col("angular_vel_z")**2).sqrt().alias("angular_vel_mag"),
    ])
    
    print("  Calculating rolling window statistics...")
    
    window_sizes = [10, 50, 100] 
    
    cols_to_roll = [
        "linear_acc_mag", 
        "angular_vel_mag",
        "linear_acc_x",
        "linear_acc_y",
        "linear_acc_z"
    ]
    
    rolling_exprs = []
    for w in window_sizes:
        for col in cols_to_roll:
            rolling_exprs.append(pl.col(col).rolling_mean(window_size=w, min_periods=1).over("sequence_id").alias(f"{col}_roll_mean_{w}"))
            rolling_exprs.append(pl.col(col).rolling_std(window_size=w, min_periods=1).over("sequence_id").alias(f"{col}_roll_std_{w}"))
            rolling_exprs.append(pl.col(col).rolling_max(window_size=w, min_periods=1).over("sequence_id").alias(f"{col}_roll_max_{w}"))
            rolling_exprs.append(pl.col(col).rolling_min(window_size=w, min_periods=1).over("sequence_id").alias(f"{col}_roll_min_{w}"))

    df = df.with_columns(rolling_exprs)
    
    # --- Step 4: Select and Save Output ---
    key_cols = ['sequence_id', 'sequence_counter']
    rolling_feature_cols = [expr.meta.output_name() for expr in rolling_exprs]
    
    final_df = df.select(key_cols + rolling_feature_cols)
    
    print("\nProcessing complete.")
    print("Final DataFrame Info:")
    print(final_df.head())
    print(f"Shape: {final_df.shape}")

    output_path = EXPORT_DIR / "imu_rolling_stats_features.parquet"
    print(f"\nSaving rolling stats features to '{output_path}'...")
    final_df.write_parquet(output_path)
    print("Save complete.")
    gc.collect()