# src/imu_cross_modal_feats.py

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
    print("▶ Starting IMU Cross-Modal Feature Engineering...")

    print(f"  Loading clean base data from '{CLEAN_DATA_FILE}'...")
    df = pl.read_parquet(CLEAN_DATA_FILE)

    # --- Step 2: Generate Base Physics Features ---
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
    
    df = df.with_columns([
        (pl.col("linear_acc_x")**2 + pl.col("linear_acc_y")**2 + pl.col("linear_acc_z")**2).sqrt().alias("linear_acc_mag"),
        (pl.col("angular_vel_x")**2 + pl.col("angular_vel_y")**2 + pl.col("angular_vel_z")**2).sqrt().alias("angular_vel_mag"),
    ])
    
    # --- Step 3: Calculate Cross-Modal Features ---
    print("  Calculating cross-modal ratio features...")
    
    cross_modal_exprs = [
        # Add a small epsilon to prevent division by zero
        (pl.col("linear_acc_mag") / (pl.col("angular_vel_mag") + 1e-6)).alias("linear_to_angular_ratio"),
        (pl.col("linear_acc_z").abs() / (pl.col("linear_acc_x").abs() + pl.col("linear_acc_y").abs() + 1e-6)).alias("vertical_to_horizontal_ratio")
    ]
    
    df = df.with_columns(cross_modal_exprs)
    
    # --- Step 4: Select and Save Output ---
    key_cols = ['sequence_id', 'sequence_counter']
    cross_modal_feature_cols = [expr.meta.output_name() for expr in cross_modal_exprs]
    
    final_df = df.select(key_cols + cross_modal_feature_cols)
    
    print("\nProcessing complete.")
    print("Final DataFrame Info:")
    print(final_df.head())
    print(f"Shape: {final_df.shape}")

    output_path = EXPORT_DIR / "imu_cross_modal_features.parquet"
    print(f"\nSaving cross-modal features to '{output_path}'...")
    final_df.write_parquet(output_path)
    print("Save complete.")
    gc.collect()