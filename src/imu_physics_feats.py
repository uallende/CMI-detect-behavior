# src/physics_feats.py

import os
import polars as pl
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import gc

# =====================================================================================
# CONFIGURATION
# =====================================================================================
RAW_DIR = Path("input/cmi-detect-behavior-with-sensor-data")
EXPORT_DIR = Path("output")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLING_RATE_HZ = 200 

# =====================================================================================
# HELPER FUNCTIONS (for base physics features)
# =====================================================================================

def remove_gravity_polars(acc_df: pl.DataFrame, rot_df: pl.DataFrame) -> np.ndarray:
    """Removes the gravity component from accelerometer data using quaternion rotations."""
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
    """Calculates angular velocity from quaternion data."""
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

def calculate_angular_acceleration(angular_velocity: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
    """Calculates angular acceleration from angular velocity."""
    angular_accel = np.zeros_like(angular_velocity)
    dt = 1.0 / sampling_rate_hz
    angular_accel[1:] = np.diff(angular_velocity, axis=0) / dt
    return angular_accel

def calculate_gravity_orientation(rot_df: pl.DataFrame) -> np.ndarray:
    """Calculates the orientation of each sensor axis with respect to the world gravity vector."""
    quat_values = rot_df.select(['rot_x', 'rot_y', 'rot_z', 'rot_w']).to_numpy()
    num_samples = quat_values.shape[0]
    orientation_angles = np.zeros((num_samples, 3))
    gravity_world = np.array([0, 0, 1.0])
    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            sensor_axes_world = rotation.apply(np.eye(3))
            for j in range(3):
                dot_product = np.dot(sensor_axes_world[j], gravity_world)
                orientation_angles[i, j] = np.arccos(np.clip(dot_product, -1.0, 1.0))
        except ValueError:
            continue
    return orientation_angles

# =====================================================================================
# MAIN PROCESSING FUNCTION
# =====================================================================================
def add_all_imu_features_polars(df: pl.DataFrame, sampling_rate_hz: int) -> pl.DataFrame:
    """Main function to add all IMU features to the DataFrame."""
    df = df.sort(["sequence_id", "sequence_counter"])
    df = df.with_columns(
        (pl.col("acc_x")**2 + pl.col("acc_y")**2 + pl.col("acc_z")**2).sqrt().alias("acc_mag"),
    )
    df = df.with_columns(
        pl.col("acc_mag").diff().over("sequence_id").fill_null(0).alias("acc_mag_jerk"),
    )

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
        angular_vel = calculate_angular_velocity(rot_df, sampling_rate_hz)
        feature_df_group = feature_df_group.with_columns(
            pl.DataFrame(angular_vel, schema=["angular_vel_x", "angular_vel_y", "angular_vel_z"])
        )
        angular_accel = calculate_angular_acceleration(angular_vel, sampling_rate_hz)
        feature_df_group = feature_df_group.with_columns(
            pl.DataFrame(angular_accel, schema=["angular_accel_x", "angular_accel_y", "angular_accel_z"])
        )
        gravity_orientation = calculate_gravity_orientation(rot_df)
        feature_df_group = feature_df_group.with_columns(
            pl.DataFrame(gravity_orientation, schema=["grav_orient_x", "grav_orient_y", "grav_orient_z"])
        )
        all_feature_dfs.append(feature_df_group)

    if all_feature_dfs:
        features_to_add = pl.concat(all_feature_dfs)
        df = df.join(features_to_add, on=["sequence_id", "sequence_counter"], how="left")

    df = df.with_columns([
        (pl.col("linear_acc_x")**2 + pl.col("linear_acc_y")**2 + pl.col("linear_acc_z")**2).sqrt().alias("linear_acc_mag"),
        (pl.col("angular_vel_x")**2 + pl.col("angular_vel_y")**2 + pl.col("angular_vel_z")**2).sqrt().alias("angular_vel_mag"),
        (pl.col("angular_accel_x")**2 + pl.col("angular_accel_y")**2 + pl.col("angular_accel_z")**2).sqrt().alias("angular_accel_mag"),
    ])
    df = df.with_columns([
        pl.col("linear_acc_mag").diff().over("sequence_id").fill_null(0).alias("linear_acc_mag_jerk"),
        pl.col("angular_vel_mag").diff().over("sequence_id").fill_null(0).alias("angular_vel_mag_jerk"),
        pl.col("angular_accel_mag").diff().over("sequence_id").fill_null(0).alias("angular_accel_mag_jerk"),
    ])
    return df

# =====================================================================================
# SCRIPT EXECUTION BLOCK
# =====================================================================================
if __name__ == "__main__":
    print("▶ Starting IMU Physics-Based Feature Engineering Script...")

    # --- Load Data ---
    print("  Loading raw data...")
    df = pl.read_csv(RAW_DIR / "train.csv")
    
    # Ensure a sequence counter exists for robust joining
    if 'sequence_counter' not in df.columns:
        df = df.with_columns(pl.int_range(0, pl.count()).over('sequence_id').alias('sequence_counter'))

    # --- Process Features ---
    print("  Calculating all IMU features...")
    processed_df = add_all_imu_features_polars(df, SAMPLING_RATE_HZ)

    # --- Select Final Columns for Output ---
    print("  Selecting final columns for output...")
    key_cols = ['sequence_id', 'sequence_counter']
    
    # Define the list of all newly engineered features to keep
    imu_engineered_cols = [
        'acc_mag', 'acc_mag_jerk', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_accel_x',
        'angular_accel_y', 'angular_accel_z', 'grav_orient_x', 'grav_orient_y',
        'grav_orient_z', 'linear_acc_mag', 'angular_vel_mag', 'angular_accel_mag',
        'linear_acc_mag_jerk', 'angular_vel_mag_jerk', 'angular_accel_mag_jerk'
    ]
    # Remove duplicates while preserving order
    imu_engineered_cols = list(dict.fromkeys(imu_engineered_cols))

    final_df = processed_df.select(key_cols + imu_engineered_cols)
    
    # --- Verification and Saving ---
    print("\nProcessing complete.")
    print("Final DataFrame Info:")
    print(final_df.head())
    print(f"Shape: {final_df.shape}")

    output_path = EXPORT_DIR / "imu_physics_feats.parquet"
    print(f"\nSaving IMU features to '{output_path}'...")
    final_df.write_parquet(output_path)
    print("Save complete.")
    gc.collect()