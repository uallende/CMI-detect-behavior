import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import gc # Import gc for memory management

# --- Configuration ---
RAW_DIR = Path("input/cmi-detect-behavior-with-sensor-data")
# RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
EXPORT_DIR = Path("output")
EXPORT_DIR.mkdir(parents=True, exist_ok=True) # Ensure export directory exists

def remove_gravity_from_acc(acc_data, rot_data):
    acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    for i in range(len(acc_values)):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/100): # Corrected sampling rate to 100Hz
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_vel = np.zeros((len(quat_values), 3))
    for i in range(len(quat_values) - 1):
        q_t, q_t_plus_dt = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q_t)) or np.all(np.isnan(q_t_plus_dt)): continue
        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError: pass
    return angular_vel

def calculate_angular_distance(rot_data):
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_dist = np.zeros(len(quat_values))
    for i in range(len(quat_values) - 1):
        q1, q2 = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q1)) or np.all(np.isnan(q2)): continue
        try:
            r1, r2 = R.from_quat(q1), R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angular_dist[i] = np.linalg.norm(relative_rotation.as_rotvec())
        except ValueError: pass
    return angular_dist

# --- Main Processing ---

print("▶ TRAIN MODE – loading dataset …")
df = pd.read_csv(RAW_DIR / "train.csv")
train_dem_df = pd.read_csv(RAW_DIR / "train_demographics.csv")
df = pd.merge(df, train_dem_df, on='subject', how='left')
le = LabelEncoder()
df['gesture_int'] = le.fit_transform(df['gesture'])
np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

print("  Removing gravity and calculating linear acceleration features...")
linear_accel_list = [pd.DataFrame(remove_gravity_from_acc(group[['acc_x', 'acc_y', 'acc_z']], group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]), columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index) for _, group in df.groupby('sequence_id')]
df = pd.concat([df, pd.concat(linear_accel_list)], axis=1)
df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

print("  Calculating angular velocity and distance from quaternions...")
angular_vel_list = [pd.DataFrame(calculate_angular_velocity_from_quat(group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]), columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index) for _, group in df.groupby('sequence_id')]
df = pd.concat([df, pd.concat(angular_vel_list)], axis=1)
angular_dist_list = [pd.DataFrame(calculate_angular_distance(group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]), columns=['angular_distance'], index=group.index) for _, group in df.groupby('sequence_id')]
df = pd.concat([df, pd.concat(angular_dist_list)], axis=1)

# --- Define Feature Columns ---
imu_cols_base = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z'] + [c for c in df.columns if c.startswith('rot_')]
imu_engineered = ['linear_acc_mag', 'linear_acc_mag_jerk', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance']
imu_cols = list(dict.fromkeys(imu_cols_base + imu_engineered))


thm_cols_original = [c for c in df.columns if c.startswith('thm_')]
tof_aggregated_cols_template = []
for i in range(1, 6): tof_aggregated_cols_template.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])

final_feature_cols = imu_cols + thm_cols_original + tof_aggregated_cols_template
metadata_cols = ['sequence_id', 'subject', 'gesture', 'gesture_int']

print(f"  Total {len(final_feature_cols)} features will be engineered.")
np.save(EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))

# --- MODIFICATION START: Build a list of processed DataFrames ---
print("  Building list of processed sequences...")
seq_gp = df.groupby('sequence_id') 
processed_sequences_dfs = [] # Initialize a list to hold processed DataFrames

for seq_id, seq_df in seq_gp:
    seq_df_copy = seq_df.copy()
    
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        tof_data = seq_df_copy[pixel_cols].replace(-1, np.nan)
        seq_df_copy[f'tof_{i}_mean'] = tof_data.mean(axis=1)
        seq_df_copy[f'tof_{i}_std'] = tof_data.std(axis=1)
        seq_df_copy[f'tof_{i}_min'] = tof_data.min(axis=1)
        seq_df_copy[f'tof_{i}_max'] = tof_data.max(axis=1)
        
    seq_df_copy[final_feature_cols] = seq_df_copy[final_feature_cols].ffill().bfill().fillna(0)
    cols_to_keep = metadata_cols + final_feature_cols
    processed_seq = seq_df_copy[cols_to_keep]
    
    # 4. Append the fully processed DataFrame for this sequence to our list
    processed_sequences_dfs.append(processed_seq)

# --- New Final Step: Concatenate all processed sequences into a single DataFrame ---
print("  Concatenating all processed sequences into a single DataFrame...")
final_df = pd.concat(processed_sequences_dfs, ignore_index=True)
gc.collect()

# --- Verification and Saving ---
print("\nProcessing complete. Final DataFrame created.")
print("Final DataFrame Info:")
final_df.info()

print("\nFinal DataFrame Head:")
print(final_df.head())

# Save the final DataFrame to a memory-efficient Parquet file
output_path = EXPORT_DIR / "imu_kaggle_0.8_feats.parquet"
print(f"\nSaving final DataFrame to '{output_path}'...")
final_df.to_parquet(output_path)
print("Save complete.")