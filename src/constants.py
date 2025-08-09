import os
import polars as pl

def return_data_path():
    if os.path.exists('/kaggle/input/'):
        return '/kaggle/input/cmi-detect-behavior-with-sensor-data/' # Kaggle path
    else:
        return './input/cmi-detect-behavior-with-sensor-data/' # local path

DATA_PATH =  return_data_path()  
df = pl.read_csv_batched(f'{DATA_PATH}train.csv')
df_batch = df.next_batches(10)[0]
meta_cols = {
    'gesture', 'sequence_type', 'behavior', 'orientation',
    'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'
    }
feature_cols = [c for c in df_batch.columns if c not in meta_cols]
tof_cols  = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
imu_cols  = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]     
emb_cols = ['orientation', 'behavior', 'phase']

imu_cols.extend([
    "lin_acc_x", "lin_acc_y", "lin_acc_z", "ang_vel_x", 
    "ang_vel_y", "ang_vel_z", "angular_distance", "enmo"
])

# df = pl.read_parquet(f"{DATA_PATH}/extended_features_df.parquet")
# fraction = 1/3
# n_rows = int(len(df) * fraction)
# df = df[:n_rows]   

if __name__ == '__main__':
    print('done')