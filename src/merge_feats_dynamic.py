# src/merge_features.py

import polars as pl
from pathlib import Path
import gc
from typing import List
from sklearn.preprocessing import LabelEncoder
import numpy as np

# =====================================================================================
# CONFIGURATION
# =====================================================================================
INPUT_DIR = Path("output")
EXPORT_DIR = Path("output")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_DATA_FILE = INPUT_DIR / "cleaned_base_train_data.parquet"

# =====================================================================================
# CORE MERGING FUNCTION (This remains unchanged)
# =====================================================================================
def merge_feature_sets(base_df: pl.DataFrame, feature_file_paths: List[Path]) -> pl.DataFrame:

    print("▶ Starting merge process...")
    final_df = base_df
    key_cols = ['sequence_id', 'sequence_counter']
    
    # Get the list of columns already in the base_df to avoid joining them again
    base_columns = set(base_df.columns)

    for f_path in feature_file_paths:
        print(f"  Loading and joining features from: {f_path.name}")
        
        feature_df = pl.read_parquet(f_path)
        
        if not all(key in feature_df.columns for key in key_cols):
            raise ValueError(f"File {f_path.name} is missing required key columns: {key_cols}")
            
        new_feature_cols = [
            col for col in feature_df.columns 
            if col not in base_columns or col in key_cols
        ]
        
        feature_df_to_join = feature_df.select(new_feature_cols)
        
        final_df = final_df.join(
            feature_df_to_join,
            on=key_cols,
            how='inner'
        )
        gc.collect()

    print("  Merge complete.")
    return final_df

# =====================================================================================
# SCRIPT EXECUTION BLOCK
# =====================================================================================
if __name__ == "__main__":
        
    files_to_merge = [
        "imu_physics_features.parquet",
        "imu_rolling_stats_features.parquet",
        "imu_cross_modal_features.parquet",
        "tof_features_advanced_train_polars.parquet",
    ]
    
    feature_paths = [FEATURE_DIR / f for f in files_to_merge]
    FINAL_DATASET_FILE = FEATURE_DIR / "final_full_feature_dataset.parquet"

    print("  Loading base DataFrame with metadata from raw CSV files...")
    base_df = pl.read_csv(RAW_DIR / "train.csv")
    demographics_df = pl.read_csv(RAW_DIR / "train_demographics.csv")
    
    base_df = base_df.join(demographics_df, on='subject', how='left')
    
    print("  Performing label encoding for 'gesture'...")
    le = LabelEncoder()
    gesture_encoded = le.fit_transform(base_df.get_column('gesture'))
    base_df = base_df.with_columns(pl.Series("gesture_int", gesture_encoded))
    np.save(FEATURE_DIR / "gesture_classes.npy", le.classes_)
    
    metadata_cols = ['sequence_id', 'sequence_counter', 'subject', 'gesture', 'gesture_int']
    base_df = base_df.select(metadata_cols)
    
    print(f"  Base DataFrame created with shape: {base_df.shape}")

    # --- Run the Merging Function ---
    final_df = merge_feature_sets(base_df, feature_paths)

    # --- Verification and Saving ---
    print("\n--- Final Combined DataFrame ---")
    print(f"  Shape: {final_df.shape}")
    print("  Head:")
    print(final_df.head())

    print(f"\n  Saving final dataset to '{FINAL_DATASET_FILE}'...")
    final_df.write_parquet(FINAL_DATASET_FILE)
    print("  Save complete.")