# src/clean_raw_data.py

import polars as pl
from pathlib import Path
import os
import gc

# =====================================================================================
# CONFIGURATION
# =====================================================================================
# Directory where raw competition data is stored
RAW_DIR = Path("input/cmi-detect-behavior-with-sensor-data")
# Directory where the clean output will be saved
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The name of our new, clean base file
CLEAN_DATA_FILE = OUTPUT_DIR / "cleaned_base_train_data.parquet"

# =====================================================================================
# MAIN CLEANING LOGIC
# =====================================================================================
if __name__ == "__main__":
    print("▶ Starting Raw Data Cleaning Script...")

    # --- Step 1: Load and Merge Raw Data ---
    print("  Loading raw train.csv and demographics.csv...")
    df = pl.read_csv(RAW_DIR / "train.csv")
    demographics_df = pl.read_csv(RAW_DIR / "train_demographics.csv")
    
    df = df.join(demographics_df, on='subject', how='left')
    print(f"  Initial merged shape: {df.shape}")

    # --- Step 2: Robust Cleaning and Imputation ---
    print("  Cleaning and imputing missing values...")
    
    # Define all columns that contain sensor readings
    sensor_cols = [c for c in df.columns if c.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
    
    # Create a list of cleaning expressions
    cleaning_expressions = [
        # Replace -1 with null ONLY in sensor columns
        pl.when(pl.col(c) == -1).then(None).otherwise(pl.col(c)).alias(c)
        for c in sensor_cols
    ]
    
    # Apply the replacement
    df = df.with_columns(cleaning_expressions)
    
    # Now, impute all nulls (from original data or from our -1 replacement)
    # This is done per-sequence to prevent data leakage.
    # We apply this to all feature columns, which now includes the raw sensor data.
    all_feature_cols = [c for c in df.columns if c not in ['row_id', 'sequence_id', 'subject', 'gesture', 'behavior', 'orientation']]
    
    df = df.with_columns(
        pl.col(all_feature_cols)
          .forward_fill()
          .backward_fill()
          .fill_null(0)
          .over("sequence_id")
    )
    
    print("  Cleaning complete.")

    # --- Step 3: Verification and Saving ---
    print("\n--- Final Cleaned DataFrame ---")
    print(f"  Shape: {df.shape}")
    print("  Head:")
    print(df.head())

    # Verify that there are no more nulls
    null_counts = df.null_count().row(0)
    if sum(null_counts) == 0:
        print("\n  ✅ No null values found in the cleaned DataFrame.")
    else:
        print("\n  WARNING: Null values still exist. Review the cleaning process.")

    print(f"\n  Saving cleaned base dataset to '{CLEAN_DATA_FILE}'...")
    df.write_parquet(CLEAN_DATA_FILE)
    print("  Save complete.")
    gc.collect()