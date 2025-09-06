# src/clean_raw_data.py

import polars as pl
from pathlib import Path
import os
import gc

# =====================================================================================
# CONFIGURATION
# =====================================================================================
RAW_DIR = Path("input/cmi-detect-behavior-with-sensor-data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DATA_FILE = OUTPUT_DIR / "cleaned_base_train_data.parquet"
FILTER_PROBLEM_SUBJECTS = True

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

    # --- Step 2: Filtering ---
    print("\n  Applying filtering rules...")

    df = df.filter(pl.col("sequence_id") != "SEQ_011975")
    print(f"  Shape after removing SEQ_011975: {df.shape}")

    df = df.with_columns(
        (pl.col("gesture").is_not_null().sum().over("sequence_id") / pl.len().over("sequence_id")).alias("gesture_ratio")
    ).filter(
        pl.col("gesture_ratio") >= 0.2
    ).drop("gesture_ratio")
    print(f"  Shape after gesture ratio filter (>= 0.2): {df.shape}")

    raw_tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
    df = df.with_columns(
        pl.sum_horizontal(pl.col(c).is_not_null() for c in raw_tof_cols).sum().over("sequence_id").alias("total_non_null_tofs")
    ).filter(
        pl.col("total_non_null_tofs") > 0
    ).drop("total_non_null_tofs")
    print(f"  Shape after removing sequences with no TOF data: {df.shape}")
    
    if FILTER_PROBLEM_SUBJECTS:
        problem_subjects = ["SUBJ_045235", "SUBJ_019262"]
        df = df.filter(~pl.col("subject").is_in(problem_subjects))
        print(f"  Shape after removing problem subjects: {df.shape}")

    # --- Step 3: Value Transformation ---
    print("\n  Applying value transformations...")
    
    df = df.with_columns(
        [pl.when(pl.col(c) == -1).then(500).otherwise(pl.col(c)).alias(c) for c in raw_tof_cols]
    )
    print("  Replaced -1 TOF values with 500.")

    # --- Step 4: Ultimate NaN Filling ---
    print("\n  Performing final imputation sweep...")
    cols_to_impute = [c for c in df.columns if c not in ['row_id', 'sequence_id', 'subject', 'gesture', 'behavior', 'orientation']]
    
    df = df.with_columns(
        pl.col(cols_to_impute)
          .forward_fill()
          .backward_fill()
          .fill_null(0)
          .over("sequence_id")
    )
    print("  Final imputation complete.")

    # --- Step 5: Verification and Saving ---
    print("\n--- Final Cleaned DataFrame ---")
    print(f"  Shape: {df.shape}")
    print("  Head:")
    print(df.head())

    null_counts = df.select(cols_to_impute).null_count().row(0)
    if sum(null_counts) == 0:
        print("\n  ✅ No null values found in the cleaned feature columns.")
    else:
        print("\n  WARNING: Null values still exist in feature columns. Review the cleaning process.")

    print(f"\n  Saving cleaned base dataset to '{CLEAN_DATA_FILE}'...")
    df.write_parquet(CLEAN_DATA_FILE)
    print("  Save complete.")
    gc.collect()