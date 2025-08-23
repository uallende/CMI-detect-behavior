# src/thermal_feats.py

import polars as pl
from pathlib import Path
import gc

# =====================================================================================
# CONFIGURATION
# =====================================================================================
# Directory where the CLEAN base data is stored
INPUT_DIR = Path("output")
# Directory where the new feature set will be saved
EXPORT_DIR = Path("output")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# This script reads from the output of the cleaning script
CLEAN_DATA_FILE = INPUT_DIR / "cleaned_base_train_data.parquet"

# =====================================================================================
# MAIN PROCESSING
# =====================================================================================
if __name__ == "__main__":
    print("▶ Starting Thermal (thm_) Statistical Feature Engineering Script...")

    # --- Step 1: Load CLEAN Data ---
    print(f"  Loading clean base data from '{CLEAN_DATA_FILE}'...")
    try:
        df = pl.read_parquet(CLEAN_DATA_FILE)
    except Exception as e:
        print(f"  ERROR: Could not load the clean base data file. Make sure '{CLEAN_DATA_FILE}' exists.")
        print(f"  Error details: {e}")
        exit()

    # --- Step 2: Define Thermal Columns ---
    thm_cols = [c for c in df.columns if c.startswith('thm_')]
    
    if not thm_cols:
        print("  ERROR: No thermal columns (thm_*) found in the input file. Exiting.")
        exit()
        
    print(f"  Found {len(thm_cols)} thermal columns to process: {thm_cols}")

    # --- Step 3: Calculate Statistical Features ---
    print("  Calculating statistical features across thermal sensors...")
    
    # --- THIS IS THE CORRECTED BLOCK ---
    thermal_feature_exprs = [
        pl.mean_horizontal(pl.col(thm_cols)).alias("thm_mean"),
        # Use the correct idiom for horizontal standard deviation
        pl.concat_list(pl.col(thm_cols)).list.std().alias("thm_std"),
        pl.max_horizontal(pl.col(thm_cols)).alias("thm_max"),
        pl.min_horizontal(pl.col(thm_cols)).alias("thm_min"),
    ]
    
    df = df.with_columns(thermal_feature_exprs)

    # --- Step 4: Select and Save Output ---
    print("  Selecting final columns for output...")
    key_cols = ['sequence_id', 'sequence_counter']
    
    thermal_engineered_cols = [
        "thm_mean",
        "thm_std",
        "thm_max",
        "thm_min",
    ]

    if not all(key in df.columns for key in key_cols):
        print(f"  ERROR: Input file is missing required key columns: {key_cols}. Exiting.")
        exit()

    final_df = df.select(key_cols + thermal_engineered_cols)
    
    # --- Step 5: Verification and Saving ---
    print("\nProcessing complete.")
    print("Final DataFrame Info:")
    print(final_df.head())
    print(f"Shape: {final_df.shape}")

    output_path = EXPORT_DIR / "thermal_features.parquet"
    print(f"\nSaving thermal features to '{output_path}'...")
    final_df.write_parquet(output_path)
    print("Save complete.")
    gc.collect()