import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import gc
from scipy.stats import skew, kurtosis

# =====================================================================================
# CONFIGURATION
# =====================================================================================
RAW_DIR = Path("input/cmi-detect-behavior-with-sensor-data")
EXPORT_DIR = Path("output")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================================
# HELPER UDFs for Polars (with numerical stability fix)
# =====================================================================================
def pl_skew(s: pl.Series) -> float:
    values = s.to_numpy()
    if len(values) < 3: return None
    if np.std(values) < 1e-9: return 0.0
    return skew(values)

def pl_kurtosis(s: pl.Series) -> float:
    values = s.to_numpy()
    if len(values) < 4: return None
    if np.std(values) < 1e-9: return 0.0
    return kurtosis(values)

# =====================================================================================
# MAIN PROCESSING FUNCTION
# =====================================================================================
def process_tof_features(df: pl.DataFrame) -> pl.DataFrame:
    tof_aggregated_cols_template = []
    for i in range(1, 6): 
        tof_aggregated_cols_template.extend([
            f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max',
            f'tof_{i}_median', f'tof_{i}_diff_mean', f'tof_{i}_mean_decay',
            f'tof_{i}_skew', f'tof_{i}_kurtosis', f'tof_{i}_active_pixels',
            f'tof_{i}_centroid_x', f'tof_{i}_centroid_y'
        ])

    final_feature_cols = tof_aggregated_cols_template
    metadata_cols = ['sequence_id', 'subject', 'gesture', 'gesture_int']

    print(f"  Total {len(final_feature_cols)} ToF statistical features will be engineered.")
    np.save(EXPORT_DIR / "tof_feature_cols_advanced.npy", np.array(final_feature_cols))

    decay_weights = np.power(0.9, np.arange(64))
    x_coords, y_coords = np.meshgrid(np.arange(8), np.arange(8))

    print("  Building and executing feature engineering expressions...")
    feature_expressions = []
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        list_expr = pl.concat_list([pl.when(pl.col(c) == -1).then(None).otherwise(pl.col(c)) for c in pixel_cols]).alias(f"tof_{i}_list")
        feature_expressions.extend([
            list_expr.list.mean().alias(f'tof_{i}_mean'),
            list_expr.list.std().alias(f'tof_{i}_std'),
            list_expr.list.min().alias(f'tof_{i}_min'),
            list_expr.list.max().alias(f'tof_{i}_max'),
            list_expr.list.median().alias(f'tof_{i}_median'),
            list_expr.list.diff().list.mean().alias(f'tof_{i}_diff_mean'),
            list_expr.list.drop_nulls().list.len().alias(f'tof_{i}_active_pixels'),
            list_expr.list.drop_nulls().map_elements(pl_skew, return_dtype=pl.Float64).alias(f'tof_{i}_skew'),
            list_expr.list.drop_nulls().map_elements(pl_kurtosis, return_dtype=pl.Float64).alias(f'tof_{i}_kurtosis'),
        ])
        tof_data_exprs = [pl.when(pl.col(c) == -1).then(None).otherwise(pl.col(c)) for c in pixel_cols]
        feature_expressions.append(pl.sum_horizontal([(expr * weight).fill_null(0) for expr, weight in zip(tof_data_exprs, decay_weights)]).alias(f'tof_{i}_mean_decay'))
        weights_exprs = [(1 / (expr + 1e-6)).fill_null(0) for expr in tof_data_exprs]
        total_weight_expr = pl.sum_horizontal(weights_exprs)
        centroid_x_expr = pl.when(total_weight_expr > 1e-9).then(pl.sum_horizontal([(w * c) for w, c in zip(weights_exprs, x_coords.ravel())]) / total_weight_expr).otherwise(None)
        centroid_y_expr = pl.when(total_weight_expr > 1e-9).then(pl.sum_horizontal([(w * c) for w, c in zip(weights_exprs, y_coords.ravel())]) / total_weight_expr).otherwise(None)
        feature_expressions.extend([centroid_x_expr.alias(f'tof_{i}_centroid_x'), centroid_y_expr.alias(f'tof_{i}_centroid_y')])

    df_featured = df.with_columns(feature_expressions)
    
    # --- FIX: Separate imputation by data type ---
    float_cols = [c for c in final_feature_cols if c.endswith(('_mean', '_std', '_min', '_max', '_median', '_diff_mean', '_mean_decay', '_skew', '_kurtosis', '_centroid_x', '_centroid_y'))]
    int_cols = [c for c in final_feature_cols if c.endswith('_active_pixels')]

    # Apply full imputation chain to float columns
    float_imputation = pl.col(float_cols).replace([np.inf, -np.inf], None).fill_nan(None).forward_fill().backward_fill().fill_null(0).over("sequence_id")
    # Apply simpler imputation to integer columns (they can't be inf/nan)
    int_imputation = pl.col(int_cols).forward_fill().backward_fill().fill_null(0).over("sequence_id")

    final_df_imputed = df_featured.with_columns(float_imputation, int_imputation)

    final_df = final_df_imputed.select(metadata_cols + final_feature_cols)
    return final_df

# =====================================================================================
# SCRIPT EXECUTION BLOCK
# =====================================================================================
if __name__ == "__main__":
    TEST_MODE = False
    print("▶ Starting ToF Feature Engineering Script...")
    print("  Loading raw dataset with Polars…")
    df = pl.read_csv(RAW_DIR / "train.csv")
    train_dem_df = pl.read_csv(RAW_DIR / "train_demographics.csv")
    df = df.join(train_dem_df, on='subject', how='left')
    le = LabelEncoder()
    gesture_encoded = le.fit_transform(df.get_column('gesture'))
    df = df.with_columns(pl.Series("gesture_int", gesture_encoded))
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    if TEST_MODE:
        print("\n" + "="*50 + "\n▶ RUNNING IN TEST MODE\n" + "="*50)
        test_sequence_ids = df.get_column("sequence_id").unique().head(2)
        print(f"  Filtering data to 2 sequences: {test_sequence_ids.to_list()}")
        df_subset = df.filter(pl.col("sequence_id").is_in(test_sequence_ids.to_list()))
        final_df = process_tof_features(df_subset)
    else:
        print("\n" + "="*50 + "\n▶ RUNNING IN FULL MODE\n" + "="*50)
        final_df = process_tof_features(df)

    print("\n--- Verifying Null and NaN counts in final DataFrame ---")
    has_missing_values = False
    for col_name in final_df.columns:
        null_count = final_df[col_name].null_count()
        nan_count = 0
        if final_df[col_name].dtype.is_float():
            nan_count = final_df[col_name].is_nan().sum()
        if null_count > 0 or nan_count > 0:
            print(f"  WARNING: Column '{col_name}' has {null_count} nulls and {nan_count} NaNs.")
            has_missing_values = True
    if not has_missing_values:
        print("  ✅ No null or NaN values found in the final DataFrame.")

    print("\nProcessing complete.")
    print("Final DataFrame Info:")
    print(final_df.head())
    print(f"Shape: {final_df.shape}")

    if not TEST_MODE:
        output_path = EXPORT_DIR / "tof_features_advanced_train_polars.parquet"
        print(f"\nSaving final DataFrame to '{output_path}'...")
        final_df.write_parquet(output_path)
        print("Save complete.")
    else:
        print("\nNOTE: TEST_MODE is True. Output file was not saved.")
    gc.collect()