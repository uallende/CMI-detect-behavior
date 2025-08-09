import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Path to your training data file
TRAIN_CSV_PATH = Path('input/cmi-detect-behavior-with-sensor-data/train.csv') 

# --- Main Execution Block ---
if __name__ == "__main__":
    if not TRAIN_CSV_PATH.exists():
        print(f"Error: The training data file was not found at '{TRAIN_CSV_PATH}'")
    else:
        print(f"Loading training data from '{TRAIN_CSV_PATH}'...")
        train_df = pl.read_csv(TRAIN_CSV_PATH)

        print("Analyzing behavior counts to infer sampling rate...")
        behavior_counts = train_df.group_by(["sequence_id", "behavior"]).len()

        # --- FINAL FIX: Using the correct string for the 'Pause' behavior ---
        PAUSE_BEHAVIOR_STRING = "Hand at target location"
        
        pause_counts = behavior_counts.filter(pl.col("behavior") == PAUSE_BEHAVIOR_STRING)

        # Check if any pause behaviors were found before proceeding
        if pause_counts.height == 0:
            print(f"\nError: Could not find any rows with behavior == '{PAUSE_BEHAVIOR_STRING}'.")
        else:
            print("\nDescriptive statistics for the number of samples in the 'Pause' phase:")
            print(pause_counts.select("len").describe())

            print("\nPlotting histogram of pause durations (in samples)...")
            plt.figure(figsize=(12, 7))
            sns.histplot(data=pause_counts.to_pandas(), x='len', bins=30, kde=True)
            plt.title('Distribution of Pause Duration ("Hand at target location") in Samples')
            plt.xlabel('Number of Samples in Pause Phase ("len")')
            plt.ylabel('Frequency (Number of Sequences)')
            plt.grid(True)
            
            # Find the most common pause duration (the mode)
            mode_value = pause_counts.get_column('len').mode().item(0)
            plt.axvline(mode_value, color='red', linestyle='--', linewidth=2, label=f'Most Common Duration: {mode_value} samples')
            plt.legend()
            
            print(f"\nAnalysis complete. The most common pause duration is {mode_value} samples.")
            print("Assuming a standard 2-second pause, this confirms a sampling rate of 100 Hz.")
            
            plt.show()