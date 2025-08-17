import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.models import CandleData
from src.constants import DATA_REPORTS_DIR

def analyze_and_save_report(data: CandleData):
    """
    Analyzes candle data for gaps, generates plots, and saves a full report.

    This function serves as the main entry point for the analysis module.
    """
    df = data.df.copy()
    if df.empty:
        print("Analysis skipped: The provided DataFrame is empty.")
        return

    # --- 1. Define Output Path ---
    start_time_str = df.index.min().strftime('%Y%m%d')
    end_time_str = df.index.max().strftime('%Y%m%d')
    report_name = f"{start_time_str}_{end_time_str}"

    output_dir = (DATA_REPORTS_DIR / data.source / data.instrument /
                  data.timeframe.pathname / report_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📈 Starting analysis... Report will be saved to: {output_dir}")

    # --- 2. Perform Gap Analysis ---
    granularity_minutes = data.timeframe.minutes
    df['time_diff'] = df.index.to_series().diff().dt.total_seconds() / 60.0
    df.fillna({'time_diff': granularity_minutes}, inplace=True)

    # Define tolerances for expected gaps
    std_dev_tolerance = 0.1 * granularity_minutes # 10% tolerance for standard gaps
    weekend_minutes = 2 * 24 * 60  # Approx. 48 hours for a weekend

    is_standard_gap = np.isclose(df['time_diff'], granularity_minutes, atol=std_dev_tolerance)
    is_weekend_gap = np.isclose(df['time_diff'], weekend_minutes + granularity_minutes, atol=granularity_minutes * 2)

    # Identify unusual gaps (not standard, not weekend, and not the first candle)
    missing_rows_mask = (df['time_diff'] > 0) & ~is_standard_gap & ~is_weekend_gap
    missing_rows = df[missing_rows_mask]

    # --- 3. Save Report and Data ---
    # Save the data itself in the report folder (using Parquet)
    df.drop(columns=['time_diff']).to_parquet(output_dir / "data.parquet")

    # Save the text report
    _save_gap_report_text(output_dir, data.instrument, granularity_minutes, missing_rows)

    # --- 4. Generate and Save Plots ---
    _plot_price_with_gaps(output_dir, df, missing_rows, data.instrument, granularity_minutes)
    _plot_volume_distribution(output_dir, df, data.instrument, granularity_minutes)

    print("✅ Analysis complete.")


def _save_gap_report_text(output_dir: Path, instrument: str, granularity_minutes: float, missing_rows: pd.DataFrame):
    """Generates and saves the missing data text file."""
    report_path = output_dir / "gap_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Analysis of time gaps for {instrument} at {granularity_minutes}-minute granularity.\n")
        f.write("-" * 50 + "\n")

        if not missing_rows.empty:
            total_missing_candles = 0
            for timestamp, row in missing_rows.iterrows():
                num_missing = (row['time_diff'] / granularity_minutes) - 1
                if num_missing > 0.1:  # Only report significant gaps
                    f.write(f"-> Unusual gap of {row['time_diff']:.2f} minutes detected before {timestamp}.\n")
                    f.write(f"   Estimated ~{num_missing:.2f} missing candles.\n\n")
                    total_missing_candles += num_missing
            f.write(f"\nTotal estimated missing candles from unusual gaps: {total_missing_candles:.2f}\n")
        else:
            f.write("No significant unexpected data gaps were found.\n")
    print(f"📄 Gap analysis report saved to {report_path.name}")


def _plot_price_with_gaps(output_dir: Path, df: pd.DataFrame, missing_rows: pd.DataFrame, instrument: str,
                          granularity_minutes: float):
    """Generates and saves a plot of close prices highlighting data gaps."""
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['close_bid'], label='Bid Close', color='deepskyblue', linewidth=1)

    if not missing_rows.empty:
        plt.scatter(missing_rows.index, missing_rows['close_bid'], color='red', marker='x', s=60, zorder=5,
                    label='Point After Unusual Gap')

    plt.title(f"{instrument} Close Prices ({granularity_minutes} Min) with Gaps Highlighted")
    plt.xlabel('Date (UTC)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_path = output_dir / "price_with_gaps_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Price plot saved to {plot_path.name}")


def _plot_volume_distribution(output_dir: Path, df: pd.DataFrame, instrument: str, granularity_minutes: float):
    """Generates and saves a histogram of trade volume."""
    if 'volume' in df.columns and not df['volume'].empty:
        # Filter out extreme outliers for a more informative plot
        volume_99th = df['volume'].quantile(0.99)
        volume_to_plot = df[df['volume'] <= volume_99th]['volume']

        if not volume_to_plot.empty:
            plt.figure(figsize=(12, 6))
            plt.hist(volume_to_plot, bins=80, color='teal', alpha=0.8)
            plt.title(f"{instrument} Volume Distribution ({granularity_minutes} Min) - up to 99th percentile")
            plt.xlabel('Volume')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle=':', alpha=0.7)

            plot_path = output_dir / "volume_distribution_plot.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"📊 Volume histogram saved to {plot_path.name}")