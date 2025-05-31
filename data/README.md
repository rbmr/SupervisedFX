# Forex Data Processor for Dukascopy

This document outlines how to use the `data.py` script to download, process, combine, and analyze historical tick data from Dukascopy.

## Overview

The script provides functionalities to:
1.  Download tick-by-tick historical data from Dukascopy for a specified financial instrument and year.
2.  Convert raw tick data into OHLCV (Open, High, Low, Close, Volume) candle data for various timeframes (e.g., M1, H1, D1).
3.  Combine multiple OHLCV data files (e.g., yearly files) into a single consolidated dataset.
4.  Perform basic analysis on the generated OHLCV data, including identifying potential data gaps and visualizing price and volume.

## How to Use

1.  **Run the commons/data.py:**

2.  **Follow the on-screen menu:**
    The script will present a menu with the following options:

    * **`0. Download Dukascopy Tick Data, convert, analyze, and save`**:
        * Use this to fetch raw tick data for a **single year** directly from Dukascopy.
        * It will prompt for the currency pair (e.g., `EURUSD`), the year, and the desired candle granularity (e.g., `H1`, `D1`).
        * The raw tick data for the year will be saved.
        * It will then automatically convert this yearly tick data to OHLCV candles of your chosen granularity.
        * Finally, it will analyze and save this yearly OHLCV data.

    * **`1. Load Tick Data and convert it to OHLCV`**:
        * Use this if you already have a CSV file containing tick data (from Dukascopy or another source) and want to convert it to OHLCV candles.
        * It will ask for the source name, instrument, path to your CSV, column names for time, bid, ask, and optional volume, and the target granularity.

    * **`2. Load multiple OHLCV files and combine them, then analyze and save`**:
        * Use this option **after** you have generated individual OHLCV files (e.g., one file per year using Option 0).
        * It will ask for the source, instrument, and granularity (which should match the files you are combining).
        * You'll then provide paths to multiple OHLCV CSV files.
        * The script will combine these files, sort them by date, remove duplicates, and then analyze and save the consolidated dataset.

    * **`3. Exit`**: Exits the application.

**Note:** DATA_DIR is most likely set to `./data`.

## Recommended Workflow for Comprehensive Research Data

To build a comprehensive dataset spanning multiple years for your research, follow these steps:

1.  **Fetch Data for Each Year Individually:**
    * Use **Option 0** from the menu.
    * Enter the currency pair (e.g., `EURUSD`), the first year you need (e.g., `2020`), and your desired base granularity (e.g., `H1`).
    * The script will download all tick data for that year, save it as a `.csv` file, then convert it to H1 (or your chosen granularity) OHLCV candles, analyze, and save these candles.
    * **Repeat this process for every year** you want to include in your research (e.g., 2021, 2022, 2023).
    * Each run of Option 0 will produce:
        * Raw yearly tick data: `DATA_DIR/TICK/DUKASCOPY/{INSTRUMENT}/{YEAR}.csv`
            * Example: `data/TICK/DUKASCOPY/EURUSD/2020.csv`
        * Yearly OHLCV candle data and analysis: `DATA_DIR/{SOURCE}/{INSTRUMENT}/{GRANULARITY_STR}/{START_DATETIME}_{END_DATETIME}/`
            * Example: `data/DUKASCOPY/EURUSD/H/{YYYYMMDDHHMMSS}_{YYYYMMDDHHMMSS}/data.csv` (and analysis files in the same folder). The start/end datetimes will span the year.

2.  **Combine Yearly OHLCV Data:**
    * Once you have generated the yearly OHLCV files from Step 1, use **Option 2** from the menu.
    * Enter the same source (`DUKASCOPY`), instrument (e.g., `EURUSD`), and granularity (e.g., `H1`) that you used in Step 1.
    * When prompted, provide the full paths to each of the yearly `data.csv` files located within the `DATA_DIR/{SOURCE}/{INSTRUMENT}/{GRANULARITY_STR}/.../` directories generated in Step 1.
    * The script will combine these yearly files into a single, continuous OHLCV dataset.

3.  **Output of Combined Data:**
    * The combined OHLCV data and its analysis will be saved to a new directory structure:
        `DATA_DIR/{SOURCE}/{INSTRUMENT}/{GRANULARITY_STR}/{COMBINED_START_DATETIME}_{COMBINED_END_DATETIME}/`
        * The `{COMBINED_START_DATETIME}` will be the earliest timestamp from your input files, and `{COMBINED_END_DATETIME}` will be the latest.
        * Inside this folder, you will find:
            * `data.csv`: The consolidated OHLCV data.
            * `missing_rows_analysis.txt`: Analysis of time gaps.
            * `close_prices_with_missing_data.png`: Plot of close prices highlighting unusual gaps.
            * `volume_distribution.png`: Histogram of trade volumes.

4.  **Custom Scripts for Setting Periods (Advanced):**
    * The main menu workflow helps you combine data over multiple years. If you need to analyze specific sub-periods from your large combined dataset, you would typically load the final `data.csv` into a custom Python script or a Jupyter Notebook using pandas.
    * The `ForexCandleData` class in the script has a `set_period(start_timestamp, end_timestamp)` method. You can instantiate this class with your combined data and then use this method in a custom script if you need to programmatically create new `ForexCandleData` objects for specific date ranges. However, for most research, filtering the final combined CSV directly with pandas is often sufficient.

## Included Analysis

When data is saved (either from a single year download/conversion or after combining multiple files), the following analysis is performed and output to the same directory as the `data.csv`:

1.  **Data Saving:**
    * The primary OHLCV data is saved as `data.csv`. Columns include: `date_gmt`, `open_bid`, `high_bid`, `low_bid`, `close_bid`, `open_ask`, `high_ask`, `low_ask`, `close_ask`, `volume`.

2.  **Missing Rows/Gap Analysis (`missing_rows_analysis.txt`):**
    * Calculates the time difference between consecutive candles.
    * Identifies gaps that are neither the standard candle interval (e.g., 60 minutes for H1) nor typical weekend gaps.
    * Estimates the number of potentially missing candles for these unusual gaps. This helps in understanding data completeness.

3.  **Close Price Plot (`close_prices_with_missing_data.png`):**
    * A line plot of both bid and ask close prices over time.
    * Markers (`x`) are placed on candles that appear *after* an unusual time gap, helping to visualize potential data discontinuities.

4.  **Volume Distribution Histogram (`volume_distribution.png`):**
    * A histogram showing the frequency of different volume amounts (up to the 99th percentile to exclude extreme outliers and improve readability).
    * Volume is either the sum of tick volumes (if available from source) or the count of ticks within each candle.

## Important Notes for Dukascopy Data

* **Timezones:** All timestamps are processed and aimed to be stored in GMT/UTC. The `ms_offset` from Dukascopy's raw data is relative to the start of the hour.
* **Price Format:** Dukascopy stores prices as integers (e.g., EURUSD at 1.23456 is stored as 123456). The script automatically divides by the correct decimal factor (e.g., 100000.0 for EURUSD, 1000.0 for JPY pairs).
* **Volume Format:** Volumes from Dukascopy ticks are typically floats representing millions (e.g., 0.25 means 0.25 million). The script scales this to the actual amount. If tick volume is not used, candle volume represents the number of ticks in that candle.
* **Weekend Gaps:** Forex markets close over the weekend. Expect larger time gaps between Friday's last candle and Monday's first candle. The gap analysis attempts to account for typical weekend durations.
* **Data Availability:** Dukascopy may not have data for all hours, especially during bank holidays or periods of extremely low liquidity. The downloader will skip hours for which Dukascopy returns an error (e.g., 404 Not Found). This will manifest as gaps in the data.
* **Rate Limiting:** While not explicitly handled by the script, be mindful if downloading very large amounts of data rapidly, as Dukascopy might have unstated rate limits. The script downloads hour by hour, which is generally fine.