from datetime import date, datetime
import pandas as pd

from src.data.analysis import analyze_and_save_report
from src.data.models import CandleData, TickData, Timeframe
from src.data.downloaders import DOWNLOADERS

def fetch_data(source: str, instrument: str, timeframe: Timeframe, start_date: date, end_date: date) -> None:
    """Fetches CandleData and intermediaries to cache.

    It handles caching, downloading, and downsampling automatically.
    1. It iterates through the required date range day by day.
    2. For each day, it tries to load the target timeframe data from disk.
    3. If not found, it tries to load tick data to downsample from.
    4. If tick data is not found, it downloads it.
    5. If tick data couldn't be downloaded, it skips it and logs a warning.

    Any newly generated/downloaded data is saved to disk for future use.
    """
    print(f"\nRequesting {instrument} {timeframe.name} data from {start_date} to {end_date}.")

    source = source.upper()
    if source not in DOWNLOADERS:
        raise ValueError(f"Source '{source}' is not supported.")

    days = pd.date_range(start_date, end_date, freq='D')

    for d in days:
        target_day = d.date()

        # 1. Check if target data already exists
        if CandleData.get_path(source, instrument, timeframe, target_day).exists():
            print(f"[{target_day}] Found cached {timeframe.name} data.")
            continue

        print(f"[{target_day}] No cached {timeframe.name} data found.")

        # 2. Check for tick data to downsample from
        tick_data = TickData.load(source, instrument, Timeframe.TICK, target_day)

        # 3. If no tick data, download it
        if tick_data is None:
            print(f"[{target_day}] No cached TICK data found. Downloading...")
            downloader = DOWNLOADERS[source]
            try:
                tick_data = downloader.fetch_day(instrument, target_day)
                tick_data.save(target_day)
            except Exception as e:
                print(f"[{target_day}] ERROR: Could not fetch or process data for this day. Skipping. Reason: {e}")
                continue
        else:
            print(f"[{target_day}] Found cached TICK data.")

        # 4. Downsample tick data and save the result
        print(f"[{target_day}] Downsampling TICK data to {timeframe.name}...")
        candle_data = tick_data.downsample(timeframe)
        candle_data.save(target_day)

def get_data(source: str, instrument: str, timeframe: Timeframe, start_date: date, end_date: date) -> CandleData:
    """Main method to fetch and then retrieve CandleData."""
    fetch_data(source, instrument, timeframe, start_date, end_date)
    return CandleData.load_range(source, instrument, timeframe, start_date, end_date)

def main_cli():
    """Main Command Line Interface to run the data processing."""
    print("=" * 30)
    print("Forex Data Processor")
    print("=" * 30)

    try:
        source_input = input("Enter the data source [default: DUKASCOPY]: ").strip().upper()
        source = source_input if source_input else "DUKASCOPY"

        instrument_input = input("Enter the instrument [default: EURUSD]: ").strip().upper()
        instrument = instrument_input if instrument_input else "EURUSD"

        start_str = input("Enter the start date (YYYYMMDD): ").strip()
        end_str = input("Enter the end date (YYYYMMDD): ").strip()
        start_date = datetime.strptime(start_str, "%Y%m%d").date()
        end_date = datetime.strptime(end_str, "%Y%m%d").date()

        print("\nAvailable granularities:")
        for tf in Timeframe:
            if tf != Timeframe.TICK:
                print(f"- {tf.name}")
        granularity_str = input("Choose a granularity: ").strip().upper()
        timeframe = Timeframe[granularity_str]

        final_data = get_data(
            source=source,
            instrument=instrument,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if final_data.df.empty:
            print("\nNo data was found or generated for the specified range.")
        else:
            print(f"\nSuccessfully retrieved data with {len(final_data.df)} rows.")
            print("First 5 rows:")
            print(final_data.df.head())
            analyze_and_save_report(final_data)

    except (ValueError, KeyError) as e:
        print(f"\nError: Invalid input. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    main_cli()