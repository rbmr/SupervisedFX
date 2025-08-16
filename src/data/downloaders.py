import io
import lzma
import struct
from datetime import date, datetime, time

import pandas as pd
from src.scripts import fetch_all
from src.data.models import TickData, Timeframe

class DukascopyDownloader:
    """Handles fetching and processing of tick data from Dukascopy for a single day."""

    SOURCE = "DUKASCOPY"
    _TICK_FORMAT = '>IIIff'  # 3 unsigned ints, 2 floats (big-endian)

    @staticmethod
    def _decompress_lzma_bytes(data_bytes: bytes) -> bytes:
        with lzma.open(io.BytesIO(data_bytes)) as f:
            return f.read()

    @staticmethod
    def _bi5_to_df_from_bytes(raw_bytes: bytes) -> pd.DataFrame:
        chunk_size = struct.calcsize(DukascopyDownloader._TICK_FORMAT)
        data = [
            struct.unpack(DukascopyDownloader._TICK_FORMAT, raw_bytes[i:i + chunk_size])
            for i in range(0, len(raw_bytes), chunk_size)
        ]
        return pd.DataFrame(data, columns=['ms_offset', 'ask', 'bid', 'ask_vol', 'bid_vol'])

    @staticmethod
    def _get_url(symbol: str, dt: datetime) -> str:
        return (f"https://datafeed.dukascopy.com/datafeed/{symbol.upper()}/"
                f"{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")

    @staticmethod
    def _process_tick_df(df: pd.DataFrame, symbol: str, dt_hour: datetime) -> pd.DataFrame:
        """Standardizes the raw tick data DataFrame."""
        if df.empty:
            return pd.DataFrame()

        base_datetime = pd.Timestamp(dt_hour, tz="UTC")
        df['time'] = base_datetime + pd.to_timedelta(df['ms_offset'], unit='ms')

        divisor = 1000.0 if 'JPY' in symbol.upper() else 100000.0
        df['ask'] = df['ask'] / divisor
        df['bid'] = df['bid'] / divisor
        df['ask_vol'] = df['ask_vol']
        df['bid_vol'] = df['bid_vol']

        return df[['time', 'bid', 'ask', 'bid_vol', 'ask_vol']]

    @classmethod
    def fetch_day(cls, instrument: str, d: date) -> TickData:
        """Fetches all available tick data for a given instrument and a single day.

        Returns a standardized TickData object.
        """
        print(f"[{d.strftime('%Y-%m-%d')}] Fetching Dukascopy data for {instrument}...")

        hours_of_day = [datetime.combine(d, time(h)) for h in range(24)]
        urls = [cls._get_url(instrument, dt) for dt in hours_of_day]
        results = fetch_all(urls, raise_on_fail=True, max_workers=2)

        hourly_dfs = []
        for res_bytes, dt_hour in zip(results, hours_of_day):
            if not res_bytes:
                continue
            raw_bytes = cls._decompress_lzma_bytes(res_bytes)
            hour_df = cls._bi5_to_df_from_bytes(raw_bytes)
            processed_df = cls._process_tick_df(hour_df, instrument, dt_hour)
            hourly_dfs.append(processed_df)

        if not hourly_dfs:
            print(f"[{d.strftime('%Y-%m-%d')}] No data found for {instrument}.")
            return TickData(cls.SOURCE, instrument, Timeframe.TICK, pd.DataFrame(columns=['time', 'bid', 'ask', 'bid_vol', 'ask_vol']))

        day_df = pd.concat(hourly_dfs, ignore_index=True).sort_values(by="time").reset_index(drop=True)
        print(f"[{d.strftime('%Y-%m-%d')}] Fetched {len(day_df)} ticks.")

        return TickData(cls.SOURCE, instrument, Timeframe.TICK, day_df)

DOWNLOADERS = {
    DukascopyDownloader.SOURCE: DukascopyDownloader,
}