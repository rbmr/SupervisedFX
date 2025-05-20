from pathlib import Path
from common.data import *
import pandas as pd
from common.constants import *
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_csv_files(p: Path):
    if not p.is_dir():
        raise ValueError("p must be a directory")
    for f in p.glob("*.csv"):
        yield f

def parse(p: Path) -> ForexData:
    
    # load csv
    if p.suffix != ".csv":
        raise ValueError("p must be a csv file")
    df = pd.read_csv(p)
    
    # parse timestamps from gmt time column
    date_format = "%d.%m.%Y %H:%M:%S.%f"
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format=date_format)
    df['Gmt time'] = df['Gmt time'].dt.tz_localize(PD_TIMEZONE)
    
    # retrieve and convert start
    start: pd.Timestamp = df["Gmt time"].iloc[0]
    start: datetime = start.to_pydatetime()
    start: datetime = start.replace(tzinfo = DT_TIMEZONE)

    # retrieve and convert end
    end: pd.Timestamp = df["Gmt time"].iloc[-1]
    end: datetime = end.to_pydatetime()
    end: datetime = end.replace(tzinfo = DT_TIMEZONE)

    # parse currencies, offer, and granularity
    pair, _, n, unit, off, _ = p.stem.split("_")
    assert len(pair) == 6
    c1 = Currency(pair[:3])
    c2 = Currency(pair[3:])
    off = OfferSide(off)
    gran = Granularity(n+unit)
    
    # create reference
    ref = ForexRef(c1,c2,gran,off,start,end)
    
    # rename columns to be as expected
    df.rename(columns={
        "Volume" : Col.VOL,
        "Gmt time" : Col.TIME,
        "Low" : Col.LOW,
        "High" : Col.HIGH,
        "Open" : Col.OPEN,
        "Close" : Col.CLOSE
    }, inplace=True)

    return ForexData(ref, df)


if __name__ == "__main__":
    dukascopy_dir = Path(__file__).parent

    for f in get_csv_files(dukascopy_dir):
        logging.info(f"Parsing {f}")
        fd = parse(f)
        logging.info(f"Finished parsing, saving to: {fd.ref.get_path()}")
        fd.save()
        logging.info("Conversion saved.")



