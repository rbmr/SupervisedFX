from pathlib import Path
from data.data import *
import pandas as pd
from common.constants import *
import logging 

def get_csv_files(p: Path):
    if not p.is_dir():
        raise ValueError("p must be a directory")
    for f in p.glob("*.csv"):
        yield f

def parse(p: Path) -> ForexData:
    if p.suffix != ".csv":
        raise ValueError("p must be a csv file")
    df = pd.read_csv(p)
    start = df["Gmt time"].iloc[0]
    end = df["Gmt time"].iloc[-1]
    date_format = "%d.%m.%Y %H:%M:%S.%f"
    start = datetime.strptime(start, date_format)
    start = start.replace(tzinfo = DT_TIMEZONE)
    end = datetime.strptime(end, date_format)
    end = end.replace(tzinfo = DT_TIMEZONE)
    pair, _, n, unit, off, _ = p.stem.split("_")
    assert len(pair) == 6
    c1 = Currency(pair[:3])
    c2 = Currency(pair[3:])
    off = OfferSide(off)
    gran = Granularity(n+unit)
    ref = ForexRef(c1,c2,gran,off,start,end)
    df.rename(columns={
        "Volume" : "volume",
        "Gmt time" : "date_gmt",
        "Low" : "low",
        "High" : "high",
        "Open" : "open",
        "Close" : "close"
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



