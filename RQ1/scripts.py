from pathlib import Path
from typing import Callable
import pandas as pd 

IMPORT_FUNCS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": lambda p: pd.read_csv(p),
    ".json": lambda p: pd.read_json(p, orient="records", lines=True),
    ".xlsx": lambda p: pd.read_excel(p),
    ".parquet": lambda p: pd.read_parquet(p),
    ".feather": lambda p: pd.read_feather(p),
    ".h5": lambda p: pd.read_hdf(p, key='df'),
    ".html": lambda p: pd.read_html(p)[0],  # Returns list of DataFrames
    ".xml": lambda p: pd.read_xml(p),
    ".pkl": lambda p: pd.read_pickle(p)
}

def load_dataframe(path: Path) -> pd.DataFrame:
    """
    Loads a pandas dataframe from the given file path.
    """
    ext = path.suffix.lower()
    if ext not in IMPORT_FUNCS:
        raise ValueError(f"Unsupported file extension: {ext}")
    return IMPORT_FUNCS[ext](path)

EXPORT_FUNCS: dict[str, Callable[[pd.DataFrame, Path], None]] = {
    ".csv": lambda df, p: df.to_csv(p, index=False),
    ".json": lambda df, p: df.to_json(p, orient="records", lines=True),
    ".xlsx": lambda df, p: df.to_excel(p, index=False),
    ".parquet": lambda df, p: df.to_parquet(p, index=False),
    ".feather": lambda df, p: df.to_feather(p),
    ".h5": lambda df, p: df.to_hdf(p, key='df', mode='w'),
    ".html": lambda df, p: df.to_html(p, index=False),
    ".xml": lambda df, p: df.to_xml(p),
    ".pkl": lambda df, p: df.to_pickle(p)
}

def save_dataframe(df: pd.DataFrame, path: Path):
    """
    Saves the pandas dataframe given a path.
    """
    ext = path.suffix.lower()
    if ext not in EXPORT_FUNCS:
        raise ValueError(f"Unsupported file extension: {ext}")
    EXPORT_FUNCS[ext](df, path)
