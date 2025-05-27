import pandas as pd
import numpy as np
from typing import Union, List, Any, Dict

class DataFrameToNumPyAccessor:
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self._data_np = df.to_numpy()
        self._columns = list(df.columns) # Store column names
        self._column_to_idx = {col_name: i for i, col_name in enumerate(df.columns)}
        self._num_rows = len(self._data_np)
        self._num_cols = len(self._columns)

    @property
    def numpy_array(self) -> np.ndarray:
        """Returns the underlying NumPy array."""
        return self._data_np

    @property
    def columns(self) -> List[str]:
        """Returns the list of column names."""
        return self._columns

    @property
    def shape(self) -> tuple:
        """Returns the shape of the underlying NumPy array."""
        return self._data_np.shape

    def get_column_index(self, column_name: str) -> int:
        """Returns the integer index for a given column name."""
        if column_name not in self._column_to_idx:
            raise KeyError(f"Column '{column_name}' not found.")
        return self._column_to_idx[column_name]

    def get_row(self, row_index: int) -> np.ndarray:
        """Gets a full row by its integer index."""
        if not 0 <= row_index < self._num_rows:
            raise IndexError(f"Row index {row_index} is out of bounds for {self._num_rows} rows.")
        return self._data_np[row_index]

    def get_column(self, column_identifier: Union[str, int]) -> np.ndarray:
        """Gets a full column by its name or integer index."""
        if isinstance(column_identifier, str):
            col_idx = self.get_column_index(column_identifier)
        elif isinstance(column_identifier, int):
            if not 0 <= column_identifier < self._num_cols:
                raise IndexError(f"Column index {column_identifier} is out of bounds for {self._num_cols} columns.")
            col_idx = column_identifier
        else:
            raise TypeError("Column identifier must be a string (name) or integer (index).")
        return self._data_np[:, col_idx]

    def get_value(self, row_index: int, column_identifier: Union[str, int]) -> Any:
        """Gets a single value by row index and column name/index."""
        if not 0 <= row_index < self._num_rows:
            raise IndexError(f"Row index {row_index} is out of bounds for {self._num_rows} rows.")

        if isinstance(column_identifier, str):
            col_idx = self.get_column_index(column_identifier)
        elif isinstance(column_identifier, int):
            if not 0 <= column_identifier < self._num_cols:
                raise IndexError(f"Column index {column_identifier} is out of bounds for {self._num_cols} columns.")
            col_idx = column_identifier
        else:
            raise TypeError("Column identifier must be a string (name) or integer (index).")
        return self._data_np[row_index, col_idx]

    def set_value(self, row_index: int, column_identifier: Union[str, int], value: Any) -> None:
        """Sets a single value by row index and column name/index."""
        if not 0 <= row_index < self._num_rows:
            raise IndexError(f"Row index {row_index} is out of bounds for {self._num_rows} rows.")

        if isinstance(column_identifier, str):
            col_idx = self.get_column_index(column_identifier)
        elif isinstance(column_identifier, int):
            if not 0 <= column_identifier < self._num_cols:
                raise IndexError(f"Column index {column_identifier} is out of bounds for {self._num_cols} columns.")
            col_idx = column_identifier
        else:
            raise TypeError("Column identifier must be a string (name) or integer (index).")
        self._data_np[row_index, col_idx] = value

    def set_row_from_dict(self, row_index: int, data_dict: Dict[str, Any]) -> None:
        """
        Sets values in a specific row using a dictionary where keys are column names.
        """
        if not 0 <= row_index < self._num_rows:
            raise IndexError(f"Row index {row_index} is out of bounds for {self._num_rows} rows.")

        for col_name, value in data_dict.items():
            if col_name not in self._column_to_idx:
                # Option 1: Raise an error for unknown columns
                raise KeyError(f"Column '{col_name}' in dictionary not found in the accessor.")
                # Option 2: Silently ignore unknown columns
                # print(f"Warning: Column '{col_name}' in dictionary not found. Skipping.")
                # continue
            col_idx = self._column_to_idx[col_name]
            self._data_np[row_index, col_idx] = value

    def __getitem__(self, key):
        """
        Allows for more flexible access, e.g., obj[row_idx, col_id] or obj[row_idx] or obj[col_name].
        This can be made as simple or complex as needed.
        Example: obj[5, 'open_bid'] or obj[5] (for a whole row) or obj['open_bid'] (for a whole column)
        """
        if isinstance(key, tuple):
            # Assuming (row_index, column_identifier)
            if len(key) == 2:
                row_idx, col_id = key
                return self.get_value(row_idx, col_id)
            else:
                raise ValueError("Tuple key must have two elements: (row_index, column_identifier).")
        elif isinstance(key, int):
            # Get a full row
            return self.get_row(key)
        elif isinstance(key, str):
            # Get a full column
            return self.get_column(key)
        # You could also add support for slices here for more advanced NumPy-like slicing
        # e.g., if isinstance(key, slice) or (isinstance(key, tuple) and any(isinstance(k, slice) for k in key))
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def __setitem__(self, key, value):
        """
        Allows for setting values, e.g., obj[row_idx, col_id] = new_value.
        """
        if isinstance(key, tuple) and len(key) == 2:
            row_idx, col_id = key
            self.set_value(row_idx, col_id, value)
        # You could also consider allowing setting a full row if value is a dict and key is row_index
        elif isinstance(key, int) and isinstance(value, dict):
             self.set_row_from_dict(key, value)
        else:
            raise TypeError(f"Unsupported key type or value type for setting. Use obj[row, col] = val or obj[row_idx] = dict_val.")

