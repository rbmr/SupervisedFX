import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

def plot_arrays(arr: Union[np.ndarray, list[np.ndarray]], filename: str = "plot.png"):
    """
    Plots a 1D NumPy array with its index as the x-axis and saves the plot
    as an image file next to the current Python script.

    Args:
        arr (np.ndarray): The 1D NumPy array to plot.
        filename (str): The name of the file to save the plot as (e.g., "my_plot.png").
                        Defaults to "plot.png".
    """
    if isinstance(arr, np.ndarray):
        arr = [arr]
    if not all(isinstance(a, np.ndarray) and a.ndim == 1 for a in arr):
        raise ValueError("Input arrays must be 1-dimensional.")

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / filename

    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
    for i, a in enumerate(arr):
        plt.plot(a, label=f"{i}")
    plt.legend()
    plt.title("1D Array Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()  # Close the plot to free up memory

value_csv = Path(__file__).resolve().parent / "value.csv"
q_min_csv = Path(__file__).resolve().parent / "q_min.csv"
v = pd.read_csv(value_csv).to_numpy(dtype=np.float64)
q_min = pd.read_csv(q_min_csv).to_numpy(dtype=np.float64)

importance = v - q_min
print(importance[-2:]) # last row has weird values, leave out.
flat_importance = importance[:-1].flatten()
print(flat_importance[-2:])
print(max_flat_importance)

plot_arrays([flat_importance, np.repeat(max_flat_importance, flat_importance.shape[0])])