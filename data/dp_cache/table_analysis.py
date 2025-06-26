import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.envs.dp import DPTable


def min_max_norm(arr):
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val)

class IntensityPlotApp:
    def __init__(self, master, data):
        self.master = master
        master.title("Intensity Plot Viewer")

        self.data = data
        self.current_start_index = 0
        self.chunk_size = 20

        # Setup the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add Matplotlib navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Setup the button
        self.next_button = ttk.Button(master, text="Load Next 20 Values", command=self.load_next_chunk)
        self.next_button.pack(side=tk.BOTTOM, pady=10)

        # Initialize the plot with the first chunk
        self.update_plot()

    def update_plot(self):
        self.ax.clear()  # Clear existing plot

        end_index = self.current_start_index + self.chunk_size
        current_chunk = self.data[self.current_start_index:end_index]

        if current_chunk.shape[0] == 0:
            self.ax.text(0.5, 0.5, "No more data to display", horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, fontsize=16, color='red')
            self.canvas.draw()
            return

        image = min_max_norm(current_chunk)
        n_timesteps, n_exposures = current_chunk.shape

        self.ax.imshow(image, cmap='gray', interpolation='nearest', origin='lower')
        self.ax.set_title(f'Intensity Plot of Data (Timesteps {self.current_start_index+1}-{end_index})')
        self.ax.set_xlabel('Exposure Number')
        self.ax.set_ylabel('Timestep Number')

        # Set ticks dynamically based on the current chunk size
        self.ax.set_xticks(np.arange(n_exposures), labels=[f'Exp {i+1}' for i in range(n_exposures)])
        self.ax.set_yticks(np.arange(n_timesteps), labels=[f'Time {self.current_start_index + i + 1}' for i in range(n_timesteps)])

        # Create a colorbar
        if not self.ax.images:  # Only add if no image exists (first plot)
             cbar = self.fig.colorbar(self.ax.images[0], ax=self.ax, label='Intensity Value')
        else: # Update existing colorbar for subsequent plots
            # This part can be tricky. Often it's easier to recreate the plot from scratch (which we do with ax.clear())
            # or manage the colorbar artist more explicitly. For simplicity with ax.clear(), a new one is fine.
            pass

        self.fig.tight_layout()
        self.canvas.draw()

    def load_next_chunk(self):
        if self.current_start_index + self.chunk_size < self.data.shape[0]:
            self.current_start_index += self.chunk_size
            self.update_plot()
        else:
            print("Reached the end of the data.")
            self.next_button.config(state=tk.DISABLED) # Disable button when no more data


def main():
    table = DPTable.load(Path("C:\\Users\\rober\\SupervisedFX\\data\\dp_cache\\dp_table_a15_e30_tc0p0001_dataf66243675ab3f906.npz"))
    policy = table.policy_table
    root = tk.Tk()
    app = IntensityPlotApp(root, policy)
    root.mainloop()

if __name__ == '__main__':
    main()