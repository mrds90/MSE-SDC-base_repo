## @file plotter.py
#  @brief Utility module for plotting real or complex signals using stem plots.
#  @details Provides a configurable Plotter class that supports multi-stage visualization,
#           either in real or complex mode, useful for signal processing chains or debugging.
#  @author Marcos Dominguez
#  @author Lucas Monzon Languasco

import numpy as np
import matplotlib.pyplot as plt
from lib.pulse_generator.pulse_generator import PulseGenerator

## @class Plotter
#  @brief A class to create stem plots for signal visualization.
#  @details Supports plotting of multiple signal processing stages in real or complex mode.
class Plotter:
    ## @brief Default color cycle for plots (color-blind friendly)
    CB_color_cycle = (
        "#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
        "#984ea3", "#999999", "#e41a1c", "#dede00"
    )

    ## @brief Constructor
    #  @param mode (str) Plotting mode: "real" or "complex"
    #  @param num_stages (int) Number of processing stages (subplots)
    #  @param fs_MHz (float) Sampling frequency in MHz
    #  @param CB_color_cycle (tuple | None) Optional custom color cycle
    def __init__(self, mode: str = "real", num_stages: int = 1, fs_MHz: float = 16.0, CB_color_cycle: tuple | None = None):
        if mode not in ["real", "complex"]:
            raise ValueError("mode debe ser 'real' o 'complex'")

        self.mode = mode
        self.num_stages = num_stages
        self.fs_MHz = fs_MHz
        self.CB_color_cycle = CB_color_cycle or Plotter.CB_color_cycle

        self.call_counter = 0
        self.fig = None
        self.axes = None
        self.initialized = False

    ## @brief Initializes the matplotlib figure and subplots
    #  @return None
    def _init_figure(self) -> None:
        if self.mode == "real":
            nrows = self.num_stages
            ncols = 1
            figsize = (15, 4 * nrows)
        else:  # complex mode
            nrows = self.num_stages
            ncols = 2  # 2 columns: real and imaginary
            figsize = (20, 4 * nrows)  # wider figure for 2 columns

        self.fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)

        # Handle different axes array shapes
        if nrows == 1 and ncols == 1:
            self.axes = [axes]
        elif nrows == 1:
            self.axes = list(axes)
        elif ncols == 1:
            self.axes = list(axes)
        else:
            # For complex mode, flatten the 2D array to match expected indexing
            self.axes = axes.flatten()

        self.initialized = True

    ## @brief Creates a stem plot on a given axis
    #  @param ax (matplotlib.axes.Axes) Matplotlib axis
    #  @param t (np.ndarray) Time vector
    #  @param y (np.ndarray) Signal values
    #  @param label (str) Label for the plot legend
    #  @param color (str) Line and marker color
    #  @param linestyle (str) Line style
    #  @param markerfmt (str) Marker format
    #  @param alpha (float) Transparency
    #  @param markersize (int) Marker size
    #  @return None
    def _stem_plot(self, ax, t: np.ndarray, y: np.ndarray, label: str, color: str, linestyle: str, markerfmt: str, alpha: float = 1.0, markersize: int = 6) -> None:
        markerline, stemlines, baseline = ax.stem(t, y, markerfmt=markerfmt, label=label)
        plt.setp(stemlines, color=color, linestyle=linestyle, alpha=alpha)
        plt.setp(markerline, color=color, alpha=alpha, markersize=markersize)

    ## @brief Adds a plot to the current figure
    #  @param signal (np.ndarray) Signal to plot
    #  @param reference_signal (np.ndarray) Reference signal to compare (e.g., deltas)
    #  @param title (str) Title of the plot
    #  @param reference_label (str) Label for the reference signal
    #  @param pulse (str) Label for the pulse signal
    #  @return None
    def add_plot(self, signal: np.ndarray, reference_signal: np.ndarray, title: str = "Signal",
                 reference_label: str = "Deltas", pulse: str = "Pulse") -> None:

        if not self.initialized:
            self._init_figure()

        idx = self.call_counter * (2 if self.mode == "complex" else 1)
        t_signal = PulseGenerator.time_vector(self.fs_MHz, len(signal))
        t_ref = PulseGenerator.time_vector(self.fs_MHz, len(reference_signal))

        if self.mode == "real":
            ax = self.axes[idx]
            self._stem_plot(ax, t_signal, np.real(signal), f"{pulse} real", self.CB_color_cycle[0], "dotted", "o")
            self._stem_plot(ax, t_ref, np.real(reference_signal), reference_label, self.CB_color_cycle[1], "-", "D", alpha=0.7, markersize=3)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, which="major", linestyle="-", linewidth=1, alpha=0.8)
            ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.minorticks_on()

        elif self.mode == "complex":
            ax_r = self.axes[idx]
            ax_i = self.axes[idx + 1]

            self._stem_plot(ax_r, t_signal, np.real(signal), f"{pulse} real", self.CB_color_cycle[0], "dotted", "o")
            self._stem_plot(ax_r, t_ref, np.real(reference_signal), reference_label, self.CB_color_cycle[1], "-", "D", alpha=0.7, markersize=3)
            ax_r.set_title(f"{title} (Real)")
            ax_r.legend()
            ax_r.grid(True, which="major", linestyle="-", linewidth=1, alpha=0.8)
            ax_r.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
            ax_r.minorticks_on()

            self._stem_plot(ax_i, t_signal, np.imag(signal), f"{pulse} imag", self.CB_color_cycle[0], "dotted", "o")
            self._stem_plot(ax_i, t_ref, np.imag(reference_signal), reference_label, self.CB_color_cycle[1], "-", "D", alpha=0.7, markersize=3)
            ax_i.set_title(f"{title} (Imag)")
            ax_i.legend()
            ax_i.grid(True, which="major", linestyle="-", linewidth=1, alpha=0.8)
            ax_i.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
            ax_i.minorticks_on()

        self.call_counter += 1

        if self.call_counter == self.num_stages:
            plt.tight_layout()
            plt.show()
            self.call_counter = 0
            self.initialized = False  # allow reuse
