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
    #  @param mode Plotting mode: "real" or "complex"
    #  @param num_stages Number of processing stages (subplots)
    #  @param fs_MHz Sampling frequency in MHz
    #  @param CB_color_cycle Optional custom color cycle
    def __init__(self, mode="real", num_stages=1, fs_MHz=16.0, CB_color_cycle=None):
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
    def _init_figure(self):
        nrows = self.num_stages if self.mode == "real" else self.num_stages * 2
        self.fig, axes = plt.subplots(nrows, 1, figsize=(15, 4 * nrows), sharex=True)

        if nrows == 1:
            self.axes = [axes]
        else:
            self.axes = list(axes)

        self.initialized = True

    ## @brief Creates a stem plot on a given axis
    #  @param ax Matplotlib axis
    #  @param t Time vector
    #  @param y Signal values
    #  @param label Label for the plot legend
    #  @param color Line and marker color
    #  @param linestyle Line style
    #  @param markerfmt Marker format
    #  @param alpha Transparency
    #  @param markersize Marker size
    def _stem_plot(self, ax, t, y, label, color, linestyle, markerfmt, alpha=1.0, markersize=6):
        markerline, stemlines, baseline = ax.stem(t, y, markerfmt=markerfmt, label=label)
        plt.setp(stemlines, color=color, linestyle=linestyle, alpha=alpha)
        plt.setp(markerline, color=color, alpha=alpha, markersize=markersize)

    ## @brief Adds a plot to the current figure
    #  @param signal Signal to plot
    #  @param reference_signal Reference signal to compare (e.g., deltas)
    #  @param title Title of the plot
    #  @param reference_label Label for the reference signal
    #  @param pulse Label for the pulse signal
    def add_plot(self, signal, reference_signal, title="Signal",
                 reference_label="Deltas", pulse="Pulse"):

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
            self.initialized = False  # permitir reutilización
