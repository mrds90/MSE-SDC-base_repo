import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import sys
sys.path.append('../..')
from Pulse_Generator import *

SQUARE = 0
SINE = 1
TRIANGULAR = 2
RAISED_COSINE = 3

fs_MHz = 16.0
samples = 16
f0_Mhz = 1
M = samples
N = samples
N_SIGNAL_B = 6 * 100
SNR_dB = 20  # Signal-to-noise ratio in dB

print("Periodo de muestreo : {} µS".format(1 / fs_MHz))
print("Tiempo de símbolo : {} µS".format((1 / fs_MHz) * samples))
print("Frecuencia de muestreo : {} MHz".format(fs_MHz))


b = GenerateBinarySignal(N_SIGNAL_B)
d = InsertZerosAndAssignValues(b, M)

p = (
    SquarePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    SinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    TriangularPulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    RaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz,  beta=0.2, samples=samples),
)

# Plot Singlas for debug
pulse_names = ["Square Pulse", "Sine Pulse", "Triangular Pulse", "Raised Cosine Pulse"]
t_us = TimeVector(fs_MHz, samples)
for i, signal in enumerate(p):
    t_us = TimeVector(fs_MHz, len(signal))  
    plt.scatter(t_us, signal, label=pulse_names[i])
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude")
    plt.title("Signal: {}".format(pulse_names[i]))
    plt.legend()
    plt.grid(True)
    plt.show()

# Convolve each pulse with the input signal d to obtain the transmitted signal x
# Use 'same' mode in np.convolve() to ensure that the output signal x has the same length as the input signal d,
# thus maintaining synchronization between the signals d and x.
x = (
    np.convolve(d, p[SQUARE], mode="same"),
    np.convolve(d, p[SINE], mode="same"),
    np.convolve(d, p[TRIANGULAR], mode="same"),
    np.convolve(d, p[RAISED_COSINE], mode="same"),
)
# Add channel effects and noise
c = (
    AddChannelNoise(x[SQUARE], SNR_dB),
    AddChannelNoise(x[SINE], SNR_dB),
    AddChannelNoise(x[TRIANGULAR], SNR_dB),
    AddChannelNoise(x[RAISED_COSINE], SNR_dB),
)


# Color Blind friendly colors
CB_color_cycle = (
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
)

fig, ax = plt.subplots(4, figsize=(8, 10))

fig.suptitle("Ejercicio 3", fontsize=16)

pulses = ("Square", "Sine", "Triangular", "Raised Cosine")

for i in range(4):

    markerline, stemlines, baseline = ax[i].stem(
        np.arange(int(len(x[i])/100)), x[i][0:int(len(x[i])/100)], markerfmt="o", label="Output"
    )
    plt.setp(stemlines, "color", CB_color_cycle[0])
    plt.setp(markerline, "color", CB_color_cycle[0])
    plt.setp(stemlines, "linestyle", "dotted")

    markerline, stemlines, baseline = ax[i].stem(
        np.arange(int(len(c[i])/100)), c[i][0:int(len(c[i])/100)], markerfmt="o", label="Output+Noise"
    )
    plt.setp(stemlines, "color", CB_color_cycle[1])
    plt.setp(markerline, "color", CB_color_cycle[1])
    plt.setp(stemlines, "linestyle", "dotted")

    markerline, stemlines, baseline = ax[i].stem(
        np.arange(int(len(d)/100)), d[0:int(len(d)/100)], markerfmt="o", label="Deltas"
    )
    plt.setp(stemlines, "color", CB_color_cycle[3])
    plt.setp(markerline, "color", CB_color_cycle[3])
    # plt.setp(stemlines, "linestyle", "dotted")

    ax[i].set_xlabel("Samples")
    ax[i].set_ylabel("Value")
    ax[i].set_title("Pulse Type: {}".format(pulses[i]))
    ax[i].legend()
    ax[i].grid(True)

plt.tight_layout()
plt.show()


# Normalize pulses
normalized_pulses = [pulse / np.sqrt(np.sum(pulse**2)) for pulse in p]

# Calculate spectra
x_spectrum = [(20 * np.log10(np.abs(fft(x_signal)))) for x_signal in x]
c_spectrum = [(20 * np.log10(np.abs(fft(c_signal)))) for c_signal in c]

# Frequency axis
f = np.fft.fftfreq(len(x[0]), 1/16)

# Plot spectral density
fig, ax = plt.subplots(4, 2, figsize=(10, 12))

for i in range(4):
    ax[i, 0].semilogy(f[f > 0], x_spectrum[i][f > 0], label=pulses[i], color=CB_color_cycle[i])
    ax[i, 1].semilogy(f[f > 0], c_spectrum[i][f > 0], label=pulses[i], color=CB_color_cycle[i])

    ax[i, 0].set_title("Spectral Density of Transmitted Signals (x)")
    ax[i, 0].set_xlabel("Frequency (MHz)")
    ax[i, 0].set_ylabel("Magnitude")
    ax[i, 0].legend()
    ax[i, 0].grid(True)

    ax[i, 1].set_title("Spectral Density of Received Signals (c)")
    ax[i, 1].set_xlabel("Frequency (MHz)")
    ax[i, 1].set_ylabel("Magnitude")
    ax[i, 1].legend()
    ax[i, 1].grid(True)

plt.tight_layout()
plt.show()

