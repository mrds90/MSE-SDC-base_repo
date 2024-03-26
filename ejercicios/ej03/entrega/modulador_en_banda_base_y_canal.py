import random
import numpy as np
import matplotlib.pyplot as plt
from pulse_generator import *

SQUARE = 0
SINE = 1
TRIANGULAR = 2
RAISED_COSINE = 3

fs_MHz = 16.0
samples = 16
f0_Mhz = 1
M = samples
N = samples
N_SIGNAL_B = 10
SNR_dB = 10  # Signal-to-noise ratio in dB

print("Periodo de muestreo : {} µS".format(1 / fs_MHz))
print("Tiempo de simbolo : {} µS".format((1 / fs_MHz) * samples))
print("Frecuencia de muestreo : {} MHz".format(fs_MHz))


def GenerateBinarySignal(length: int) -> np.ndarray:
    """
    Generates a random binary signal of specified length.

    Parameters:
        length (int): Length of the binary signal.

    Returns:
        numpy.ndarray: Random binary signal of specified length.
    """

    # Generate a sequence of random binary values (0 or 1)
    binary_sequence = np.random.randint(2, size=length)

    # Generate a sequence of random binary values (0 or 1)
    binary_sequence[binary_sequence == 0] = -1

    return binary_sequence


def InsertZerosAndAssignValues(binary_signal: np.ndarray, M: int) -> np.ndarray:
    """
    Inserts M-1 zeros between each bit of the binary signal and assigns values to each bit.

    Parameters:
        binary_signal (numpy.ndarray): Binary signal.
        M (int): Number of zeros to insert between each bit.

    Returns:
        numpy.ndarray: Modified signal with inserted zeros and assigned values.
    """
    length = len(binary_signal)
    modified_length = length * M
    modified_signal = np.zeros(
        modified_length, dtype=float
    )  # Initialize a signal of zeros with the modified length

    # Create an array to represent indices of each bit in the modified signal
    bit_indices = np.arange(length) * M

    # Assign values to the initial position of each bit
    modified_signal[bit_indices] = binary_signal * 1.0

    return modified_signal


def AddChannelNoise(signal: np.ndarray, SNR_dB: float) -> np.ndarray:
    """
    Adds additive white Gaussian noise (AWGN) to the signal.

    Parameters:
        signal (numpy.ndarray): Input signal.
        SNR_dB (float): Signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: Signal with added noise.
    """
    signal_power = np.sum(signal**2) / len(signal)
    noise_power = signal_power / (10 ** (SNR_dB / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy_signal = signal + noise
    return noisy_signal


b = GenerateBinarySignal(N_SIGNAL_B)
d = InsertZerosAndAssignValues(b, M)


p = (
    SquarePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    SinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    TriangularPulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    RaisedCosinePulse(fs_MHz=fs_MHz, beta=1, samples=samples),
)

# Plot Singlas for debug
# t_us = TimeVector(fs_MHz, samples)
# pulse_names = ["Square Pulse", "Sine Pulse", "Triangular Pulse", "Raised Cosine Pulse"]
# for i, signal in enumerate(p):
#     plt.plot(t_us, signal, label=pulse_names[i])
#     plt.xlabel("Time (µs)")
#     plt.ylabel("Amplitude")
#     plt.title("Signal: {}".format(pulse_names[i]))
#     plt.legend()
#     plt.grid(True)
#     plt.show()

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
        np.arange(len(x[i])), x[i], markerfmt="o", label="Output"
    )
    plt.setp(stemlines, "color", CB_color_cycle[1])
    plt.setp(stemlines, "linestyle", "dotted")

    markerline, stemlines, baseline = ax[i].stem(
        np.arange(len(c[i])), c[i], markerfmt="o", label="Output+Noise"
    )
    plt.setp(stemlines, "color", CB_color_cycle[2])
    plt.setp(stemlines, "linestyle", "dotted")

    markerline, stemlines, baseline = ax[i].stem(
        np.arange(len(d)), d, markerfmt="o", label="Deltas"
    )
    plt.setp(stemlines, "color", CB_color_cycle[0])
    plt.setp(stemlines, "linestyle", "dotted")

    ax[i].set_xlabel("Samples")
    ax[i].set_ylabel("Value")
    ax[i].set_title("Pulse Type: {}".format(pulses[i]))
    ax[i].legend()
    ax[i].grid(True)

plt.tight_layout()
plt.show()


from scipy.fft import fft

# Normalize pulses
normalized_pulses = [pulse / np.sqrt(np.sum(pulse**2)) for pulse in p]

# Calculate spectra
x_spectrum = [np.abs(fft(x_signal)) for x_signal in x]
c_spectrum = [np.abs(fft(c_signal)) for c_signal in c]

# Frequency axis
f = np.fft.fftfreq(len(x[0]))

# Plot spectral density
fig, ax = plt.subplots(2, figsize=(10, 8))

for i in range(4):
    ax[0].semilogy(f, x_spectrum[i], label=pulses[i], color = CB_color_cycle[i])
    ax[1].semilogy(f, c_spectrum[i], label=pulses[i], color = CB_color_cycle[i])

ax[0].set_title("Spectral Density of Transmitted Signals (x)")
ax[0].set_xlabel("Frequency (MHz)")
ax[0].set_ylabel("Magnitude")
ax[0].legend()
ax[0].grid(True)

ax[1].set_title("Spectral Density of Received Signals (c)")
ax[1].set_xlabel("Frequency (MHz)")
ax[1].set_ylabel("Magnitude")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
