import numpy as np
from scipy.signal import convolve
import os
import platform
if "microsoft" in platform.uname().release.lower() or "WSL" in os.environ:
    import matplotlib
    matplotlib.use("TkAgg")  # Puedes probar también con 'Qt5Agg'
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from Pulse_Generator import *


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

def PlotRealSignal(signal, reference_signal, title="Signal", reference_label="Deltas", pulse="Pulso", fs_MHz=16.0, CB_color_cycle=None):
    if CB_color_cycle is None:
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

    fig, axes = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
    t_us = TimeVector(fs_MHz, len(np.real(signal)))

    # Plot signal
    markerline, stemlines, baseline = axes.stem(t_us, np.real(signal), markerfmt="o", label=f"{pulse} real")
    plt.setp(stemlines, "color", CB_color_cycle[0])
    plt.setp(markerline, "color", CB_color_cycle[0])
    plt.setp(stemlines, "linestyle", "dotted")


    # Plot reference signal with alpha and stem style
    t_us = TimeVector(fs_MHz, len(np.real(reference_signal)))
    markerline, stemlines, baseline = axes.stem(t_us, np.real(reference_signal), markerfmt="D", label=reference_label)
    plt.setp(stemlines, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(markerline, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(stemlines, "linestyle", "-")  # Change linestyle to stem style
    plt.setp(markerline, "markersize", 3)

    axes.legend()
    axes.grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.8)
    axes.grid(True, which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes.minorticks_on()

    plt.suptitle(title)
    # plt.tight_layout()
    plt.show()

def PlotComplexSignal(signal, reference_signal, title="Signal", reference_label="Deltas", pulse="Pulso", fs_MHz=16.0, CB_color_cycle=None):
    if CB_color_cycle is None:
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

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    t_us = TimeVector(fs_MHz, len(np.real(signal)))

    # Plot signal
    markerline, stemlines, baseline = axes[0].stem(t_us, np.real(signal), markerfmt="o", label=f"{pulse} real")
    plt.setp(stemlines, "color", CB_color_cycle[0])
    plt.setp(markerline, "color", CB_color_cycle[0])
    plt.setp(stemlines, "linestyle", "dotted")


    # Plot reference signal with alpha and stem style
    t_us = TimeVector(fs_MHz, len(np.real(reference_signal)))
    markerline, stemlines, baseline = axes[0].stem(t_us, np.real(reference_signal), markerfmt="D", label=reference_label)
    plt.setp(stemlines, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(markerline, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(stemlines, "linestyle", "-")  # Change linestyle to stem style
    plt.setp(markerline, "markersize", 3)

    axes[0].legend()
    axes[0].grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.8)
    axes[0].grid(True, which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[0].minorticks_on()

    t_us = TimeVector(fs_MHz, len(np.real(signal)))
    markerline, stemlines, baseline = axes[1].stem(t_us, np.imag(signal), markerfmt="o", label=f"{pulse} imag")
    plt.setp(stemlines, "color", CB_color_cycle[0])
    plt.setp(markerline, "color", CB_color_cycle[0])
    plt.setp(stemlines, "linestyle", "dotted")


    t_us = TimeVector(fs_MHz, len(np.real(reference_signal)))
    markerline, stemlines, baseline = axes[1].stem(t_us, np.imag(reference_signal), markerfmt="D", label=reference_label)
    plt.setp(stemlines, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(markerline, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(stemlines, "linestyle", "-")  # Change linestyle to stem style
    plt.setp(markerline, "markersize", 3)

    axes[1].legend()
    axes[1].grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.8)
    axes[1].grid(True, which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[1].minorticks_on()

    plt.suptitle(title)
    # plt.tight_layout()
    plt.show()

def PlotRealSignalDual(signal, reference_signal, num_stages, title="Signal", reference_label="Deltas", pulse="Pulso", fs_MHz=16.0, CB_color_cycle=None):
    if CB_color_cycle is None:
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

    # Incremental counter to keep track of the number of times the function has been called
    PlotRealSignalDual.call_counter = getattr(PlotRealSignalDual, "call_counter", 0)

    t_us = np.linspace(0, len(signal)/fs_MHz, len(signal))

    # Create the figure if it's the first call
    if PlotRealSignalDual.call_counter == 0:
        fig, axes = plt.subplots(num_stages, 1, figsize=(15, 8*num_stages//2), sharex=True)
        PlotRealSignalDual.fig = fig
        PlotRealSignalDual.axes = axes
    else:
        fig = PlotRealSignalDual.fig
        axes = PlotRealSignalDual.axes

    # Calculate row and column index for the subplot
    row_idx = int(PlotRealSignalDual.call_counter)


    t_us = np.linspace(0, len(signal)/fs_MHz, len(signal))

    # Plot signal
    markerline, stemlines, baseline = axes[row_idx].stem(t_us, np.real(signal), markerfmt="o", label=f"{pulse} real" )
    plt.setp(stemlines, "color", CB_color_cycle[0])
    plt.setp(markerline, "color", CB_color_cycle[0])
    plt.setp(stemlines, "linestyle", "dotted")


    # Plot reference signal with alpha and stem style
    t_us = np.linspace(0, len(reference_signal)/fs_MHz, len(reference_signal))
    markerline, stemlines, baseline = axes[row_idx].stem(t_us, np.real(reference_signal), markerfmt="D", label=reference_label)
    plt.setp(stemlines, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(markerline, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(stemlines, "linestyle", "-")  # Change linestyle to stem style
    plt.setp(markerline, "markersize", 3)

    axes[row_idx].legend()
    axes[row_idx].grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.8)
    axes[row_idx].grid(True, which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[row_idx].minorticks_on()

    # Set title to the center subplot of each row
    axes[row_idx].set_title(title, loc='center')

    # Increment the call counter
    PlotRealSignalDual.call_counter += 1

    # Show the figure after all calls
    if PlotRealSignalDual.call_counter == num_stages:
        PlotRealSignalDual.call_counter = 0
        plt.tight_layout()
        plt.show()

def create_packet(byte_seq, spar):
    pre = np.zeros(spar['n_pre'], dtype=int)
    pre[1::2] = 1

    sfd = np.zeros(spar['n_sfd'], dtype=int)
    if spar['n_pre'] % 2 == 0:
        sfd[0::2] = 1
    else:
        sfd[1::2] = 1

    data = np.hstack([np.array(list(np.binary_repr(byte, 8)), dtype=int) for byte in byte_seq])

    packet = np.hstack([pre, sfd, data])

    x = 2 * packet - 1

    return x, data

def modulator(byte_seq, spar):
    assert np.all(np.array(byte_seq) <= 255), "All values in byte_seq must be between 0 and 255"

    x, data = create_packet(byte_seq, spar)
    xx = np.zeros(len(x) * spar['n_pulse'])
    xx[::spar['n_pulse']] = x
    xxx = np.convolve(xx, spar['pulse'])

    # Igualar longitudes rellenando xx con ceros
    pad_size = (len(xxx) - len(xx)) // 2
    xx_padded = np.pad(xx, (pad_size, len(xxx) - len(xx) - pad_size), mode='constant')

    mis = {'d': xx_padded, 'bits': data}

    return xxx, mis



# Placeholder function for pulse shaping
def pulse(Ts, Tsymb, type : str):
    fs_MHz = 1/Ts/1000
    f0_MHz = 1/Tsymb/1000
    samples = np.ceil(fs_MHz/f0_MHz)

    pulses = {
        "sq": SquarePulse(fs_MHz=fs_MHz, f0_MHz=f0_MHz, samples=samples),
        "sin": SinePulse(fs_MHz=fs_MHz, f0_MHz=f0_MHz, samples=samples),
        "tr": TriangularPulse(fs_MHz=fs_MHz, f0_MHz=f0_MHz, samples=samples),
        "rc": RaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_MHz, beta=0.5, samples=samples),
        "rrc": RootRaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_MHz, beta=0.5, samples=samples),
    }
    if type in pulses:
        pulse_shape = pulses[type]
        return pulse_shape, len(pulse_shape)
    else:
        raise KeyError("Key is not valid")



###############################################################################################

# System Parameters
Tsymb = 1e-3  # Symbol Period
Ts = Tsymb / 16  # Sampling Period
fs_MHz = int(1/Ts/1000)
f0_MHz = int(1/Tsymb/1000)
E_pulse = 1e-6
selected_channel_type = 1  # 0: Delta, 1: Pasa Bajos
delta_phase = 0
samples = int(np.ceil(fs_MHz/f0_MHz))
PWR_N = 0.01

if selected_channel_type == 0:  # Si se selecciona Delta
    if not 0 <= delta_phase < 360:
        delta_phase = float(input("Ingrese la cantidad de grados [0 a 360) para el tipo de canal Delta: "))
    channel = "Delta"
    h = DeltaPulse(fs_MHz=fs_MHz, f0_MHz=f0_MHz, samples=samples, phase=delta_phase)
else:  # Si se selecciona Pasa Bajos
    channel = "Pasa Bajos"
    h = LowPassFilterFIR(fs_MHz=fs_MHz, fc_MHz=fs_MHz/20, order=samples)

def channel(x, pwr_n, h):
    y = np.convolve(x, h)
    y = AddPwrNoise(y, pwr_n)
    return y

pulse_shape, n_fir = pulse(Ts, Tsymb, 'rrc')

spar = {
    'Tsymb': Tsymb,
    'Ts': Ts,
    'n_bytes': 4,
    'n_pre': 16,
    'n_sfd': 2,
    'pulse': pulse_shape,
    'n_pulse': int(Tsymb / Ts),
    'E_pulse': E_pulse,
    'n_fir': n_fir,
    'det_th': 0.25,
    'pll': {'kp': 0.7, 'ki': 0.01, 'delay': 0}
}

# Simulation Parameters
N_TX = 3
N_ZEROS = 123

# Discrete Time
kzeros = Ts * N_ZEROS
khalfp = Ts * (n_fir - 1) / 2
kmod = Tsymb * (spar['n_pre'] + spar['n_sfd'] + spar['n_bytes'] * 8)
k0 = kzeros + khalfp
kend = kzeros + khalfp + kmod
data_sent = []
# Modulator
x = np.array([])
k = np.array([])
d = []
bits_sent = []
kk = np.arange(k0, kend, Tsymb)
for i in range(0, N_TX):
    bytes_seq = np.arange(1 + i, spar['n_bytes'] + 1 + i)
    data_sent.extend(bytes_seq)
    xaux, mis = modulator(bytes_seq, spar)
    x = np.concatenate((x, np.zeros(N_ZEROS), xaux))
    d.extend([np.zeros(N_ZEROS), mis['d']])
    bits_sent.extend(mis["bits"])
    k = np.concatenate((k, kk + (kend + khalfp) * i))

d = np.concatenate(d) # Unificar d en un solo array si es necesario

PlotRealSignalDual(x, d, num_stages=3)

c = channel(x, PWR_N, h)

PlotRealSignalDual(c, d,num_stages=3)

hat_bytes, dis = demodulator(c, spar)

bit_received = np.unpackbits(hat_bytes)


elementos_perdidos = abs(len(data_sent) - len(hat_bytes))
print(f"Elementos perdidos: {elementos_perdidos}")
if(len(hat_bytes) > len(data_sent)):
    elementos_distintos = np.count_nonzero(np.array(hat_bytes[:len(data_sent)]) != np.array(data_sent))
else:
    elementos_distintos = np.count_nonzero(np.array(hat_bytes) != np.array(data_sent[:len(hat_bytes)]))
print(f"Elementos distintos: {elementos_distintos}")
print(f"Porcentaje de error: {(elementos_distintos + elementos_perdidos)/len(data_sent)*100}%")
PlotRealSignalDual(dis['y_mf'], d, num_stages=3)
PlotRealSignal(hat_bytes, data_sent[:len(hat_bytes)], title="Bytes", reference_label="Sent Bytes", pulse="Byte", fs_MHz=16.0, CB_color_cycle=CB_color_cycle)
PlotRealSignal(bits_sent, bit_received, title="bits", reference_label="Bits sent", pulse="Bit", fs_MHz=16.0, CB_color_cycle=CB_color_cycle)
