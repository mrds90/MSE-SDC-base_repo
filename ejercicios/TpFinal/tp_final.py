import numpy as np
from scipy.signal import convolve
import os
import platform
if "microsoft" in platform.uname().release.lower() or "WSL" in os.environ:
    import matplotlib
    matplotlib.use("TkAgg")  # Puedes probar también con 'Qt5Agg'
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
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

def modulator(byte_seq, spar):
    assert np.all(np.array(byte_seq) <= 255), "All values in byte_seq must be between 0 and 255"
     
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
    
    xx = np.zeros(len(x) * spar['n_pulse'])
    xx[::spar['n_pulse']] = x
    xxx = np.convolve(xx, spar['pulse'])

    # Igualar longitudes rellenando xx con ceros
    pad_size = (len(xxx) - len(xx)) // 2
    xx_padded = np.pad(xx, (pad_size, len(xxx) - len(xx) - pad_size), mode='constant')
    
    print("x_n", len(xxx))
    print("xx_padded", len(xx_padded))

    mis = {'d': xx_padded}
    
    return xxx, mis



# Placeholder function for pulse shaping
def pulse(Ts, Tsymb, type : str):
    fs_MHz = 1/Ts/1000
    f0_Mhz = 1/Tsymb/1000
    samples = np.ceil(fs_MHz/f0_Mhz)
    print("fs_MHz",fs_MHz)
    print("f0_Mhz",f0_Mhz)
    print("samples",samples)
    # fs_MHz = 16.0
    # f0_Mhz = 1
    # samples = 16
    pulses = {
        "sq": SquarePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
        "sin": SinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
        "tr": TriangularPulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
        "rc": RaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, beta=0.5, samples=samples),
        "rrc": RootRaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, beta=0.5, samples=samples),
    }
    if type in pulses:
        pulse_shape = pulses[type]
        print("max_pulse", max(pulse_shape))
        
        return pulse_shape, len(pulse_shape)
    else:
        raise KeyError("Key is not valid")



###############################################################################################

# System Parameters
Tsymb = 1e-3  # Symbol Period
Ts = Tsymb / 16  # Sampling Period
E_pulse = 1e-6

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

# Modulator
bytes_seq = np.arange(1, spar['n_bytes'] + 1)
xaux, mis = modulator(bytes_seq, spar)
x = np.concatenate((np.zeros(N_ZEROS), xaux))
d = mis['d']
kk = np.arange(k0, kend, Tsymb)
k = kk

PlotRealSignal(xaux, d)