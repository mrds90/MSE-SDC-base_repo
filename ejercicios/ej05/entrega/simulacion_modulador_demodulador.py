import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import sys
sys.path.append('../..')
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

pulse_names = ("Square", "Sine", "Triangular", "Raised Cosine", "Root Raised Cosine")
channel_names = ("Delta", "Delta (90°)", "Low Pass Filter")


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

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
    markerline, stemlines, baseline = axes[0].stem(t_us, np.real(signal), markerfmt="o", label=f"{pulse} real", )
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
    plt.tight_layout()
    plt.show()




# Parte 1: Generación de señales y simulación del sistema

# Definición de parámetros del sistema
fs_MHz = 16.0
f0_Mhz = 1
samples = 16
N_SIGNAL_B = 6
SNR_dB = 10

# Generación de la señal binaria aleatoria
b = GenerateBinarySignal(N_SIGNAL_B, complex=True)

# Insertar ceros y asignar valores a la señal binaria
d = InsertZerosAndAssignValues(b, samples)

# Generar pulsos
pulses = (
    SquarePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    SinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    TriangularPulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    RaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, beta=0.5, samples=samples),
    RootRaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, beta=0.5, samples=samples),
)

delta_phase = 0
channel = "Delta"
# Menú interactivo para elegir el tipo de pulso
print("Seleccione el tipo de pulso:")
for i, pulse in enumerate(pulse_names):
    print(f"{i+1}. {pulse}")

selected_pulse_index = input("Ingrese el número correspondiente al tipo de pulso: ")
try:
    selected_pulse_index = int(selected_pulse_index) - 1
    if selected_pulse_index < 0 or selected_pulse_index >= len(pulse_names):
        raise ValueError
except ValueError:
    print("Entrada inválida. Se seleccionará el primer tipo de pulso por defecto.")
    selected_pulse_index = 0

selected_pulse = pulses[selected_pulse_index]

# Definir la función para mostrar el menú de selección del tipo de canal Delta
def choose_delta_phase() -> float:
    while True:
        delta_phase = input("Ingrese la cantidad de grados [0 a 360) para el tipo de canal Delta: ")
        try:
            delta_phase = float(delta_phase)
            if 0 <= delta_phase < 360:
                return delta_phase
            else:
                print("La cantidad de grados debe estar en el rango de 0 a 360.")
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número válido.")

# Menú interactivo para elegir el tipo de canal
print("\nSeleccione el tipo de canal:")
print("1. Delta")
print("2. Pasa Bajos")

selected_channel_type = input("Ingrese el número correspondiente al tipo de canal: ")
try:
    selected_channel_type = int(selected_channel_type)
    if selected_channel_type not in (1, 2):
        raise ValueError
except ValueError:
    print("Entrada inválida. Se seleccionará Delta por defecto.")
    selected_channel_type = 1

if selected_channel_type == 1:  # Si se selecciona Delta
    delta_phase = choose_delta_phase()
    h = DeltaPulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples, phase=delta_phase)
    # plt.scatter(np.arange(samples), h)
    # plt.show()
else:  # Si se selecciona Pasa Bajos
    channel = "Low Pass Filter"
    h = LowPassFilterFIR(fs_MHz=fs_MHz, fc_MHz=fs_MHz / 4, order=samples)

# Menú interactivo para elegir la relación señal-ruido (SNR)
selected_snr = input("Ingrese la relación señal-ruido (SNR) en dB: ")
try:
    SNR_dB = float(selected_snr)
except ValueError:
    print("Entrada inválida. Se utilizará un SNR de 10 dB por defecto.")
    SNR_dB = 10

# Realizar el resto del proceso con el SNR y el tipo de canal seleccionados



# Realizar la convolución para el caso seleccionado
discard_length = ((len(selected_pulse)) - 1) // 2
x = np.convolve(d, selected_pulse)[discard_length:-(discard_length + 1)]

PlotComplexSignal(x, d, reference_label="Deltas", title=f"Señal con pulso \"{pulse_names[selected_pulse_index]}\" a transmitir",pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)

discard_lengths_h = (len(h) - 1)
xh = np.convolve(x, h)[:-discard_lengths_h]

PlotComplexSignal(xh, d, reference_label="Deltas", title=f"Señal con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" y defasaje de {delta_phase}", pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)

# Agregar ruido al resultado
c = AddChannelNoise(xh, SNR_dB)

# Graficar el resultado
PlotComplexSignal(c, d, reference_label="Deltas",title=f"Señal con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" con un SNR de {SNR_dB} [dBs] y defasaje de {delta_phase}", pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)


discard_length = ((len(selected_pulse)) - 1) // 2
y = np.convolve(c, selected_pulse)[discard_length:-(discard_length+1)]  # Scale the pulse
y = np.real(y) / np.max(np.abs(np.real(y))) + 1j * np.imag(y) / np.max(np.abs(np.imag(y)))

PlotComplexSignal(y, d, reference_label="Deltas",title=f"Señal recibida con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" con un SNR de {SNR_dB} [dBs] y defasaje de {delta_phase} después del FIR", pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)

b_received = y[d!=0]

b_received = np.sign((np.real(b_received)))+1j*np.sign((np.imag(b_received)))


error = np.array([np.sign((np.real(b_received))) != np.sign((np.real(b)))]).sum() + np.array([np.sign((np.imag(b_received))) != np.sign((np.imag(b)))]).sum()
print(f"cantidad de errores con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" con un SNR de {SNR_dB} [dBs] y defasaje de {delta_phase}  : {error}")

PlotComplexSignal(b_received, b, reference_label="Original",title=f"Bits recibidos con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" con un SNR de {SNR_dB} [dBs] y defasaje de {delta_phase} despues del FIR", pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)
