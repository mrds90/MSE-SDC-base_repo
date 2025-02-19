import random
import numpy as np
import os
import platform
if "microsoft" in platform.uname().release.lower() or "WSL" in os.environ:
    import matplotlib
    matplotlib.use("TkAgg")  # Puedes probar también con 'Qt5Agg'
import matplotlib.pyplot as plt

from scipy.fft import fft
import sys
sys.path.append('../..')
from Pulse_Generator import *

# # invalid 
delta_phase = 360
selected_channel_type = -1
selected_pulse_index = 6
SNR_dB = -1  # Valido solo mayores a 0.


# delta_phase = 0
# selected_channel_type = 0
# selected_pulse_index = 1
# SNR_dB = 50

fs_MHz = 16.0
f0_Mhz = 1
samples = 16
N_SIGNAL_B = 10
N_ZEROS = samples - 1

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


def PlotComplexSignalDual(signal, reference_signal, num_stages, title="Signal", reference_label="Deltas", pulse="Pulso", fs_MHz=16.0, CB_color_cycle=None):
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
    PlotComplexSignalDual.call_counter = getattr(PlotComplexSignalDual, "call_counter", 0)
    
    t_us = np.linspace(0, len(signal)/fs_MHz, len(signal))

    # Create the figure if it's the first call
    if PlotComplexSignalDual.call_counter == 0:
        fig, axes = plt.subplots(num_stages, 2, figsize=(15, 8*num_stages//2), sharex=True)
        PlotComplexSignalDual.fig = fig
        PlotComplexSignalDual.axes = axes
    else:
        fig = PlotComplexSignalDual.fig
        axes = PlotComplexSignalDual.axes

    # Calculate row and column index for the subplot
    row_idx = int(PlotComplexSignalDual.call_counter)
        

    t_us = np.linspace(0, len(signal)/fs_MHz, len(signal))

    # Plot signal
    markerline, stemlines, baseline = axes[row_idx, 0].stem(t_us, np.real(signal), markerfmt="o", label=f"{pulse} real" )
    plt.setp(stemlines, "color", CB_color_cycle[0])
    plt.setp(markerline, "color", CB_color_cycle[0])
    plt.setp(stemlines, "linestyle", "dotted")
    

    # Plot reference signal with alpha and stem style
    t_us = np.linspace(0, len(reference_signal)/fs_MHz, len(reference_signal))
    markerline, stemlines, baseline = axes[row_idx, 0].stem(t_us, np.real(reference_signal), markerfmt="D", label=reference_label)
    plt.setp(stemlines, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(markerline, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(stemlines, "linestyle", "-")  # Change linestyle to stem style
    plt.setp(markerline, "markersize", 3)

    axes[row_idx, 0].legend()
    axes[row_idx, 0].grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.8)
    axes[row_idx, 0].grid(True, which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[row_idx, 0].minorticks_on()

    t_us = np.linspace(0, len(signal)/fs_MHz, len(signal))
    markerline, stemlines, baseline = axes[row_idx, 1].stem(t_us, np.imag(signal), markerfmt="o", label=f"{pulse} imag")
    plt.setp(stemlines, "color", CB_color_cycle[0])
    plt.setp(markerline, "color", CB_color_cycle[0])
    plt.setp(stemlines, "linestyle", "dotted")
    

    t_us = np.linspace(0, len(reference_signal)/fs_MHz, len(reference_signal))
    markerline, stemlines, baseline = axes[row_idx, 1].stem(t_us, np.imag(reference_signal), markerfmt="D", label=reference_label)
    plt.setp(stemlines, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(markerline, "color", CB_color_cycle[1], alpha=0.7)
    plt.setp(stemlines, "linestyle", "-")  # Change linestyle to stem style
    plt.setp(markerline, "markersize", 3)

    axes[row_idx, 1].legend()
    axes[row_idx, 1].grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.8)
    axes[row_idx, 1].grid(True, which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[row_idx, 1].minorticks_on()

    # Set title to the center subplot of each row
    axes[row_idx, 0].set_title(title, loc='center')

    # Increment the call counter
    PlotComplexSignalDual.call_counter += 1

    # Show the figure after all calls
    if PlotComplexSignalDual.call_counter == num_stages:
        PlotComplexSignalDual.call_counter = 0
        plt.tight_layout()
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


# Parte 1: Generación de señales y simulación del sistema


# Generar pulsos
pulses = (
    SquarePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    SinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    TriangularPulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
    RaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, beta=0.5, samples=samples),
    RootRaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, beta=0.5, samples=samples),
)


# Validación de parámetros
if selected_channel_type not in (0, 1):
    selected_channel_type = int(input("0 - Delta\n1 - Pasa Bajos\nIngrese el número correspondiente al tipo de canal: "))
    
if selected_channel_type == 0:  
    if not 0 <= delta_phase < 360:
        delta_phase = float(input("Ingrese la cantidad de grados [0 a 360) para el tipo de canal Delta: "))
else:
    delta_phase = 0
    
if not 0 <= selected_pulse_index < len(pulses):
    print("Seleccione el tipo de pulso:")
    for i, pulse in enumerate(pulse_names):
        print(f"{i + 1}. {pulse}")
    selected_pulse_index = int(input("Ingrese el número correspondiente al tipo de pulso: ")) - 1

if SNR_dB <= 0:
    SNR_dB = float(input("Ingrese la relación señal-ruido (SNR) en dB (debe ser mayor a 0): "))

# Definir el pulso seleccionado
selected_pulse = pulses[selected_pulse_index]

# Definir el canal según el tipo seleccionado
if selected_channel_type == 0:  # Si se selecciona Delta
    if not 0 <= delta_phase < 360:
        delta_phase = float(input("Ingrese la cantidad de grados [0 a 360) para el tipo de canal Delta: "))
    channel = "Delta"
    h = DeltaPulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples, phase=delta_phase)
else:  # Si se selecciona Pasa Bajos
    channel = "Pasa Bajos"
    h = LowPassFilterFIR(fs_MHz=fs_MHz, fc_MHz=fs_MHz / 20, order=samples)

# Solicitar la relación señal-ruido (SNR)
if SNR_dB <= 0:
    SNR_dB = float(input("Ingrese la relación señal-ruido (SNR) en dB (debe ser mayor a 0): "))

# Realizar el resto del proceso con los parámetros válidos




# Generación de la señal binaria aleatoria
b = GenerateBinarySignal(N_SIGNAL_B, complex=True)

# Insertar ceros y asignar valores a la señal binaria
d = InsertZerosAndAssignValues(b, samples)

# Realizar la convolución para el caso seleccionado
discard_length = ((len(selected_pulse)) - 1) // 2 #hhalfp
d_sync = np.zeros(discard_length + len(d), dtype=np.complex128)
d_sync[discard_length:] = d
x = np.convolve(d, selected_pulse)[:-(discard_length + 1)]


PlotComplexSignalDual(x, d_sync,num_stages=4, reference_label="Deltas", title=f"Señal con pulso \"{pulse_names[selected_pulse_index]}\" a transmitir",pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)
# PlotComplexSignal(x, d, reference_label="Deltas", title=f"Señal con pulso \"{pulse_names[selected_pulse_index]}\" a transmitir",pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)

discard_lengths_h = (len(h) - 1)
if(selected_channel_type == 0):
    xh = np.convolve(x, h)[:-discard_lengths_h]
else:
    xh = np.convolve(x, h)[int(discard_lengths_h/2)+1:-int(discard_lengths_h/2)]



PlotComplexSignalDual(xh, d_sync, num_stages=4, reference_label="Deltas", title=f"Señal con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" y defasaje de {delta_phase}", pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)
# PlotComplexSignal(xh, d, reference_label="Deltas", title=f"Señal con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" y defasaje de {delta_phase}", pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)

# Agregar ruido al resultado
c = AddChannelNoise(xh, SNR_dB)


# Graficar el resultado
PlotComplexSignalDual(c, d_sync,num_stages=4, reference_label="Deltas",title=f"Señal con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" con un SNR de {SNR_dB} [dBs] y defasaje de {delta_phase}", pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)
# PlotComplexSignal(c, d, reference_label="Deltas",title=f"Señal con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" con un SNR de {SNR_dB} [dBs] y defasaje de {delta_phase}", pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)


discard_length = ((len(selected_pulse)) - 1) // 2
selected_pulse_normalized = selected_pulse / np.sum(np.abs(selected_pulse))
 
SquarePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
SinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
TriangularPulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, samples=samples),
RaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, beta=0.5, samples=samples),
RootRaisedCosinePulse(fs_MHz=fs_MHz, f0_MHz=f0_Mhz, beta=0.5, samples=samples),

adjust_factor = (1, 1.267869510755348, 1.4879795574889882, 1.5195511971782591, 1.3614806367722807)

y = adjust_factor[selected_pulse_index] * np.convolve(c, selected_pulse_normalized)[discard_length:-(discard_length+1)]  # Scale the pulse

b_received = y[d_sync!=0]

b_with_noise = ((np.real(b_received)))+1j*((np.imag(b_received)))

b_received = np.sign((np.real(b_received)))+1j*np.sign((np.imag(b_received)))



error = np.array([np.sign((np.real(b_received))) != np.sign((np.real(b)))]).sum() + np.array([np.sign((np.imag(b_received))) != np.sign((np.imag(b)))]).sum()
print(f"cantidad de errores con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" con un SNR de {SNR_dB} [dBs] y defasaje de {delta_phase}  : {error} ({error/(2*N_SIGNAL_B) * 100:.2f}%)")



PlotComplexSignalDual(y, d_sync,num_stages=4, reference_label="Deltas",title=f"Señal recibida con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" con un SNR de {SNR_dB} [dBs] y defasaje de {delta_phase} después del FIR", pulse=f"{pulse_names[selected_pulse_index]}", fs_MHz=fs_MHz)

PlotComplexSignal(b_with_noise, b, reference_label="Original",title=f"Bits recibidos con pulso \"{pulse_names[selected_pulse_index]}\" en canal \"{channel}\" con un SNR de {SNR_dB} [dBs] y defasaje de {delta_phase} después del FIR", pulse="Bits recibidos", fs_MHz=fs_MHz)

plt.scatter(np.real(b), np.imag(b), marker='o', color=CB_color_cycle[1], label="Transmitido")
plt.scatter(np.real(b_with_noise), np.imag(b_with_noise), marker='D', s=3, color=CB_color_cycle[0], label="Recibido")

plt.axhline(0, color='black', linewidth=1)  # Eje horizontal en y=0
plt.axvline(0, color='black', linewidth=1)  # Eje vertical en x=0

# Agregar título y etiquetas de los ejes
plt.title("Constelación")
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginaria")

# Agregar la leyenda
plt.legend()

# Agregar grid mayor y menor
plt.grid(True, which='major', color='gray', linestyle='-', linewidth=0.8, alpha=0.8)
plt.grid(True, which='minor', color='gray', linestyle='--', linewidth=0.4, alpha=0.5)
plt.minorticks_on()

# Mostrar el gráfico
plt.show()