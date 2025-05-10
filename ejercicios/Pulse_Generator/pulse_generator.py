import numpy as np
from scipy.signal import firwin
from scipy.signal import lfilter
import matplotlib.pyplot as plt


def GenerateBinarySignal(length: int, complex: bool = False) -> np.ndarray:
    """
    Generates a random binary signal of specified length.

    Parameters:
        length (int): Length of the binary signal.
        complex (bool): Indicates whether the signal should have a complex part. Default is False.

    Returns:
        numpy.ndarray: Random binary signal of specified length.
    """

    # Generate a sequence of random binary values (0 or 1)
    binary_sequence_real = np.random.randint(2, size=length)

    # Generate a sequence of random binary values (0 or 1) for imaginary part if complex_signal is True
    if complex:
        binary_sequence_imag = np.random.randint(2, size=length)
    else:
        binary_sequence_imag = np.zeros(length)

    # Map binary values to {-1, 1}
    binary_sequence_real[binary_sequence_real == 0] = -1
    binary_sequence_imag[binary_sequence_imag == 0] = -1

    # Combine real and imaginary parts to form a complex signal
    binary_signal = binary_sequence_real + 1j * binary_sequence_imag

    return binary_signal


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
    if np.iscomplexobj(binary_signal):
        modified_signal = np.zeros(modified_length, dtype=complex)
    else:
        modified_signal = np.zeros(modified_length, dtype=float)  # Initialize a signal of zeros with the modified length

    # Create an array to represent indices of each bit in the modified signal
    bit_indices = np.arange(length) * M

    # Assign values to the initial position of each bit
    modified_signal[bit_indices] = binary_signal * 1.0

    return modified_signal

def AddPwrNoise(x, pwr):
    sigma = np.sqrt(pwr)
    y = x + sigma * np.random.randn(len(x))

    return y

def AddChannelNoise(signal: np.ndarray, SNR_dB: float) -> np.ndarray:
    """
    Adds additive white Gaussian noise (AWGN) to the signal.

    Parameters:
        signal (numpy.ndarray): Input signal.
        SNR_dB (float): Signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: Signal with added noise.
    """
    if np.iscomplexobj(signal):
        # Calcular la potencia de la señal
        signal_power = np.sum(np.abs(signal)**2) / len(signal)

        # Calcular la potencia del ruido según la relación señal-ruido (SNR)
        noise_power = signal_power / (10 ** (SNR_dB / 10))

        # Generar ruido gaussiano con la misma longitud que la señal
        noise_real = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise = noise_real + 1j * noise_imag

        # Sumar el ruido a la señal
        noisy_signal = signal + noise
    else:
        # Si la señal es real, agregar solo ruido gaussiano
        noise_power = 10 ** (SNR_dB / 10)
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        noisy_signal = signal + noise

    return noisy_signal



def TimeVector(fs_MHz: float, samples: int) -> np.ndarray:
    t_us = np.arange(1, samples + 1) / fs_MHz
    return t_us

def DeltaPulse(fs_MHz: float, f0_MHz: float, samples: int, phase: float) -> np.ndarray:

    if phase < 0 or phase >= 360:
        raise ValueError("El argumento 'phase' debe estar en el rango de 0 a 360 grados.")

    # Calcula el período de la señal de entrada
    T = 1 / f0_MHz
    samples_by_cycle = T * fs_MHz

    # Determina el índice más cercano para el desfase de fase
    phase_shift_index = int(round(phase / 360 * samples_by_cycle))

    # Inicializa el vector de salida con ceros
    delta_pulse = np.zeros(samples)

    # Establece el valor del delta en el índice correspondiente
    delta_pulse[phase_shift_index] = 1

    return delta_pulse


def SquarePulse(fs_MHz: float, f0_MHz: float, samples: int, complex: bool = False) -> np.ndarray:
    t_us = TimeVector(fs_MHz, samples)
    signal = np.sign(np.sin(2 * np.pi * (f0_MHz/2) * t_us))
    signal[signal < 0] = 0

    if(complex == True):
        env_complex = np.exp(1j * 2 * np.pi * (f0_MHz/2) * t_us)
        signal = signal * env_complex

    return signal

    return signal


def TriangularPulse(fs_MHz: float, f0_MHz: float, samples: int, complex: bool = False) -> np.ndarray:
    t_us = TimeVector(fs_MHz, samples)
    signal = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * (f0_MHz/2) * t_us))

    if(complex == True):
        env_complex = np.exp(1j * 2 * np.pi * f0_MHz * t_us)
        signal = signal * env_complex

    return signal


def SinePulse(fs_MHz: float, f0_MHz: float, samples: int, complex: bool = False) -> np.ndarray:

    t_us = TimeVector(fs_MHz, samples)
    signal = np.sin(2 * np.pi * (f0_MHz/2) * t_us)

    if(complex == True):
        signal = signal * (np.sin(2 * np.pi * (f0_MHz / 2) * t_us))

    return signal


def RaisedCosinePulse(fs_MHz: float, f0_MHz:float, beta: float, samples: int, complex: bool = False) -> np.ndarray:

    t = np.arange(-3 * samples + 1,  3 * samples + 1) / fs_MHz
    T0=1/f0_MHz

    signal = np.zeros_like(t)
    beta_term = np.pi * beta * t / T0
    mask = np.abs(1 - (2 * beta * t / T0) ** 2) != 0
    # Aplicar la máscara
    signal[mask] = (1 / T0) * np.sinc(t[mask] / T0) * np.cos(beta_term[mask]) / (1 - (2 * beta * t[mask] / T0) ** 2)
    signal[t==T0/2/beta] = np.pi/4/T0*np.sinc(1/2/beta)
    signal[t==-T0/2/beta] = np.pi/4/T0*np.sinc(1/2/beta)

    if(complex == True):
        env_complex = np.exp(1j * 2 * np.pi * f0_MHz * t)
        signal = signal * env_complex

    return signal


def RootRaisedCosinePulse(fs_MHz: float, f0_MHz:float, beta: float, samples: int, complex: bool = False) -> np.ndarray:
    t = np.arange(-3 * samples + 1,  3 * samples + 1) / fs_MHz
    T0=1/f0_MHz
    a = np.sin(np.pi*t/T0*(1-beta)) + 4*beta*t/T0*np.cos(np.pi*t/T0*(1+beta))
    b = np.pi*t/T0*(1-(4*beta*t/T0)**2)
    mask = b != 0
    signal = np.zeros_like(t)
    signal[mask] = 1 / T0 * a[mask] / b[mask]
    signal [t==0] = 1/T0*(1+beta*(4/np.pi-1))
    signal [t==T0/4/beta] = beta/T0/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/4/beta)+(1-2/np.pi)*np.cos(np.pi/4/beta))
    signal [t==-T0/4/beta] = beta/T0/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/4/beta)+(1-2/np.pi)*np.cos(np.pi/4/beta))

    if(complex == True):
        env_complex = np.exp(1j * 2 * np.pi * f0_MHz * t)
        signal = signal * env_complex

    return signal


def LowPassFilterFIR(fs_MHz: float, fc_MHz: float, order: int) -> np.ndarray:
    """
    Design a low-pass FIR filter.

    Parameters:
        fs_MHz (float): Sampling frequency in MHz.
        fc_MHz (float): Cut-off frequency of the low-pass filter in MHz.
        order (int): Order of the FIR filter.

    Returns:
        np.ndarray: Coefficients of the FIR filter.
    """
    nyquist_frequency = 0.5 * fs_MHz
    normalized_cutoff_frequency = fc_MHz / nyquist_frequency
    fir_coefficients = firwin(order, normalized_cutoff_frequency)
    return fir_coefficients


def pll(input_signal, f0, fs, kp, ki, delay=0):
    """
    Phase-Locked Loop (PLL) implementation.
    :param input_signal: Input signal array
    :param f0: VCO center frequency
    :param fs: Sampling frequency
    :param kp: Proportional constant
    :param ki: Integrator constant
    :param delay: Optional delay parameter
    :return: VCO output and PLL internal signals
    """
    phi_hat = np.zeros(len(input_signal) + delay)
    err = np.zeros(len(input_signal) + delay)
    phd = np.zeros(len(input_signal) + delay)
    vco = np.zeros(len(input_signal) + delay, dtype=complex)
    int_err = np.zeros(len(input_signal) + delay)

    for it in range(1 + delay, len(input_signal)):
        vco[it] = np.exp(-1j * (2 * np.pi * it * f0 / fs + phi_hat[it - 1 - delay]))
        phd[it] = np.imag(input_signal[it] * vco[it])
        int_err[it] = ki * phd[it] + int_err[it - 1]
        err[it] = kp * phd[it] + int_err[it]
        phi_hat[it] = phi_hat[it - 1] + err[it]

    pllis = {
        "phd": phd,
        "err": err,
        "phi_hat": phi_hat
    }

    return vco, pllis

def bits_to_bytes(bit_array):
    """
    Convierte un array de bits en un array de bytes.
    Cada 8 bits se transforman en un número entero.
    """
    # Asegurar que la longitud sea múltiplo de 8
    bit_array = bit_array[:len(bit_array) - (len(bit_array) % 8)]

    # Reshape para agrupar en bloques de 8 bits
    bit_array = bit_array.reshape(-1, 8)

    # Convertir cada grupo de 8 bits en un byte
    byte_array = np.packbits(bit_array, axis=1)

    # Aplanar el array si es necesario
    return byte_array.flatten()

def demodulator(y, spar):
    """
    Demodulator function using PLL.
    :param y: Input signal array
    :param spar: Dictionary containing system parameters
    :return: Demodulated bytes and internal signals
    """
    n_bytes = spar['n_bytes']

    # Matched filter
    y_mf = lfilter(spar['pulse'], [1], y)
    y_mf /= np.sum(spar['pulse']**2)

    # Square and moving average filter
    y_mf_sq = y_mf**2
    n_ma = spar['n_pulse']
    y_mf_sq_ma = lfilter(np.ones(int(n_ma)) / n_ma, [1], y_mf_sq)

    # Read pre-filter and bandpass filter coefficients
    prefilter_data = [[0.01925927815873231,0,-0.01925927815873231],[1,-1.924163036247956,0.9614814515953285]]
    prefilter_b, prefilter_a = prefilter_data[0], prefilter_data[1]
    filter_data = [[0.004884799809161248,0,-0.004884799809161248],[1,-1.838755285386028,0.9902304008963749]]

    filter_b, filter_a = filter_data[0], filter_data[1]

    y_mf_pf = lfilter(prefilter_b, prefilter_a, y_mf)
    y_mf_pf_sq = y_mf_pf**2
    y_mf_pf_sq_bpf = lfilter(filter_b, filter_a, y_mf_pf_sq)

    # np.savetxt("y_mf_pf_sq_bpf.txt", y_mf_pf_sq_bpf)
    # np.savetxt("y_mf_sq_ma.txt", y_mf_sq_ma)
    # np.savetxt("y_mf.txt", y_mf)
    # y_mf_pf_sq_bpf = np.loadtxt("y_mf_pf_sq_bpf.txt")
    # y_mf_sq_ma = np.loadtxt("y_mf_sq_ma.txt")
    # y_mf = np.loadtxt("y_mf.txt")
    # PLL processing

    f0 = 1.0 / spar['Tsymb']
    fs = 1.0 / spar['Ts']

    vco, pllis = pll(y_mf_pf_sq_bpf, f0, fs, spar['pll']['kp'], spar['pll']['ki'], spar['pll']['delay'])

    pll_cos, pll_sin = np.real(vco), np.imag(vco)
    pll_clk_i, pll_clk_q = pll_cos >= 0, pll_sin >= 0

    detection = np.array(y_mf_sq_ma) >= spar['det_th']

    flank_in = pll_clk_i & np.concatenate(([0], ~pll_clk_i[:-1])).astype(bool)

    en_sample = flank_in & detection

    hat_xn = np.array(y_mf[en_sample == 1])
    hat_packet = hat_xn > 0
    sfd = np.zeros(spar['n_sfd'], dtype=int)
    if spar['n_pre'] % 2 == 0:
        sfd[0::2] = 1
    else:
        sfd[1::2] = 1
    sfd = np.array(sfd)
    pattern = np.flip(np.concatenate([~sfd, sfd]) * 2 - 1)
    conv_result = np.convolve(hat_packet * 2 - 1, pattern, mode='valid')

    hat_bytes = []
    i = 0
    while i < len(conv_result):
        if conv_result[i] <= -3:
            while (i < len(conv_result)) and (hat_packet[i] != 0) and (conv_result[i] < 0):
                i += 1

            if i >= len(conv_result):
                break

            i += 1

            packet = hat_packet[i : i + 8 * spar['n_bytes']]
            hat_bytes.extend(np.packbits(packet, bitorder='big'))

            i += 8 * spar['n_bytes']

        else:
            i += 1

    dis = {
        'y_mf': y_mf,
        'y_mf_pf': y_mf_pf,
        'y_mf_pf_sq': y_mf_pf_sq,
        'y_mf_pf_sq_bpf': y_mf_pf_sq_bpf,
        'y_mf_sq_ma': y_mf_sq_ma,
        'vco': vco,
        'pll': pllis,
        'hat_packet': hat_packet,
    }
    return hat_bytes, dis