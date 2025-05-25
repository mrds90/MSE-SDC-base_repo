## @file pulse_generator.py
#  @brief This file contains the definition of the PulseGenerator class.
#  @details The PulseGenerator class is responsible for generating different pulse shapes used in digital communications,
#           including square, sine, triangular, raised cosine, and root raised cosine pulses. It supports different sampling
#           and symbol periods, and provides access to pulse shape and properties useful for FIR filter design.
# @author Marcos Dominguez
# @author Lucas Monzon Languasco

import numpy as np

class PulseGenerator:
    ## @brief Constructor for PulseGenerator
    #  @param pulse (str) Type of pulse: 'sq', 'sin', 'tr', 'rc', or 'rrc'
    #  @param Ts (float) Sampling interval
    #  @param Tsymb (float) Symbol period
    def __init__(self, pulse: str, Ts: float, Tsymb: float):
        self._pulse, self._n_fir = self._genetate_pulse(Ts, Tsymb, pulse)
        self._ts = Ts
        self._tsymb = Tsymb
        self._n_pulse = int(Tsymb / Ts)

    ## @brief Generates the selected pulse shape
    #  @param Ts (float) Sampling interval
    #  @param Tsymb (float) Symbol period
    #  @param type (str) Pulse type
    #  @return tuple[np.ndarray, int] Pulse shape and its length
    def _genetate_pulse(self, Ts: float, Tsymb: float, type: str) -> tuple[np.ndarray, int]:
        fs_MHz = 1/Ts/1000
        f0_MHz = 1/Tsymb/1000
        samples = int(np.ceil(fs_MHz/f0_MHz))
        pulse_shape = None
        match(type):
            case "sq":
                pulse_shape = self.square_pulse(fs_MHz, f0_MHz, samples)
            case "sin":
                pulse_shape = self.sine_pulse(fs_MHz, f0_MHz, samples)
            case "tr":
                pulse_shape = self.triangular_pulse(fs_MHz, f0_MHz, samples)
            case "rc":
                pulse_shape = self.raised_cosine_pulse(fs_MHz, f0_MHz, 0.5, samples)
            case "rrc":
                pulse_shape = self.root_raised_cosine_pulse(fs_MHz, f0_MHz, 0.5, samples)
            case _:
                raise ValueError(f"Invalid pulse type: {type}. Use 'sq', 'sin', 'tr', 'rc', or 'rrc'.")
        return pulse_shape, len(pulse_shape)

    ## @brief Generates a square pulse
    #  @param fs_MHz (float) Sampling frequency in MHz
    #  @param f0_MHz (float) Symbol frequency in MHz
    #  @param samples (int) Number of samples
    #  @return np.ndarray Square pulse
    def square_pulse(self, fs_MHz: float, f0_MHz: float, samples: int) -> np.ndarray:
        t_us = self.time_vector(fs_MHz, samples)
        signal = np.sign(np.sin(2 * np.pi * (f0_MHz/2) * t_us))
        signal[signal < 0] = 0
        return signal

    ## @brief Generates a sine pulse
    #  @param fs_MHz (float) Sampling frequency in MHz
    #  @param f0_MHz (float) Symbol frequency in MHz
    #  @param samples (int) Number of samples
    #  @return np.ndarray Sine pulse
    def sine_pulse(self, fs_MHz: float, f0_MHz: float, samples: int) -> np.ndarray:
        t_us = self.time_vector(fs_MHz, samples)
        return np.sin(2 * np.pi * (f0_MHz/2) * t_us)

    ## @brief Generates a triangular pulse
    #  @param fs_MHz (float) Sampling frequency in MHz
    #  @param f0_MHz (float) Symbol frequency in MHz
    #  @param samples (int) Number of samples
    #  @return np.ndarray Triangular pulse
    def triangular_pulse(self, fs_MHz: float, f0_MHz: float, samples: int) -> np.ndarray:
        t_us = self.time_vector(fs_MHz, samples)
        return (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * (f0_MHz/2) * t_us))

    ## @brief Generates a raised cosine pulse
    #  @param fs_MHz (float) Sampling frequency in MHz
    #  @param f0_MHz (float) Symbol frequency in MHz
    #  @param beta (float) Roll-off factor
    #  @param samples (int) Number of samples
    #  @return np.ndarray Raised cosine pulse
    def raised_cosine_pulse(self, fs_MHz: float, f0_MHz: float, beta: float, samples: int) -> np.ndarray:
        t = np.arange(-3 * samples + 1,  3 * samples + 1) / fs_MHz
        T0=1/f0_MHz
        signal = np.zeros_like(t)
        beta_term = np.pi * beta * t / T0
        mask = np.abs(1 - (2 * beta * t / T0) ** 2) != 0
        signal[mask] = (1 / T0) * np.sinc(t[mask] / T0) * np.cos(beta_term[mask]) / (1 - (2 * beta * t[mask] / T0) ** 2)
        signal[t==T0/2/beta] = np.pi/4/T0*np.sinc(1/2/beta)
        signal[t==-T0/2/beta] = np.pi/4/T0*np.sinc(1/2/beta)
        return signal

    ## @brief Generates a root raised cosine pulse
    #  @param fs_MHz (float) Sampling frequency in MHz
    #  @param f0_MHz (float) Symbol frequency in MHz
    #  @param beta (float) Roll-off factor
    #  @param samples (int) Number of samples
    #  @return np.ndarray Root raised cosine pulse
    def root_raised_cosine_pulse(self, fs_MHz: float, f0_MHz: float, beta: float, samples: int) -> np.ndarray:
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
        return signal

    ## @brief Generates a time vector for pulse generation
    #  @param fs_MHz (float) Sampling frequency in MHz
    #  @param samples (int) Number of samples
    #  @return np.ndarray Time vector
    @staticmethod
    def time_vector(fs_MHz: float, samples: int) -> np.ndarray:
        return np.arange(1, samples + 1) / fs_MHz

    ## @brief Returns the number of samples per symbol (pulse length)
    #  @return int
    @property
    def n_pulse(self) -> int:
        return self._n_pulse

    ## @brief Returns the generated pulse shape
    #  @return np.ndarray
    @property
    def pulse(self) -> np.ndarray:
        return self._pulse

    ## @brief Returns the sampling interval
    #  @return float
    @property
    def Ts(self) -> float:
        return self._ts

    ## @brief Returns the symbol period
    #  @return float
    @property
    def Tsymb(self) -> float:
        return self._tsymb

    ## @brief Returns the length of the FIR pulse filter
    #  @return int
    @property
    def n_fir(self) -> int:
        return self._n_fir
