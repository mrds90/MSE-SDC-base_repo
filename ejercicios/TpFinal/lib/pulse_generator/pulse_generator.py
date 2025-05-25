## @file pulse_generator.py
#  @brief This file contains the definition of the PulseGenerator class.
#  @details The PulseGenerator class is responsible for generating different pulse shapes used in digital communications,
#           including square, sine, triangular, raised cosine, and root raised cosine pulses. It supports different sampling
#           and symbol periods, and provides access to pulse shape and properties useful for FIR filter design.
# @author Marcos Dominguez
# @author Lucas Monzon Languasco

import numpy as np

class PulseGenerator:
    def __init__(self, pulse:str, Ts, Tsymb):
        self._pulse, self._n_fir = self._genetate_pulse(Ts, Tsymb, pulse)
        self._ts = Ts
        self._tsymb = Tsymb
        self._n_pulse = int(Tsymb / Ts)

    def _genetate_pulse(self, Ts, Tsymb, type: str):
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

    def square_pulse(self, fs_MHz, f0_MHz, samples):
        t_us = self.time_vector(fs_MHz, samples)
        signal = np.sign(np.sin(2 * np.pi * (f0_MHz/2) * t_us))
        signal[signal < 0] = 0
        return signal

    def sine_pulse(self, fs_MHz, f0_MHz, samples):
        t_us = self.time_vector(fs_MHz, samples)
        return np.sin(2 * np.pi * (f0_MHz/2) * t_us)

    def triangular_pulse(self, fs_MHz, f0_MHz, samples):
        t_us = self.time_vector(fs_MHz, samples)
        return (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * (f0_MHz/2) * t_us))

    def raised_cosine_pulse(self, fs_MHz, f0_MHz, beta, samples):
        t = np.arange(-3 * samples + 1,  3 * samples + 1) / fs_MHz
        T0=1/f0_MHz
        signal = np.zeros_like(t)
        beta_term = np.pi * beta * t / T0
        mask = np.abs(1 - (2 * beta * t / T0) ** 2) != 0
        signal[mask] = (1 / T0) * np.sinc(t[mask] / T0) * np.cos(beta_term[mask]) / (1 - (2 * beta * t[mask] / T0) ** 2)
        signal[t==T0/2/beta] = np.pi/4/T0*np.sinc(1/2/beta)
        signal[t==-T0/2/beta] = np.pi/4/T0*np.sinc(1/2/beta)
        return signal

    def root_raised_cosine_pulse(self, fs_MHz, f0_MHz, beta, samples):
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

    @staticmethod
    def time_vector(fs_MHz, samples):
        return np.arange(1, samples + 1) / fs_MHz

    @property
    def n_pulse(self):
        return self._n_pulse
    @property
    def pulse(self):
        return self._pulse
    @property
    def Ts(self):
        return self._ts
    @property
    def Tsymb(self):
        return self._tsymb
    @property
    def n_fir(self):
        return self._n_fir
