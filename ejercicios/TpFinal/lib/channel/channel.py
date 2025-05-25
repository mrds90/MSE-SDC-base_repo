## @file channel.py
#  @brief This module models the transmission channel, including filtering and additive noise.
#  @details Provides a Channel class that can simulate an ideal delta channel or a low-pass FIR channel.
#           It also adds Additive White Gaussian Noise (AWGN) based on a specified Signal-to-Noise Ratio (SNR).
#           Useful for simulating signal propagation in digital communication systems.
#  @author Marcos Dominguez
#  @author Lucas Monzon Languasco

import numpy as np
from scipy.signal import firwin

## @class Channel
#  @brief Simulates a transmission channel with optional FIR filtering and AWGN.
#  @details Supports two types of channel models: 'delta' (ideal) and 'fir' (low-pass filter).
#           Adds AWGN to the transmitted signal based on a configurable SNR.
class Channel:
    ## @brief Constructs a Channel instance.
    #  @param channel_type Type of channel: 'delta' or 'fir'.
    #  @param fs_MHz Sampling frequency in MHz.
    #  @param f0_MHz Carrier or cutoff frequency in MHz.
    #  @param phase Phase shift in degrees for 'delta' channel. Default is 0.
    #  @param snr_dB Signal-to-noise ratio in dB. Default is None (no noise).
    #  @param kwargs Additional parameters:
    #         - 'samples': number of samples for delta channel.
    #         - 'order': FIR filter order for 'fir' channel.
    #  @raises ValueError If required parameters are missing or invalid.
    def __init__(self, channel_type, fs_MHz, f0_MHz, phase=0, snr_dB: float | None = None, **kwargs):
        if channel_type == 'delta':
            samples = kwargs.get('samples')
            if samples is None:
                raise ValueError("Missing 'samples' for delta channel.")
            self._channel = self._delta_pulse(fs_MHz, f0_MHz, samples, phase)
        elif channel_type == 'fir':
            order = kwargs.get('order')
            if order is None:
                raise ValueError("Missing 'order' for fir channel.")
            self._channel = self._low_pass_fir(fs_MHz, f0_MHz, order)
        else:
            raise ValueError("Invalid channel type. Use 'delta' or 'fir'.")
        self._snr_dB = snr_dB

    ## @brief Adds Additive White Gaussian Noise (AWGN) to the input signal.
    #  @param signal Input signal as a NumPy array.
    #  @return Noisy signal as a NumPy array.
    #  @raises ValueError If input is not a NumPy array.
    def add_awgn(self, signal: np.ndarray) -> np.ndarray:
        if not isinstance(signal, np.ndarray):
            raise ValueError("The signal must be a numpy array.")
        if self._snr_dB is None:
            return signal

        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(self._snr_dB / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)
        noise = noise_std * np.random.randn(len(signal))
        return signal + noise

    ## @brief Applies the channel effect (filtering) and adds AWGN to the input signal.
    #  @param signal Input signal as a NumPy array.
    #  @return Output signal after convolution with the channel and noise addition.
    #  @raises ValueError If input is not a NumPy array.
    def transmit(self, signal: np.ndarray) -> np.ndarray:
        if not isinstance(signal, np.ndarray):
            raise ValueError("The signal must be a numpy array.")
        y = np.convolve(signal, self._channel)
        y = self.add_awgn(y)
        return y

    ## @brief Generates a delta pulse with optional phase shift.
    #  @param fs_MHz Sampling frequency in MHz.
    #  @param f0_MHz Frequency used to calculate samples per cycle.
    #  @param samples Number of samples in the output pulse.
    #  @param phase Phase shift in degrees (0 to <360).
    #  @return Delta pulse as a NumPy array.
    #  @raises ValueError If phase is outside [0, 360) or pulse index exceeds array bounds.
    def _delta_pulse(self, fs_MHz: float, f0_MHz: float, samples: int, phase: float) -> np.ndarray:
        if not (0 <= phase < 360):
            raise ValueError("The 'phase' argument must be in the range 0 to 360 degrees.")
        T = 1 / f0_MHz
        samples_per_cycle = T * fs_MHz
        phase_index = int(round(phase / 360 * samples_per_cycle))
        delta_pulse = np.zeros(samples)
        if phase_index < samples:
            delta_pulse[phase_index] = 1
        return delta_pulse

    ## @brief Designs a low-pass FIR filter using windowed-sinc method.
    #  @param fs_MHz Sampling frequency in MHz.
    #  @param f0_MHz Cutoff frequency in MHz.
    #  @param order Filter order.
    #  @return FIR filter coefficients as a NumPy array.
    def _low_pass_fir(self, fs_MHz, f0_MHz, order):
        nyquist = 0.5 * fs_MHz
        norm_cutoff = f0_MHz / nyquist
        fir_coeffs = firwin(order, norm_cutoff)
        return fir_coeffs

    ## @brief Getter for the signal-to-noise ratio (SNR) in dB.
    #  @return SNR in decibels.
    @property
    def snr_dB(self):
        return self._snr_dB

    ## @brief Setter for the signal-to-noise ratio (SNR) in dB.
    #  @param value SNR value to set, must be numeric.
    #  @raises ValueError If value is not a number.
    @snr_dB.setter
    def snr_dB(self, value):
        if not isinstance(value, (int, float, type(None))):
            raise ValueError("The 'snr_dB' argument must be a numeric value or None.")
        self._snr_dB = value
