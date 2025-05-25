## @file demodulator.py
# @brief Module containing the Demodulator class for digital signal demodulation.
# @details This module implements a Demodulator class that handles signal reception and demodulation
# for a digital communication system. It performs matched filtering, detection, and symbol
# synchronization using a phase-locked loop (PLL) to recover transmitted data.
# @author Marcos Dominguez
# @author Lucas Monzon Languasco

import numpy as np
from scipy.signal import lfilter


## @class Demodulator
#  @brief Handles the demodulation process for a received signal.
#  @details This class performs matched filtering, detection, and symbol synchronization using a phase-locked loop (PLL) to extract transmitted bytes from a received waveform.
class Demodulator:
    ## @brief Constructor for the Demodulator class.
    #  @param pulse The pulse shape used in transmission.
    #  @param n_bytes Number of bytes expected in the payload.
    #  @param Tsymb Symbol period.
    #  @param Ts Sampling period.
    #  @param n_sfd Number of symbols in the start frame delimiter (SFD).
    #  @param n_pre Number of symbols in the preamble.
    #  @param det_th Detection threshold for energy detection.
    #  @param pll Dictionary of PLL parameters: kp (proportional gain), ki (integral gain), and delay.
    def __init__(self, pulse, n_bytes:int, Tsymb, Ts, n_sfd, n_pre, det_th:float=0.25, pll={'kp': 0.1, 'ki': 0.01, 'delay': 0}):
        self._pulse = pulse
        self._Tsymb = Tsymb
        self._Ts = Ts
        self._n_pulse = int(Tsymb / Ts)
        self._n_sfd = n_sfd
        self._n_pre = n_pre
        self._det_th = det_th
        self._pll_params = pll
        self._last_dis = None
        self._n_bytes = n_bytes

    ## @brief Getter for the last demodulation diagnostic information.
    @property
    def last_dis(self):
        return self._last_dis

    ## @brief Getter for PLL parameters.
    @property
    def pll_params(self):
        return self._pll_params

    ## @brief Setter for PLL parameters with validation.
    #  @param value Dictionary with 'kp', 'ki', and 'delay' keys.
    #  @throws ValueError If the input is not a valid dictionary or contains invalid values.
    @pll_params.setter
    def pll_params(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("The 'pll' argument must be a dictionary.")
        if 'kp' not in value or 'ki' not in value or 'delay' not in value:
            raise ValueError("The 'pll' dictionary must contain 'kp', 'ki', and 'delay' keys.")
        if not isinstance(value['kp'], (int, float)):
            raise ValueError("The 'kp' value must be a numeric value.")
        if not isinstance(value['ki'], (int, float)):
            raise ValueError("The 'ki' value must be a numeric value.")
        if not isinstance(value['delay'], int):
            raise ValueError("The 'delay' value must be an integer.")
        self._pll_params = value

    ## @brief Phase-Locked Loop for symbol timing recovery.
    #  @param input_signal Input signal to synchronize.
    #  @param f0 Nominal frequency of the clock (1 / Tsymb).
    #  @param fs Sampling frequency (1 / Ts).
    #  @param kp Proportional gain of PLL.
    #  @param ki Integral gain of PLL.
    #  @param delay Delay in feedback loop.
    #  @return Tuple of VCO signal and a dictionary of PLL internals ('phd', 'err', 'phi_hat').
    def pll(self, input_signal, f0, fs, kp, ki, delay=0):
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

    ## @brief Performs full demodulation of the received signal.
    #  @param y The received signal.
    #  @return List of decoded bytes.
    def demodulate(self, y):
        y_mf = lfilter(self._pulse, [1], y)
        y_mf /= np.sum(self._pulse**2)

        y_mf_sq = y_mf**2
        n_ma = self._n_pulse
        y_mf_sq_ma = lfilter(np.ones(int(n_ma)) / n_ma, [1], y_mf_sq)

        # Band-pass filtering pipeline
        prefilter_data = [[0.01925927815873231,0,-0.01925927815873231],[1,-1.924163036247956,0.9614814515953285]]
        filter_data = [[0.004884799809161248,0,-0.004884799809161248],[1,-1.838755285386028,0.9902304008963749]]
        prefilter_b, prefilter_a = prefilter_data[0], prefilter_data[1]
        filter_b, filter_a = filter_data[0], filter_data[1]

        y_mf_pf = lfilter(prefilter_b, prefilter_a, y_mf)
        y_mf_pf_sq = y_mf_pf**2
        y_mf_pf_sq_bpf = lfilter(filter_b, filter_a, y_mf_pf_sq)

        f0 = 1.0 / self._Tsymb
        fs = 1.0 / self._Ts
        vco, pllis = self.pll(y_mf_pf_sq_bpf, f0, fs, self._pll_params['kp'], self._pll_params['ki'], self._pll_params['delay'])
        pll_cos, pll_sin = np.real(vco), np.imag(vco)
        pll_clk_i, pll_clk_q = pll_cos >= 0, pll_sin >= 0

        detection = np.array(y_mf_sq_ma) >= self._det_th
        flank_in = pll_clk_i & np.concatenate(([0], ~pll_clk_i[:-1])).astype(bool)
        en_sample = flank_in & detection

        hat_xn = np.array(y_mf[en_sample == 1])
        hat_packet = hat_xn > 0

        # Start frame delimiter detection
        sfd = np.zeros(self._n_sfd, dtype=int)
        if self._n_pre % 2 == 0:
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
                packet = hat_packet[i : i + 8 * self._n_bytes]
                hat_bytes.extend(np.packbits(packet, bitorder='big'))
                i += 8 * self._n_bytes
            else:
                i += 1

        # Save diagnostics
        self._last_dis = {
            'y_mf': y_mf,
            'y_mf_pf': y_mf_pf,
            'y_mf_pf_sq': y_mf_pf_sq,
            'y_mf_pf_sq_bpf': y_mf_pf_sq_bpf,
            'y_mf_sq_ma': y_mf_sq_ma,
            'vco': vco,
            'pll': pllis,
            'hat_packet': hat_packet,
            'en_sample': en_sample,
        }
        return hat_bytes

    ## @brief Returns the last stored demodulation diagnostics.
    @property
    def dis(self):
        return self._last_dis
