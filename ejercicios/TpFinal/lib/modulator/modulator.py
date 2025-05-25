## @file modulator.py
# @brief Module containing the Modulator class for digital signal modulation.
# @details This module implements a Modulator class that handles packet creation and modulation
# for a digital communication system. It uses different pulse shapes for modulation and includes
# functionality for adding preamble and start frame delimiter (SFD) sequences.
# @author Marcos Dominguez
# @author Lucas Monzon Languasco

import numpy as np
from lib.pulse_generator.pulse_generator import PulseGenerator
## @class Modulator
#  @brief Handles the modulation process for a byte sequence.
#  @details Uses a specified pulse shape to modulate binary sequences derived from byte arrays.
#           Adds preamble and start frame delimiter (SFD) sequences before the data bits.
#           Produces a baseband modulated signal using pulse shaping.
class Modulator:
    """
    @brief Handles the modulation process for a byte sequence.
    """

    ## @brief Constructor for Modulator
    #  @param n_bytes Number of bytes in the payload
    #  @param n_pre Number of bits in the preamble
    #  @param n_sfd Number of bits in the start frame delimiter (SFD)
    #  @param pulse Name of the pulse shape to use (passed to PulseGenerator)
    #  @param Ts Sampling interval
    #  @param Tsymb Symbol duration
    def __init__(self, n_bytes, n_pre, n_sfd, pulse:str, Ts, Tsymb):
        self._n_bytes = n_bytes
        self._n_pre = n_pre
        self._n_sfd = n_sfd
        self._pulse = PulseGenerator(pulse, Ts, Tsymb)

    ## @brief Creates a packet with preamble, SFD, and data bits
    #  @param byte_seq Sequence of bytes to encode
    #  @return A tuple (x, data) where x is the bipolar sequence and data is the binary array of payload bits
    def create_packet(self, byte_seq):
        pre = np.zeros(self._n_pre, dtype=int)
        pre[1::2] = 1
        sfd = np.zeros(self._n_sfd, dtype=int)
        if self._n_pre % 2 == 0:
            sfd[0::2] = 1
        else:
            sfd[1::2] = 1
        data = np.hstack([np.array(list(np.binary_repr(byte, 8)), dtype=int) for byte in byte_seq])
        packet = np.hstack([pre, sfd, data])
        x = 2 * packet - 1
        return x, data

    ## @brief Modulates a sequence of bytes into a baseband waveform
    #  @param byte_seq Sequence of bytes to modulate (values must be in [0, 255])
    #  @return The modulated signal (convolved with pulse shape)
    def modulate(self, byte_seq):
        assert np.all(np.array(byte_seq) <= 255), "All values in byte_seq must be between 0 and 255"
        x, self._bits = self.create_packet(byte_seq)
        xx = np.zeros(len(x) * self._pulse.n_pulse)
        xx[::self._pulse.n_pulse] = x
        xxx = np.convolve(xx, self._pulse.pulse)
        pad_size = (len(xxx) - len(xx)) // 2
        self._d = np.pad(xx, (pad_size, len(xxx) - len(xx) - pad_size), mode='constant')
        return xxx

    ## @brief Returns number of bytes in the payload
    @property
    def n_bytes(self):
        return self._n_bytes

    ## @brief Returns number of bits in the preamble
    @property
    def n_pre(self):
        return self._n_pre

    ## @brief Returns number of bits in the start frame delimiter (SFD)
    @property
    def n_sfd(self):
        return self._n_sfd

    ## @brief Returns number of samples per symbol (pulse length)
    @property
    def n_pulse(self):
        return self._pulse.n_pulse

    ## @brief Returns the pulse shape used for modulation
    @property
    def pulse(self):
        return self._pulse.pulse

    ## @brief Returns the sampling interval
    @property
    def Ts(self):
        return self._pulse.Ts

    ## @brief Returns the symbol duration
    @property
    def Tsymb(self):
        return self._pulse.Tsymb

    ## @brief Returns the length of the FIR pulse filter
    @property
    def n_fir(self):
        return self._pulse._n_fir

    ## @brief Returns the binary payload bits
    @property
    def bits(self):
        return self._bits

    ## @brief Returns the zero-padded version of the bipolar signal before convolution
    @property
    def d(self):
        return self._d
