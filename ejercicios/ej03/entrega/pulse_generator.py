import numpy as np


def TimeVector(fs_MHz: float, samples: int) -> np.ndarray:
    t_us = np.arange(0, samples) / fs_MHz
    return t_us


def SquarePulse(fs_MHz: float, f0_MHz: float, samples: int) -> np.ndarray:
    t_us = TimeVector(fs_MHz, samples)
    signal = np.sign(np.sin(2 * np.pi * f0_MHz * t_us))

    return signal


def TriangularPulse(fs_MHz: float, f0_MHz: float, samples: int) -> np.ndarray:
    t_us = TimeVector(fs_MHz, samples)
    signal = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f0_MHz * t_us))
    return signal


def SinePulse(fs_MHz: float, f0_MHz: float, samples: int) -> np.ndarray:
    t_us = TimeVector(fs_MHz, samples)
    signal = np.sin(2 * np.pi * (f0_MHz / 2) * t_us)
    return signal


def RaisedCosinePulse(fs_MHz: float, beta: float, samples: int) -> np.ndarray:

    Ts = 1/fs_MHz
    t_us = TimeVector(fs_MHz, samples)

    signal = (1 / Ts) * np.sinc(t_us / Ts) * np.cos(np.pi * beta * t_us / Ts) / (1 - (2 * beta * t_us / Ts) ** 2)
    signal[np.isclose(t_us, Ts / (2 * beta))] = np.pi / (4 * Ts) * np.sinc(1 / (2 * beta))
    signal[np.isclose(t_us, -Ts / (2 * beta))] = np.pi / (4 * Ts) * np.sinc(1 / (2 * beta))
    signal = signal/np.max(np.abs(signal))

    return signal
