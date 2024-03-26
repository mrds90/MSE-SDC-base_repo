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


def RaisedCosinePulse(fs_MHz: float, samples: int) -> np.ndarray:
    t_us = TimeVector(fs_MHz, samples)

    a = 0.5
    # Ensure the denominator is never zero
    t_us[np.abs(t_us) < 1e-12] = 1e-12

    # Calculate the raised cosine pulse
    signal = np.cos(np.pi * t_us) / (1 - (2 * a * t_us) ** 2)
    signal = signal/np.max(np.abs(signal))

    return signal
