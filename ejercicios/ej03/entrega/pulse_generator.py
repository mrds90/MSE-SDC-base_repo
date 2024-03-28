import numpy as np


def TimeVector(fs_MHz: float, samples: int) -> np.ndarray:
    t_us = np.arange(0, samples) / fs_MHz
    return t_us


def SquarePulse(fs_MHz: float, f0_MHz: float, samples: int) -> np.ndarray:
    t_us = TimeVector(fs_MHz, samples)
    signal = np.sign(np.sin(2 * np.pi * f0_MHz * t_us))
    signal[signal < 0] = 0
    

    return signal


def TriangularPulse(fs_MHz: float, f0_MHz: float, samples: int) -> np.ndarray:
    t_us = TimeVector(fs_MHz, samples)
    signal = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f0_MHz * t_us))
    return signal


def SinePulse(fs_MHz: float, f0_MHz: float, samples: int) -> np.ndarray:
    t_us = TimeVector(fs_MHz, samples)
    signal = np.sin(2 * np.pi * (f0_MHz / 2) * t_us)
    return signal


def RaisedCosinePulse(fs_MHz: float, f0_MHz:float, beta: float, samples: int) -> np.ndarray:

    
    t = np.arange(-samples, samples) / fs_MHz
    T0=1/f0_MHz
    
    signal = 1/T0*np.sinc(t/T0)*np.cos(np.pi*beta*t/T0)/(1-(2*beta*t/T0)**2)
    signal[t==T0/2/beta] = np.pi/4/T0*np.sinc(1/2/beta)
    signal[t==-T0/2/beta] = np.pi/4/T0*np.sinc(1/2/beta)
    

    return signal
