import numpy as np
from scipy.fft import fft, ifft


def precompute_kernel():
    return

def fht(data: np.ndarray) -> np.ndarray:
    
    result = fft(ifft() * fft())
    return result 