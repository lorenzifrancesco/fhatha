import numpy as np
from scipy.fft import fft, ifft
from scipy.special import jv, j0



def correlate(fs, js):
    return fft(fft(fs) * ifft(js))


# SIEGMAN original method
def naive_siegman(fs: np.ndarray, alpha: float, x0: float, y0: float) -> np.ndarray:
    n_samples = len(fs)
    fsp = np.pad(fs, (0, n_samples), mode='constant', constant_values=0)
    xs = np.exp(np.arange(n_samples) * alpha) * x0
    ys = np.exp(np.arange(n_samples) * alpha) * y0
    js = precalculate_js(x0*y0, n_samples, alpha)
    return xs, ys, correlate(fsp, js)

# alpha = dln
def precalculate_js(xy0, n_samples, alpha) -> np.ndarray:
    xy_padded = xy0 * np.exp(np.arange((2 * n_samples)) * alpha)
    return 2* np.pi * xy_padded * alpha * j0(2*np.pi * xy_padded)




# MAGNI original method
def magni_grid(n_samples: int, alpha: float, x0: float) -> tuple:
    idx = np.arange((n_samples))
    log_xy   = x0 * np.exp(idx * alpha)
    log_xy[0] = 0.0
    return log_xy

def fhta_single(
    fs: np.ndarray, 
    alpha: float, 
    x0: float, 
    fresnel:float) -> np.ndarray:
    
    n_samples = len(fs)
    fsp = np.pad(fs, (0, n_samples), mode='constant', constant_values=0)
    xs = magni_grid(n_samples, alpha, x0)
    ys = xs
    ysp = magni_grid(2*n_samples, alpha, x0)
    k0 = 3/(8*alpha) + 1/2 # FIXME, still the approximation
    
    phis = (fsp - np.roll(fsp, -1)) * np.exp((np.arange(2*n_samples) - n_samples + 1) * alpha)
    phis[0] *= k0
    phis[n_samples:] *= 0.0
    
    js = precalculate_js_magni(x0, n_samples, alpha, fresnel)
    print("lenght of js", len(js))
    # remember x0 = y0 in Magni
    return xs, ys, correlate(phis, js) / ysp / fresnel
    
def precalculate_js_magni(
    x0, 
    n_samples, 
    alpha, 
    fresnel) -> np.ndarray:
    x_padded = x0 * np.exp((np.arange((2 * n_samples)) + 1 - n_samples)* alpha)
    print("length of x_padded", len(x_padded))
    return jv(1, 2 * np.pi * fresnel * x_padded)