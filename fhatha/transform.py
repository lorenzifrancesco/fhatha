import numpy as np
from scipy.fft import fft, ifft
from scipy.special import jv, j0
from scipy import optimize
from matplotlib import pyplot as plt
import logging as lg

lg.basicConfig(
    level=lg.WARN,
    format="%(asctime)s [%(levelname)s] %(message)s"  # basic format
)

"""
- r_max is taken in order to fix the normalization in the physical applications
"""
class FastAccurateHankel():
    def __init__(self, n_samples, fresnel, r_max=1):
        self.n_samples = n_samples
        self.fresnel = fresnel
        self.r_max = r_max
        grid, x0, alpha =  self.generate_grid()
        self.grid = grid * self.r_max
        self.r = self.grid  # walkaround
        self.x0 = x0 * self.r_max
        self.alpha = alpha
        self.js = self.prepare_js()
        self.k0 =  3/(8*self.alpha) + 1/2  # TODO, still the approximation
        self.kr = grid * 2 * np.pi * fresnel / self.r_max # besides the numerical grid, this is the actual k space in normalized units
        

    # core definitions: optimal grid and convolution kernel    
    def generate_grid(self) -> tuple:
        def fun(alpha): return alpha + np.log(1 - np.exp(-alpha))/(self.n_samples - 1)
        alpha = optimize.root_scalar(fun, bracket=[0, 10], method='bisect').root
        x0 = np.exp(-alpha*(self.n_samples))/2 * (1+np.exp(alpha)) # FIXME
        idx = np.arange((self.n_samples))
        log_xy = x0 * np.exp(idx * alpha)
        # assert np.isclose(log_xy[0], x0) # this is true
        # assert np.isclose(log_xy[-1], 1.0) #  this is not true
        return log_xy, x0, alpha
    
    def prepare_js(self):
        x_padded = self.x0 * \
        np.exp((np.arange((2 * self.n_samples)) - self.n_samples + 1) * self.alpha)
        return jv(1, 2 * np.pi * self.fresnel * x_padded)

    def correlate(self, fs, js):
        return fft(fft(fs) * ifft(js))
        
 
    # interface with the user       
    def sample(self, function):
        return function(self.grid)
    
    def pad_2x(self, fs):
        return np.pad(fs, (0, self.n_samples), mode='constant', constant_values=0)
        
    # this implements internal padding and unpadding
    def fht(self, fs):
        assert(len(fs)==self.n_samples)
        fs_rolled = np.roll(fs, -1)
        fs_rolled[-1] = 0.0
        assert fs_rolled[1] == fs[2] # TODO HALT FIXME
        phis = np.zeros(2*self.n_samples, dtype=complex)
        phis[:self.n_samples] = (fs - fs_rolled) * \
            np.exp((np.arange(self.n_samples) - self.n_samples + 1) * self.alpha)
        phis[0] *= self.k0
        print(self.fresnel)
        return self.correlate(phis, self.js)[:self.n_samples] / (self.fresnel * self.grid / self.r_max)  # FIXME what is the right scaling
        
    def ifht(self, fs):
        return self.fht(fs)
         