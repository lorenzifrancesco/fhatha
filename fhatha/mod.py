import numpy as np
from scipy.fft import fft, ifft
from scipy.special import jv, j0
from scipy import optimize
from matplotlib import pyplot as plt
import logging as lg

# One-time setup (do this near program start)
lg.basicConfig(
    level=lg.WARN,
    format="%(asctime)s [%(levelname)s] %(message)s"  # basic format
)

class FastAccurateHankel():
    def __init__(self, n_samples, fresnel):
        self.n_samples = n_samples
        self.fresnel = fresnel
        grid, x0, alpha =  self.generate_grid()
        self.grid = grid
        self.x0 = x0
        self.alpha = alpha
        self.js = self.prepare_js()
        self.k0 =  3/(8*self.alpha) + 1/2  # TODO, still the approximation

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
        
    def fht(self, fsp):
        fsp[self.n_samples] = 0.0
        fsp_rolled = np.roll(fsp, -1)
        fsp_rolled[-1] = 0.0
        assert fsp_rolled[1] == fsp[2]
        phis = (fsp - fsp_rolled) * \
            np.exp((np.arange(2*self.n_samples) - self.n_samples + 1) * self.alpha)
        phis[0] *= self.k0
        phis[self.n_samples:] *= 0.0
        return self.correlate(phis, self.js)[:self.n_samples] / (self.fresnel * self.grid)
        
    def ifht(self, fs):
        return self.fht(fs)
         