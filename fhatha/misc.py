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

def correlate(fs, js):
    return fft(fft(fs) * ifft(js))

# SIEGMAN original method (Eq. 10 in Magni)
def naive_siegman(fs: np.ndarray, alpha: float, x0: float, fresnel: float) -> np.ndarray:
    n_samples = len(fs)
    xs = x0 * np.exp(np.arange(n_samples) * alpha)
    y0 = x0 * fresnel
    ys = y0 * np.exp(np.arange(n_samples) * alpha)
    fs *= xs**2
    xy_padded = x0 * y0 * np.exp(np.arange((2 * n_samples)) * alpha)
    js = j0(2 * np.pi * xy_padded)
    fsp = np.pad(fs, (0, n_samples), mode='constant', constant_values=0)
    return xs, np.pad(ys, (0, n_samples), mode="constant", constant_values=1), 2*np.pi*alpha * correlate(fsp, js) # FIXME


# MAGNI original method
def optimal_dln(n_samples: int) -> float:
    def fun(alpha): return alpha + np.log(1 - np.exp(-alpha))/(n_samples - 1)
    alpha = optimize.root_scalar(fun, bracket=[0, 10], method='bisect').root
    # print("ALPHA:", alpha)
    return alpha


def magni_grid(n_samples: int) -> tuple:  # used for the sampling phase
    alpha = optimal_dln(n_samples)
    x0 = np.exp(-alpha*(n_samples))/2 * (1+np.exp(alpha))  # FIXME
    # x0 = 0.01
    # x0 *= 1.075 # FIXME
    idx = np.arange((n_samples))
    log_xy = x0 * np.exp(idx * alpha)
    # assert np.isclose(log_xy[0], x0) # this is true
    # assert np.isclose(log_xy[-1], 1.0) #  this is not true
    return log_xy, x0, alpha


def magni_xi(n_samples: int, extend=False) -> np.array:
    alpha = optimal_dln(n_samples)
    if extend:
        idx = np.arange(n_samples * 2)
        xis = np.zeros(n_samples*2 + 1)
    else:
        idx = np.arange(n_samples)
        xis = np.zeros(n_samples + 1)
    xis = np.exp((idx-n_samples) * alpha)
    xis[0] = 0.0
    return xis


def precalculate_js_magni(
        x0,
        n_samples,
        alpha,
        fresnel) -> np.ndarray:
    # * (1-1/n_samples)
    x_padded = x0 * \
        np.exp((np.arange((2 * n_samples)) - n_samples + 1) * alpha)
    # x_padded = (1+1/512) * x0 * np.exp((1+1/512) * (np.arange((2 * n_samples)) - n_samples)* alpha)
    return jv(1, 2 * np.pi * fresnel * x_padded)


def fhta_single(
        fs: np.ndarray,
        fresnel: float) -> np.ndarray:

    n_samples = len(fs)
    fsp = np.pad(fs, (0, n_samples), mode='constant', constant_values=0)
    # xs, x0, alpha = magni_grid(n_samples)

    alpha = optimal_dln(n_samples)
    x0 = np.exp(-alpha*(n_samples))/2 * (1+np.exp(alpha))  # FIXME
    idx = np.arange((n_samples))
    xs = x0 * np.exp(idx * alpha)
    print("alpha: ", alpha)
    xis = magni_xi(n_samples)
    plt.scatter(range(n_samples), xis, s=0.1,
                label="xi", marker='x', color='red')
    plt.scatter(range(n_samples-1), xis[1:],
                s=0.1, label="xi", marker='x', color="red")
    plt.scatter(range(n_samples), xs, s=0.1, label="xi", marker='x')
    plt.savefig("media/tests/xi_vs_x.pdf")
    plt.close()
    for i in range(n_samples-1):
        lg.debug(
            f"checking subinterval {i+1:>5}, : xi_n {xis[i]:.5f}, xi_n+1 {xis[i+1]:.5f} | avg= {(xis[i] + xis[i+1])/2:.5f} selected= {xs[i]:.5f}")
        pass
    ys = xs
    ysp = np.pad(ys, (0, n_samples), mode='constant', constant_values=1)
    k0 = 3/(8*alpha) + 1/2  # TODO, still the approximation
    print("k0", k0)
    fsp[n_samples] = 0.0
    fsp_rolled = np.roll(fsp, -1)
    fsp_rolled[-1] = 0.0
    assert fsp_rolled[1] == fsp[2]
    phis = (fsp - fsp_rolled) * \
        np.exp((np.arange(2*n_samples) - n_samples + 1) * alpha)
    phis[0] *= k0
    phis[n_samples:] *= 0.0

    js = precalculate_js_magni(x0, n_samples, alpha, fresnel)
    print("lenght of js", len(js))
    # remember x0 = y0 in Magni
    lg.warning(fsp)
    return xs, ys, correlate(phis, js) / ysp / fresnel

def foolproof_magni(
        fun,
        n_samples: int,
        fresnel: float) -> np.ndarray:

    # grid construction
    idx = np.arange(n_samples)
    alpha = optimal_dln(n_samples)
    x0 = np.exp(-alpha*(n_samples))/2 * (1+np.exp(alpha))
    print("x0", x0)
    grid = x0 * np.exp(alpha * idx)
    k0 = 3/(8*alpha) + 1/2  # TODO, still the approximation

    fs = np.zeros(n_samples + 1)
    fs[:n_samples] = fun(grid, transform=False)
    fs[n_samples] = 0.0
    phis = np.zeros(2*n_samples)
    for i in range(2*n_samples):
        if i < n_samples:
            phis[i] = (fs[i]-fs[i+1]) * np.exp(alpha*(i+1-n_samples))
        else:
            phis[i] = 0
    phis[0] *= k0

    js = np.zeros(2*n_samples)
    for i in range(2*n_samples):
        js[i] = jv(1, 2*np.pi*fresnel*x0*np.exp((i+1-n_samples)*alpha))
    convolution = fft(fft(phis) * ifft(js))
    lg.warning(fs)
    return 1, 1, convolution[:n_samples] / (fresnel * grid)