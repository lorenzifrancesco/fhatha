from fhatha import transform as tr
import numpy as np
from scipy.special import jv, j0
import scipy.fft as sp
from matplotlib import pyplot as plt


def magni(x, transform=True):
    if transform:
        return np.sqrt(10 * np.pi) * x**(-4) * (2 * x**2 * j0(x) + (x**3-4*x) * jv(1, x))
    else:
        return np.sqrt(5/2 * np.pi) * x**2

def flat_top(x, transform=True):
    if transform:
        # beware, we are using jv1, we should use jv0
        b = 3.938
        return b * jv(1, 2*np.pi* 3.938 * x) / (x) # only valid for the parameters alpha, beta, b, r0, rho0 in siegman
    else:
        return x/x

def test_basic():
    radius = np.linspace(0, 20, 500)
    data = jv(0, radius)
    assert (tr.fht(data) == 2 *data / np.sum(data)).all()
    
    
def test_scipy_compatibility():
    radius = np.linspace(0, 20, 500)
    data = np.random(0, radius)
    assert(False)
    # FIXME: the scipy version needs a bias
    assert np.allclose(tr.fht(data), fht(data))
    assert np.allclose(tr.ifht(fht(data)), data)
    
def test_siegman(function=flat_top, label="ft"):
    n_samples = 256
    idx =  np.array(range(n_samples))
    idxp = np.array(range(n_samples*2))
    # radius = np.linspace(0, 1, n_samples)
    r0  = 0.063
    dln = 0.01612
    log_r   = r0 * np.exp(idx * dln)
    # log_k   = 1 / r0 * np.exp((idx - n_samples + 1) * dln)
    # log_k_p = 1 / r0 * np.exp((idxp - n_samples + 1) * dln) # balanced
    # change here the rescaling since then it gets inside the offset
    log_k   =  r0 * np.exp((idx)  * dln) # siegman
    log_k_p =  r0 * np.exp((idxp) * dln) # siegman
    offset_lin = log_r[0] * log_k_p[-1]  * 2 * np.pi
    print(f"Offset linear: {offset_lin}")
    
    k0 = 1 / r0 * np.exp((-n_samples+1) * dln)
    r_end = r0 * np.exp((n_samples - 1) * dln)
    assert(np.isclose(k0 * r_end, 1.0))
    f = function(log_r, transform=False)
    
    # prepare the scipy f function from the Siegman (physical)
    fpy = log_r * f
    fpy = np.pad(fpy, (0, n_samples), 'constant', constant_values=0)
    gpy = sp.fht(fpy, 
               dln=dln, 
               mu=0, 
               offset=np.log(offset_lin), 
               bias = 0.0)
    # get the Siegman transformed function from scipy
    g = gpy # note that in the plots of Siegman we are plotting k g(k).
    
    plt.figure(figsize=(4, 2.5))
    plt.plot(idxp, fpy, label='Original Data')
    plt.xlabel('idx')
    plt.tight_layout()
    plt.savefig('media/'+label+'initial.png')
    plt.clf()

    exact = function(log_k_p, transform=True)
    plt.figure(figsize=(4, 2.5))
    plt.plot(log_k_p * 2 * np.pi, (g/log_k_p) / np.max(g/log_k_p) * 1.5, label='Transformed Data')
    plt.plot(log_k_p, exact, label='Transformed Data')
    plt.xlabel(r'$k$')
    plt.xlim(0, n_samples/4)
    plt.tight_layout()
    plt.savefig('media/'+label+'_physical_transform.png')
    
    exact = function(log_k_p, transform=True) * log_k_p
    plt.figure(figsize=(4, 2.5))
    plt.plot(idxp, g, label='Transformed Data')
    plt.plot(idxp, exact, label='Transformed Data')
    plt.xlabel('idx')
    plt.axvline(n_samples, color='red', linestyle='--', label='n_samples')
    # plt.axvline(np.log(1/log_k_p[-1]) * n_samples, color='red', linestyle='--', label='n_samples')
    # plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig('media/'+label+'_transformed_samples.png')
    
if __name__ == "__main__":
    test_siegman(function=flat_top, label="ft")
    test_siegman(function=magni,    label="mg")
    