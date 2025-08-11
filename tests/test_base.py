from fhatha import transform as tr
import numpy as np
from scipy.special import jv, j0
import scipy.fft as sp
from matplotlib import pyplot as plt


def magni(x, transform=True):
    if transform:
        # beware that here we use eta. It depends on the fresnel number
        return np.sqrt(10 * np.pi) * x**(-4) * (2 * x**2 * j0(x) + (x**3-4*x) * jv(1, x))
    else:
        return np.sqrt(5/(2 * np.pi)) * x**2

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
    
def test_siegman(function=flat_top, label="ft", fresnel=1):
    n_samples = 256
    idx =  np.arange((n_samples))
    idxp = np.arange((n_samples*2))
    # radius = np.linspace(0, 1, n_samples)
    r0  = 0.06349
    dln = 0.01612
    
    # r0  = 0.063
    # dln = 0.01612
    log_r   = r0 * np.exp(idx * dln)
    
    # fftlog
    # log_k   = 1 / r0 * np.exp((idx - n_samples + 1) * dln) 
    # log_k_p = 1 / r0 * np.exp((idxp - n_samples + 1) * dln)
    
    # magni
    log_r   = 1 * np.exp((idx - n_samples + 1) * dln)
    log_k   = 1 * np.exp((idx - n_samples + 1) * dln)
    log_k_p = 1 * np.exp((idxp - (n_samples - 1)/2) * dln)
    print(log_k)
    # siegman
    # log_k   =  r0 * np.exp((idx)  * dln)
    # log_k_p =  r0 * np.exp((idxp) * dln)
    
    offset_lin = log_r[0] * log_k[-1] * 2 * np.pi * fresnel # accomodate the definition of Magni
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
    _, _, g_siegman   = tr.naive_siegman(fpy[:n_samples], dln, r0, k0)
    
    plt.figure(figsize=(4, 2.5))
    plt.plot(idxp, fpy, label='Original Data')
    plt.xlabel('idx')
    plt.tight_layout()
    plt.savefig('media/'+label+'initial.png')
    plt.clf()

    # physics
    exact = function(log_k_p, transform=True)
    plt.figure(figsize=(4, 2.5))
    plt.plot(log_k_p, g / (log_k_p * fresnel), label='Transformed Data')
    plt.plot(log_k_p, exact, label='Transformed Data')
    plt.xlabel(r'$k$')
    plt.xlim(0, n_samples/4)
    plt.ylim(-0.5, 1.6)
    plt.tight_layout()
    name = 'media/'+label+'_physical_transform.pdf'
    plt.savefig(name)
    print("Saved to ", name)
    
    # samples
    exact = function(log_k_p, transform=True) * (log_k_p * fresnel)
    plt.figure(figsize=(4, 2.5))
    plt.plot(idxp, g, label="scipy")
    plt.plot(idxp, g_siegman, label="siegman")
    plt.plot(idxp, exact, label="exact")
    plt.legend()
    plt.xlabel('idx')
    plt.axvline(n_samples, color='red', linestyle='--', label='n_samples')
    # plt.axvline(np.log(1/log_k_p[-1]) * n_samples, color='red', linestyle='--', label='n_samples')
    # plt.xlim(0, 100)
    plt.tight_layout()
    name = 'media/'+label+'_transformed_samples.pdf'
    plt.savefig(name)
    print("Saved to ", name)
    

def test_naive_siegman():
    n_samples = 256
    idx = np.arange((n_samples))
    idxp = np.arange((2*n_samples))
    r0  = 0.06349
    dln = 0.01612
    
    # desired_r_max = 1.0
    # dln = np.log(desired_r_max / r0) / (n_samples - 1)
    
    print("dln:", dln)
    log_r   = r0 * np.exp(idx * dln)
    log_r_magni = tr.magni_grid(n_samples, dln, r0)
    
    print("selected range:", log_r[:4], "...", log_r[-4:])
    print("last value:", log_r[-1])
    # assert np.isclose(log_r[-1], 3.938)
    log_k   = 1 / r0 * np.exp((idx - n_samples + 1) * dln)
    
    f = magni(log_r, transform=False)
    f_magni = magni(log_r_magni, transform=False)
    fs = log_r * f
    fs_m = log_r_magni * f
    # fs = np.pad(fs, (0, n_samples), 'constant', constant_values=0) -> we are already padding
    _, _, g = tr.naive_siegman(fs, dln, r0, r0)
    _, _, g_m = tr.fhta_single(fs_m, dln, r0, fresnel=3.1415)
    
    # assert np.isclose(g, 0.0)
    plt.figure(figsize=(4, 2.5))
    plt.plot(idxp, np.real(g),   lw=0.6, ls="-")
    plt.plot(idxp, np.imag(g),   lw=0.6, ls="-")
    plt.plot(idxp, np.real(g_m), lw=1.2, ls=":")
    plt.plot(idxp, np.imag(g_m), lw=1.2, ls=":")
    plt.xlabel('idx')
    plt.tight_layout()
    plt.savefig('media/naive.pdf')


def test_naive_magni(fresnel = 1, clip_reference = 1):
    n_samples = 400
    idx = np.arange((n_samples))
    idxp = np.arange((2*n_samples))
    r0  = 0.006349
    dln = 0.01612
    
    # desired_r_max = 1.0
    # dln = np.log(desired_r_max / r0) / (n_samples - 1)
    
    print("dln:", dln)
    log_r   = r0 * np.exp(idx * dln)
    log_r_magni = tr.magni_grid(n_samples, dln, r0)
    
    print("selected range:", log_r[:4], "...", log_r[-4:])
    print("last value:", log_r[-1])
    # assert np.isclose(log_r[-1], 3.938)
    log_k   = 1 / r0 * np.exp((idx - n_samples + 1) * dln)
    log_k_p   = 1 / r0 * np.exp((idxp - n_samples + 1) * dln)
    
    f       = magni(log_r, transform=False)
    f_magni = magni(log_r_magni, transform=False)
    fs = log_r * f
    fs_m = log_r_magni * f_magni
    # fs = np.pad(fs, (0, n_samples), 'constant', constant_values=0) -> we are already padding  
    equivalent_k0 = r0 * fresnel / (2*np.pi) # for the classical Siegman method
    _, _, g   = tr.naive_siegman(fs, dln, r0, equivalent_k0)
    _, _, g_m = tr.fhta_single(fs_m, dln, r0, fresnel=fresnel)
    g_m /= 2 * np.pi #  FIXME this trial is probably wrong
    g /= log_k_p # this seems to be needed. Still the oscillations seem to be wrong. 
    
    # We notice (check.pdf)
    # - the magni transform is out of scale, but behaves good (about 11 over 1.3 desired)
    # - the Siegman transform is very much in scale, but the equivalent k0 I think we are missing it
    # - every transform starts from the same point at null k 
    # assert np.isclose(g, 0.0)
    
    exact = magni(2* np.pi * fresnel * log_r_magni, transform=True) # adapt for the calculation with the actual eta
    siegman = g
    fig, ax1 = plt.subplots(figsize=(5, 4))
    color1 = 'tab:blue'
    color1b = 'orange'
    for reference in [exact, siegman][:clip_reference]:
        ax1.plot(log_r_magni, np.real(reference)[:n_samples], lw=1.2, ls="-")
        # ax1.plot(log_r_magni, np.imag(reference)[:n_samples], lw=1.2, ls=":")
   
    ax1.set_ylabel("reference", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    # Shared x-axis
    ax1.set_xlabel(r'$k$')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(3, 0))

    # T_max_high = 
    # Second axis: T_max
    ax2 = ax1.twinx()
    color2 = 'tab:red'

    ax2.plot(log_r_magni, np.real(g_m)[:n_samples], lw=1.2, ls="-", color=color2)
    ax2.set_ylabel("magni", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    # ax2.plot(log_r_magni, np.imag(g_m)[:n_samples], lw=1.2, ls=":")
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(3, 0))
    plt.xlabel('idx')
    plt.tight_layout()
    name = 'media/naive.pdf'
    plt.savefig(name)
    print(f"Saved plot to {name}")
  
    
if __name__ == "__main__":
    test_siegman(function=flat_top, label="ft", fresnel = 1)
    exit()
    test_naive_magni(clip_reference=2)
    test_siegman(function=flat_top, label="ft", fresnel = 1)
    test_siegman(function=magni,    label="mg", fresnel = 3)