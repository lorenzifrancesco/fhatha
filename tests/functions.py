from fhatha import misc as tr
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
        b = 1.0 # compatible with the specification of Magni
        return b * jv(1, 2*np.pi* b * x) / (x) # only valid for the parameters alpha, beta, b, r0, rho0 in siegman
    else:
        if x<=1:
            return x/x
        else:
            return 0
        
        
## old functions
def test_siegman(function=flat_top, label="ft", fresnel=1):
    n_samples = 256
    idx =  np.arange((n_samples))
    idxp = np.arange((n_samples*2))

    # # radius = np.linspace(0, 1, n_samples)
    # r0  = 0.06349
    # r_f = 1.0
    # dln = np.log(r_f/r0) / (n_samples - 1)
    
    # # r0  = 0.063
    # # dln = 0.01612
    # log_r   = r0 * np.exp(idx * dln)
    # assert np.isclose(log_r[-1], r_f)
    
    # # fftlog
    # # log_k   = 1 / r0 * np.exp((idx - n_samples + 1) * dln) 
    # # log_k_p = 1 / r0 * np.exp((idxp - n_samples + 1) * dln)
    
    # # magni
    # log_r   = 1 * np.exp((idx - n_samples + 1) * dln)
    # log_k   = 1 * np.exp((idx - n_samples + 1) * dln)
    # log_k_p = 1 * np.exp((idxp - (n_samples - 1)/2) * dln)
    # r0 = log_r[0]
    # assert np.isclose(log_r[-1], r_f)
    # assert np.isclose(log_k[-1], 1.0)
    # print(log_k)

    
    # REDEFINITIONS according to the Magni scheme
    r0 = 0.01
    dln = np.log(1/r0) / (n_samples - 1)
    # r0 = (1+np.exp(dln)) * np.exp(-dln / n_samples)/2
    log_r = r0 * np.exp(idx * dln) # enforce the maximum to 1
    log_r_p = r0 * np.exp((idxp - n_samples + 1) * dln) # this is the first value of the log_r, by construction
    assert np.isclose(log_r[-1], 1.0)
    log_k = log_r / r0
    log_k_p = log_r_p / (r0)
    k0 = log_k[0]
    # print(log_r, log_k)
    assert np.isclose(log_r[-1], 1/log_k[0])
    assert np.isclose(log_r[0], 1/log_k[-1])
    # siegman
    # log_k   =  r0 * np.exp((idx)  * dln)
    # log_k_p =  r0 * np.exp((idxp) * dln)
    
    # make Magni sample in the correct space again
   
    offset_lin = log_r[0] * log_k[-1] * 2 * np.pi # * fresnel # accomodate the definition of Magni
    print(f"Offset linear: {offset_lin}")
    
    # k0 = 1 / r0 * np.exp((-n_samples+1) * dln)
    # k0 = log_k[0] # this is the last value of the log_k
    # r_end = r0 * np.exp((n_samples - 1) * dln)
    # print(k0 * r_end)
    f = function(log_r, transform=False)
    
    # prepare the scipy f function from the Siegman (physical)
    fpy = log_r * f
    fpy = np.pad(fpy, (0, n_samples), 'constant', constant_values=0)
    g_scipy = sp.fftshift(sp.fht(fpy, 
               dln=dln, 
               mu=0, 
               offset=np.log(offset_lin), 
               bias = 0.0)
            )
    # get the Siegman transformed function from scipy
    _, _, g_siegman   = tr.naive_siegman(fpy[:n_samples], dln, r0, k0)
    
    equivalent_fresnel = r0 # optimal fresnel
    _, _, g_magni = tr.fhta_single(fpy[:n_samples], dln, r0, fresnel=equivalent_fresnel) # r0 is optimized with Eq.(5)
    
    plt.figure(figsize=(4, 2.5))
    plt.plot(idxp, fpy, label='Original Data')
    plt.xlabel('idx')
    plt.tight_layout()
    plt.savefig('media/'+label+'initial.png')
    plt.clf()

    # physics
    exact = function(log_k_p, transform=True)
    plt.figure(figsize=(4, 2.5))
    plt.plot(log_k_p, g_scipy / (log_k_p * fresnel), label='Transformed Data')
    plt.plot(log_k_p, exact, label='Transformed Data')
    plt.xlabel(r'$k$')
    plt.xlim(0, n_samples/4)
    plt.ylim(-0.5, 1.6)
    plt.tight_layout()
    name = 'media/'+label+'_physical_transform.pdf'
    plt.savefig(name)
    print("Saved to ", name)
    
    # samples
    exact = function(log_k_p, transform=True)
    plt.figure(figsize=(4, 2.5))
    plt.plot(idxp, g_scipy, label="scipy", lw=0.1)
    plt.plot(idxp, g_siegman, label="siegman", lw=0.3, ls="--")
    plt.plot(idxp, g_magni * log_k_p, label="magni", lw=0.1)
    plt.plot(idxp, exact/(np.pi), label="exact", lw=0.1)
    plt.legend()
    plt.xlabel('idx')
    plt.axvline(n_samples, color='red', linestyle='--', label='n_samples', lw = 0.1)
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

