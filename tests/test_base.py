from fhatha import transform as tr
from fhatha import mod as md
import numpy as np
from scipy.special import jv, j0
import scipy.fft as sp
from matplotlib import pyplot as plt
from functions import magni, flat_top


def test_naive_magni(fresnel = 10, clip_reference = 1, func = magni):
    n_samples = 512 # with 128 it is really really good
    # FIXME hypotesis: one needs to be aware of how high it sets the minimum radius. it is not good to have it too low, since then an enormous number of radii will be low.
    idx = np.arange((n_samples))
    idxp = np.arange((2*n_samples))
   
    log_r_magni, r0, dln = tr.magni_grid(n_samples)
    log_r_p = r0 * np.exp((idxp - n_samples + 1) * dln) # this is the first value of the log_r, by construction
    print("selected range:", log_r_magni[:4], "...", log_r_magni[-4:])
    print("first value: ", r0,"last value:", log_r_magni[-1])
    log_k     = log_r_magni
    
    f_siegman       = func(r0 * np.exp(np.arange(n_samples) * dln), transform=False) 
    _, ys, g_siegman   = tr.naive_siegman(f_siegman, dln, r0, fresnel)
    
    ### CALCULATIONS WITHIN THE CLASS
    fh = md.FastAccurateHankel(n_samples, fresnel)
    f_organized = fh.pad_2x(fh.sample(lambda k: func(k, transform=False)))
    g_m = fh.fht(f_organized)
    
    exact = func(2 * np.pi * fresnel * log_k * (1-1/n_samples), transform=True) # adapt for the calculation with the actual eta
    fig, ax1 = plt.subplots(figsize=(5, 4))
    magnic = 'blue'
    siegmanc = 'green'
    exactc = 'red'
    color1b = 'black'
    ax1.plot(log_k, np.real(exact)[:n_samples], lw=0.1, ls="-", color=exactc, label="EXACT")
   
    ax1.plot(log_k, np.real(g_m)[:n_samples], lw=0.1, ls="-", color=magnic, label="MAGNI")
    ax1.plot(log_k, np.real(g_siegman)[:n_samples], lw=0.1, ls="-", color=siegmanc, label="SIEGMAN")
    ax1.set_ylabel("reference", color=exactc)
    ax1.tick_params(axis='y', labelcolor="black")
    # Shared x-axis
    ax1.legend()
    ax1.set_xlabel(r'$k$')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(3, 0))

    ax2 = ax1.twinx()

    ax2.tick_params(axis='y', labelcolor="black")
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(3, 0))
    ax1.axhline(0, color='black', lw=0.1, ls="--")
    plt.tight_layout()
    name = 'media/naive.pdf'
    plt.savefig(name)
    print(f"Saved plot to {name}")
  
    plt.figure(figsize=(6, 3.5))
    plt.plot(log_k, np.clip(np.real(g_m[:n_samples]-exact), a_min =-1e-2, a_max=1e-2  ),   lw=0.6, ls="-", label=f"MAGNI", color=magnic)
    plt.plot(log_k, np.clip(np.real(g_siegman[:n_samples]-exact), a_min =-1e-2, a_max=1e-2),   lw=0.6, ls="-", label=f"SIEGMAN", color=siegmanc)
    plt.xlabel(r'$k$')
    plt.ylabel(r"$\varepsilon_a$")
    plt.ylim([-5e-3, 5e-3])
    plt.legend()
    plt.tight_layout()
    name = 'media/error.pdf'
    plt.savefig(name)
    print(f"Saved plot to {name}")
    
if __name__ == "__main__":
    test_naive_magni(clip_reference=2)