from fhatha import misc as tr
from fhatha import transform as md
import numpy as np
from scipy.special import jv, j0
import scipy.fft as sp
from matplotlib import pyplot as plt
from functions import magni, flat_top, gaussian


def test_inverse_transform(fresnel = 10, clip_reference = 1, func = magni):
    n_samples = 1024 # with 128 it is really really good
    # FIXME hypotesis: one needs to be aware of how high it sets the minimum radius. it is not good to have it too low, since then an enormous number of radii will be low.
    idx = np.arange((n_samples))
    idxp = np.arange((2*n_samples))
    r_max = 1/2
    
    log_r_magni, r0, dln = tr.magni_grid(n_samples)
    log_r_p = r0 * np.exp((idxp - n_samples + 1) * dln) # this is the first value of the log_r, by construction
    print("selected range:", log_r_magni[:4], "...", log_r_magni[-4:])
    print("first value: ", r0,"last value:", log_r_magni[-1])
    log_k     = 2*np.pi/log_r_magni[::-1]
    
    f_siegman       = func(r_max * r0 * np.exp(np.arange(n_samples) * dln), transform=False) 
    _, ys, g_siegman   = tr.naive_siegman(f_siegman, dln, r0, fresnel, r_max)
    
    ### CALCULATIONS WITHIN THE CLASS
    fh = md.FastAccurateHankel(n_samples, r_max = r_max)
    f_organized = fh.sample(lambda r: func(r*r_max, transform=False))
    g_m = fh.fht(f_organized)
    
    f_tmp = f_organized
    for rr in range(5000):
        g_m = fh.fht(f_organized)
        f_tmp = fh.ifht(g_m)
    
    ig_magni = f_tmp
    print(np.max(np.abs(ig_magni - f_organized))) 
    g_siegman /= r_max**4
    
    exact = func(log_k, transform=True) # FIXME adapt for the calculation with the actual eta
    
    ## PLOTTIN
    fig, ax1 = plt.subplots(figsize=(5, 4))
    magnic = 'blue'
    siegmanc = 'green'
    exactc = 'red'
    color1b = 'black'
    ax1.plot(log_k, np.real(f_organized)[:n_samples], lw=1.3, ls="--", color=exactc, label="EXACT")
    ax1.plot(log_k, np.real(ig_magni)[:n_samples], lw=0.1, ls="-", color=magnic, label="MAGNI")
    
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
    name = 'media/naive_inverse.pdf'
    plt.savefig(name)
    print(f"Saved plot to {name}")
  
    plt.figure(figsize=(6, 3.5))
    plt.plot(log_k, np.clip(np.real(g_m[:n_samples]-exact), a_min =-1e-2, a_max=1e-2  ),   lw=0.6, ls="-", label=f"MAGNI", color=magnic)
    plt.plot(log_k, np.clip(np.real(g_siegman[:n_samples]-exact), a_min =-1e-2, a_max=1e-2),   lw=0.6, ls="-", label=f"SIEGMAN", color=siegmanc)
    plt.xlabel(r'$k$')
    plt.ylabel(r"$\varepsilon_a$")
    plt.ylim([-1e-3, 1e-3])
    plt.legend()
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.tight_layout()
    name = 'media/error_inverse.pdf'
    plt.savefig(name)
    print(f"Saved plot to {name}")
    
if __name__ == "__main__":
    test_inverse_transform(clip_reference=2, func=gaussian)