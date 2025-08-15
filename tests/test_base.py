from fhatha import transform as tr
from fhatha import mod as md
import numpy as np
from scipy.special import jv, j0
import scipy.fft as sp
from matplotlib import pyplot as plt
from functions import magni, flat_top


def test_naive_magni(fresnel = 10, clip_reference = 1, func = magni):
    n_samples = 1024 # with 128 it is really really good
    # FIXME hypotesis: one needs to be aware of how high it sets the minimum radius. it is not good to have it too low, since then an enormous number of radii will be low.
    idx = np.arange((n_samples))
    idxp = np.arange((2*n_samples))
   
    log_r_magni, r0, dln = tr.magni_grid(n_samples)
    log_r_p = r0 * np.exp((idxp - n_samples + 1) * dln) # this is the first value of the log_r, by construction
    print("selected range:", log_r_magni[:4], "...", log_r_magni[-4:])
    print("first value: ", r0,"last value:", log_r_magni[-1])
    log_k     = log_r_magni
    log_k_p   = log_r_p
    
    f       = func(log_r_magni, transform=False)
    f_magni = func(log_r_magni, transform=False)
    fs   = log_r_magni * f
    fs_m = log_r_magni * f_magni
    fs_m = f_magni
    # fs = np.pad(fs, (0, n_samples), 'constant', constant_values=0) -> we are already padding  
    equivalent_k0 = r0 * fresnel # scaling the k space for agreement with Siegman
    _, _, g_siegman   = tr.naive_siegman(fs, dln, r0, equivalent_k0)
    _, _, g_m = tr.fhta_single(fs_m, fresnel=fresnel)
    # _, _, g_m_fool = tr.foolproof_magni(magni, n_samples, fresnel)
    g_siegman /= log_k_p # this seems to be needed. Still the oscillations seem to be wrong.
    
    # We notice (check.pdf)
    # - the magni transform is out of scale, but behaves good (fabout 11 over 1.3 desired)
    # - the Siegman transform is very much in scale, but the equivalent k0 I think we are missing it
    # - every transform starts from the same point at null k 
    # assert np.isclose(g, 0.0)
    ### CALCULATIONS WITHIN THE CLASS
    fh = md.FastAccurateHankel(n_samples, fresnel)
    f_organized = fh.pad_2x(fh.sample(lambda k: func(k, transform=False)))
    g_organized = fh.fht(f_organized)
    assert(np.isclose(g_organized, g_m[:n_samples]).all())
    
    exact = func(2 * np.pi * fresnel * log_k * (1-1/n_samples), transform=True) # adapt for the calculation with the actual eta
    fig, ax1 = plt.subplots(figsize=(5, 4))
    magnic = 'blue'
    siegmanc = 'green'
    exactc = 'red'
    color1b = 'black'
    ax1.plot(log_k, np.real(exact)[:n_samples], lw=0.1, ls="-", color=exactc, label="EXACT")
   
    ax1.plot(log_k, np.real(g_m)[:n_samples], lw=0.1, ls="-", color=magnic, label="MAGNI")
    ax1.plot(log_k, np.real(g_siegman)[:n_samples], lw=0.1, ls="-", color=siegmanc, label="SIEGMAN")
    ax1.plot(log_k, np.real(g_organized)[:n_samples], lw=0.5, ls=":", color=magnic, label="MAGNI ORGANIZED")
    # ax1.plot(log_k, np.real(g_m_fool)[:n_samples], lw=0.5, ls=":", color=color1b)
    ax1.set_ylabel("reference", color=exactc)
    ax1.tick_params(axis='y', labelcolor="black")
    # Shared x-axis
    ax1.legend()
    ax1.set_xlabel(r'$k$')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(3, 0))

    # T_max_high = 
    # Second axis: T_max
    ax2 = ax1.twinx()

    # ax2.plot(log_k, np.real(g_siegman)[:n_samples], lw=0.1, ls="-", color=color2)
    # ax2.set_ylabel("siegman", color=color2)
    ax2.tick_params(axis='y', labelcolor="black")
    # ax2.plot(log_r_magni, np.imag(g_m)[:n_samples], lw=1.2, ls=":")
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(3, 0))
    ax1.axhline(0, color='black', lw=0.1, ls="--")
    plt.tight_layout()
    name = 'media/naive.pdf'
    plt.savefig(name)
    print(f"Saved plot to {name}")
  
    plt.figure(figsize=(6, 3.5))
    plt.plot(log_k, np.clip(np.real(g_m[:n_samples]-exact), a_min =-1e-2, a_max=1e-2  ),   lw=0.6, ls="-", label=f"MAGNI", color=magnic)
    # plt.plot(log_k, np.clip(np.real(g_m_fool[:n_samples]-exact), a_min =-1e-2, a_max=1e-2  ),   lw=0.6, ls="-", label=f"abs error foolproof")
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
    exit()
    for ff in [flat_top, magni]:
        test_siegman(function=ff, label=ff.__name__, fresnel=1)
    # test_siegman(function=magni, label="ft", fresnel = 10)
    test_naive_magni(clip_reference=2)
    test_siegman(function=flat_top, label="ft", fresnel = 1)
    test_siegman(function=magni,    label="mg", fresnel = 3)