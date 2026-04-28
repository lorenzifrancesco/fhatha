# Implementation of the Fast HAnkel Transform of High Accuracy (FHATHA) of Magni et al (1992).

## Sampling
The sampling is done in logarithmic space, defined based on the number of discretization points $N$.
- The logarithmic step $\alpha$ is built in a way to assure that the first and the last interval are of the same lenght.
- The starting position $x_0$ is defined such that all the sampling positions $x_i$ are in the middle of each interval, except for $i=0$.
Within these definitions, the interval boundaries are 
$$
\xi_i=\exp[\alpha(i-N)] \quad i=0, ..., N
$$
and the center sampling points are 
$$
x_i = x_0 \exp[\alpha i]
$$

Given a physical function $f(r)$ defined in the radii $[0, r_{\mathrm{max}}]$, to get the corresponding transform we perform a mapping into the numerical space $[0, 1]$ and perform a sampling $f_i = f(r_\mathrm{max} x_i)$. The corresponding momentum space is from.
Physical values of the momentum for such a function are spaced by $dk=2\pi/r_{\mathrm{max}}$.
The shortest spacing we get in space may not be the first interval, which is as long as the last interval, but it is of the same order, approximately. Therefore, the maximum momentum is about $2\pi/(x_0 \times r_\mathrm{max})$.

The transform map the function to the momentum space 
$$
K_i = 2 \pi N_f x_0 \exp[\alpha i]
$$
providing $\hat{f}(K_i)$, with $i=0, ..., N-1$, the sampled Hankel transform defined as
$$
\hat{f}(y) = 2 \pi \int_0^1 dx \, x \, f(x) \operatorname{J}_0(yx)
$$
The transform is involutive, and the numerical method is approximately involutive.
The form of this transform is justified by the far-field diffraction integrals (Fraunhofer regime). 
In this view, the transform maps the function in radial space to a propagated function in radial space. The usage of the same space vectors allows for repeated application of the function.
Since we are focused (no pun intended) in the near field, we consider the simplified version of the transform with $N_f = 1/x_0$, i.e. real space in $[$ about $ x_0, 1]$, and momentum space in $2\pi\times[1, 1/x_0]$. Imposing the log shape of the momentum space:
$$
K_i = 2 \pi \exp[\alpha i]
$$
considering that $\exp[\alpha N] \sim x_0$, this makes sense.
(remember the DFT kernel $\exp[i2\pi ij/N]$).

### Construction of the right basis