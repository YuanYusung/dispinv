# dispinv
A simple demo repository for surface-wave dispersion inversion using several optimization strategies.

This project is intended for **initial learning and testing** of 1D layered-model inversion for Rayleigh-wave phase velocity dispersion curves. It now includes forward modeling and three inversion approaches:

- **Least Squares**
- **BFGS**
- **MCMC**
![Demo](BFGS_inversion_result.png)
![Demo](MCMC_inversion_result.png)
## Features

- Simple layered-earth parameterization:  
  `parameters = [thk(n-1), vs(n)]`
- Forward calculation of Rayleigh-wave phase velocity dispersion
- Conversion from inversion parameters to a full model:  
  `[thk, vp, vs, rho]`
- Example inversion workflows in Python / notebook form

## Repository structure

- `subfunctions.py`  
  Core utility functions, including:
  - parameter-to-model conversion
  - forward dispersion calculation

- `BFGS.py`  
  Dispersion inversion using the BFGS optimization method

- `Least_squares.py`  
  Dispersion inversion using `scipy.optimize.least_squares`

- `MCMC_inversion.py`  
  Dispersion inversion using `emcee` for posterior sampling

- `demo.ipynb`  
  Example notebook for testing and visualization

## Dependencies

This project depends on:

- `numpy`
- `scipy`
- `matplotlib`
- `emcee`
- `disba`

You can install them with:

```bash
pip install numpy scipy matplotlib emcee disba
```


## Quick example

```python
from MCMC_inversion import run_mcmc
import numpy as np

mode = 0

thk0 = np.array([0.01, 0.04])     # km
vs0  = np.array([0.2, 0.6, 1.1])  # km/s
n_layers = len(vs0)

x0 = np.r_[thk0, vs0]

sampler = run_mcmc(
    observed_periods=syn_period,
    observed_velocity=syn_velocity,
    mode=mode,
    x0=x0,
    n_layers=n_layers,
    sigma=0.02,
    nwalkers=100,
    nsteps=500
)
```

See demo.ipynb for a more complete example with plotting.

## Notes
- This repository is mainly a demo codebase for learning
- The current implementation is designed for simple 1D layered models
- The inversion setup, parameter bounds, and empirical relations should be adjusted depending on the target problem
- MCMC results should generally be interpreted with burn-in removal and convergence checks in real applications



## Joint fundamental Rayleigh + Love inversion (restricted VTI)

Run `BFGS_joint_VTI.py` to invert layered **Vsv** and **radial anisotropy** with BFGS. Original isotropic scripts remain unchanged.

Parameters: `gamma = Vsh/Vsv - 1`; the output also includes `xi = (Vsh/Vsv)^2 = (1+gamma)^2`. **Finite layer thickness, density and Vp/Vsv are fixed.** The VTI constants are `A=C=rho*Vp^2`, `F=A-2L`, `L=rho*Vsv^2`, `N=rho*Vsh^2`. Rayleigh uses the isotropic P-SV subsystem via `disba`, while Love solves the VTI SH equation with both L and N. This is a **restricted VTI**, not general five-constant VTI.

Generate and invert synthetic curves:

```bash
python BFGS_joint_VTI.py --demo --output-dir demo_vti
```

Input observation CSVs need `period_s,c_km_s,sigma_km_s` columns, or `frequency_hz` instead of `period_s`. Uncertainty is optional. Periods for Rayleigh and Love may differ. Initial model CSV columns: `thickness_km,vsv_km_s,rho_g_cm3,gamma_init` (last row is the half-space with thickness zero).

```bash
python BFGS_joint_VTI.py --rayleigh rayleigh.csv --love love.csv \
  --model initial_model.csv --output-dir results \
  --gamma-min -0.15 --gamma-max 0.15 --maxiter 150
```

Check `results/summary.json` for an actual `success: true` convergence status. Other outputs: `inverted_model.csv`, `rayleigh_fit.csv`, `love_fit.csv`, and `joint_inversion.png`.

Regression test:

```bash
python -m unittest discover -s tests -p 'test_joint_vti.py' -v
```

**Limitations:** A good two-curve fit does not prove uniqueness of Vs and gamma, especially at depth. Fixed layer geometry, Vp/Vsv, regularization and azimuthal structure require sensitivity analysis before interpretation. This model cannot resolve general VTI P anisotropy or HTI fracture effects.
