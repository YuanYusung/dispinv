from scipy.optimize import least_squares
import numpy as np
from subfunctions import forward_disp, para2model
# Functions for least-squares inversion

def misfit_residual(parameters, observed_periods, observed_velocity, mode=0):
    """
    Least-squares residual vector (1D) required by scipy.optimize.least_squares.
    """
    pred = forward_disp(parameters, observed_periods, mode=mode)
    return observed_velocity - pred

def invert_dispersion_ls(
    observed_periods,
    observed_velocity,
    mode,
    x0,
    n_layers,
    thk_bounds=(0.001, 1.0),   # km
    vs_bounds=(0.05,  2.5),    # km/s
    method="trf",
    verbose=1,
    max_nfev=2000,
):
    """
    Least-squares inversion for Rayleigh dispersion.

    x0: initial parameter vector [thk(n-1), vs(n)]
    n_layers: number of layers (= len(vs))
    """
    x0 = np.asarray(x0, dtype=float)
    assert len(x0) == (n_layers - 1) + n_layers

    fun = lambda p: misfit_residual(p, observed_periods, observed_velocity, mode=mode)

    if method.lower() == "lm":
        # LM does NOT support bounds
        res = least_squares(
            fun, x0=x0,
            method="lm",
            verbose=verbose,
            max_nfev=max_nfev,
            x_scale="jac",
        )
    else:
        # Build bounds arrays for TRF / dogbox
        thk_lb = np.full(n_layers - 1, thk_bounds[0], dtype=float)
        thk_ub = np.full(n_layers - 1, thk_bounds[1], dtype=float)
        vs_lb  = np.full(n_layers,     vs_bounds[0],  dtype=float)
        vs_ub  = np.full(n_layers,     vs_bounds[1],  dtype=float)

        lb = np.r_[thk_lb, vs_lb]
        ub = np.r_[thk_ub, vs_ub]

        res = least_squares(
            fun, x0=x0,
            bounds=(lb, ub),
            method=method,
            verbose=verbose,
            max_nfev=max_nfev,
            x_scale="jac",
        )

    best_model = para2model(res.x)
    return res, best_model