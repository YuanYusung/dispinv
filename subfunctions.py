import numpy as np
from disba import PhaseDispersion

def para2model(parameters):
    """
    Convert parameter vector to a layered model matrix for disba.

    parameters = [thk(n-1), vs(n)]
    returns model with columns: [thk(n), vp(n), vs(n), rho(n)]
    """
    parameters = np.asarray(parameters, dtype=float)
    n_layers = (len(parameters) + 1) // 2

    thk = parameters[:n_layers - 1].copy()     # (n-1,)
    vs  = parameters[n_layers - 1:].copy()     # (n,)

    vp = vs * 2.0                              # simple assumption: Vp = 2*Vs
    rho = np.full(n_layers, 2.0, dtype=float)  # constant density as an array (n,)

    # disba expects thickness per layer; last layer thickness = 0 for half-space
    thk_full = np.append(thk, 0.0)

    return np.column_stack((thk_full, vp, vs, rho))

def params2_thk_vs(parameters, last_thk_for_plot=1.0):
    """
    Parse parameters = [thk(n-1), vs(n)] into arrays for depthplot.
    """
    parameters = np.asarray(parameters, dtype=float)
    n_layers = (len(parameters) + 1) // 2

    thk = parameters[:n_layers - 1].copy()
    vs  = parameters[n_layers - 1:].copy()

    thk_plot = np.append(thk, last_thk_for_plot)
    return thk_plot, vs

def forward_disp(parameters, periods, mode=0):
    """
    Predict Rayleigh phase velocity at given periods using disba.
    """
    model = para2model(parameters)
    pd = PhaseDispersion(*model.T)
    cpr = pd(periods, mode=mode, wave="rayleigh")

    # Ensure output matches the given periods order
    return np.interp(periods, cpr.period, cpr.velocity)