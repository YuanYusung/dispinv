import numpy as np
import emcee
import matplotlib.pyplot as plt
from subfunctions import forward_disp

def log_likelihood(p, observed_periods, observed_velocity, mode, sigma=0.01):
    """
    Log likelihood function for the MCMC inversion.
    """
    try:
        model_velocity = forward_disp(p, observed_periods, mode=mode)  
        residual = observed_velocity - model_velocity
        likelihood_value = -0.5 * np.sum((residual / sigma) ** 2)

    except Exception as e:
        # 如果forward_disp计算失败，返回无穷大的残差值
        likelihood_value = -np.inf
    return likelihood_value

def log_prior(p, thk_bounds=(0.001, 1.0), vs_bounds=(0.05, 2.5)):
    if len(p) % 2 == 0:
        return -np.inf

    n_layers = (len(p) + 1) // 2
    thk = p[:n_layers-1]  # Thickness parameters
    vs = p[n_layers-1:]   # Shear wave velocity parameters

    if np.any(thk < thk_bounds[0]) or np.any(thk > thk_bounds[1]):
        return -np.inf  # Invalid prior if any thickness is out of range
    if np.any(vs < vs_bounds[0]) or np.any(vs > vs_bounds[1]):
        return -np.inf  # Invalid prior if any velocity is out of range
    
    # if np.any(np.diff(vs) < 0):
    #     return -np.inf

    return 0.0  # Flat prior between the defined bounds

def log_probability(p, observed_periods, observed_velocity, mode, sigma=0.01):
    """
    Log posterior function (likelihood + prior).
    """
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(p, observed_periods, observed_velocity, mode, sigma=sigma)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def initialize_walkers(x0, nwalkers, n_layers):

    p0 = np.tile(x0, (nwalkers, 1))
    # Initialize the walkers in a Gaussian distribution around x0
    # choose the perturbation level according to the parameter scale in your case
    p0[:, :n_layers - 1] += 0.02 * np.random.randn(nwalkers, n_layers - 1)
    p0[:, n_layers - 1:] += 0.02 * np.random.randn(nwalkers, n_layers)

    # 
    for i in range(nwalkers):
        while not np.isfinite(log_prior(p0[i])):
            p0[i, :n_layers - 1] = x0[:n_layers - 1] + 0.002 * np.random.randn(n_layers - 1)
            p0[i, n_layers - 1:] = x0[n_layers - 1:] + 0.02 * np.random.randn(n_layers)

    return p0

def run_mcmc(observed_periods, observed_velocity, mode, x0, n_layers, sigma=0.02, nwalkers=100, nsteps=500):
    """
    Run MCMC sampling using emcee.
    """
    ndim = len(x0)
    p0 = initialize_walkers(x0, nwalkers, n_layers)

    # Set up the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(observed_periods, observed_velocity, mode, sigma))

    # Run the MCMC chain
    print("Running MCMC...")
    sampler.run_mcmc(p0, nsteps, progress=True)

    # Return the MCMC results
    return sampler