from scipy.optimize import minimize
import numpy as np
from subfunctions import forward_disp, para2model

def invert_dispersion_bfgs(observed_periods, observed_velocity, mode, x0, n_layers):
    """
    Invert Rayleigh dispersion using the BFGS algorithm.

    x0: Initial parameter vector [thk(n-1), vs(n)]
    n_layers: Number of layers (= len(vs))
    """
    x0 = np.asarray(x0, dtype=float)
    assert len(x0) == (n_layers - 1) + n_layers

    # Define objective function and gradient
    def objective_and_grad(p):
        residual, grad = misfit_residual_with_grad(p, observed_periods, observed_velocity, mode)
        return residual, grad

    # Use BFGS method to minimize
    result = minimize(
        lambda p: objective_and_grad(p)[0],  # Objective function
        x0=x0,  # Initial guess
        jac=lambda p: objective_and_grad(p)[1],  # Gradient
        method='BFGS',  # BFGS method
        options={
            'disp': True,  # Display optimization info
            'maxiter': 5000,  # Max iterations
            'gtol': 1e-3,  # Gradient tolerance (lower precision requirement)
        }
    )

    # Return the best model
    best_model = para2model(result.x)
    return result, best_model


def misfit_residual(p, observed_periods, observed_velocity, mode):
    """
    Compute the residual (without gradient) for the objective function.
    """
    try:
        model_velocity = forward_disp(p, observed_periods, mode)
        residual = observed_velocity - model_velocity
        return np.sum(residual**2)  # Sum of squared residuals
    except Exception as e:
        return np.inf


def compute_gradient(p, observed_periods, observed_velocity, mode):
    """
    Compute the gradient of the objective function using finite differences.
    """
    epsilon = 1e-4  # Small perturbation for numerical gradient
    grad = np.zeros_like(p)
    
    # Compute the gradient using finite differences
    for j in range(len(p)):
        p_plus = np.copy(p)
        p_plus[j] += epsilon
        p_minus = np.copy(p)
        p_minus[j] -= epsilon
        
        # Compute residual for perturbed parameters
        residual_plus = misfit_residual(p_plus, observed_periods, observed_velocity, mode)
        residual_minus = misfit_residual(p_minus, observed_periods, observed_velocity, mode)
        
        # Numerical gradient
        grad[j] = (residual_plus - residual_minus) / (2 * epsilon)
    
    return grad


def misfit_residual_with_grad(p, observed_periods, observed_velocity, mode):
    """
    Compute both the residual and gradient for BFGS optimization.
    """
    residual = misfit_residual(p, observed_periods, observed_velocity, mode)
    grad = compute_gradient(p, observed_periods, observed_velocity, mode)
    return residual, grad