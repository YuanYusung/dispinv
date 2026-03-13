from scipy.optimize import minimize
import numpy as np
from subfunctions import forward_disp, para2model

def invert_dispersion_bfgs(
    observed_periods,
    observed_velocity,
    mode,
    x0,
    n_layers,
):
    """
    使用BFGS算法进行Rayleigh色散反演。

    x0: 初始参数向量 [thk(n-1), vs(n)]
    n_layers: 层数 (= len(vs))
    """
    x0 = np.asarray(x0, dtype=float)
    assert len(x0) == (n_layers - 1) + n_layers

    # 定义目标函数和梯度
    def objective_and_grad(p):
        residual, grad = misfit_residual_with_grad(p, observed_periods, observed_velocity, mode=mode)
        return residual, grad

    # 使用BFGS算法进行最小化
    result = minimize(
        lambda p: objective_and_grad(p)[0],  # 目标函数
        x0=x0,  # 初始猜测
        jac=lambda p: objective_and_grad(p)[1],  # 梯度
        method='BFGS',  # 使用BFGS方法
            options={
        'disp': True,               # 打印优化过程信息
        'maxiter': 5000,            # 最大迭代次数
        'gtol': 1e-3,               # 增加梯度容忍度（降低精度要求）
    }
    )

    # 获取最佳模型
    best_model = para2model(result.x)
    return result, best_model

import numpy as np

def misfit_residual(p, observed_periods, observed_velocity, mode):
    """
    计算目标函数的残差（不计算梯度），用于BFGS优化。
    """
    # 计算模型预测值和误差残差
    model_velocity = forward_disp(p, observed_periods, mode=mode)
    residual = observed_velocity - model_velocity
    
    return np.sum(residual**2)  # 返回残差的平方和

def compute_gradient(p, observed_periods, observed_velocity, mode):
    """
    计算目标函数关于参数的梯度（使用有限差分法）。
    """
    epsilon = 1e-4  # 小的扰动，用于数值计算梯度
    grad = np.zeros_like(p)
    
    # 对每个参数使用有限差分计算梯度
    for j in range(len(p)):
        p_plus = np.copy(p)
        p_plus[j] += epsilon
        p_minus = np.copy(p)
        p_minus[j] -= epsilon
        
        # 计算目标函数在扰动后的值
        residual_plus = misfit_residual(p_plus, observed_periods, observed_velocity, mode)
        residual_minus = misfit_residual(p_minus, observed_periods, observed_velocity, mode)
        
        # 数值计算梯度（有限差分）
        grad[j] = (residual_plus - residual_minus) / (2 * epsilon)
    
    return grad

def misfit_residual_with_grad(p, observed_periods, observed_velocity, mode):
    """
    计算目标函数的残差和梯度，用于BFGS优化。
    """
    # 计算目标函数的残差
    residual = misfit_residual(p, observed_periods, observed_velocity, mode)
    
    # 计算梯度
    grad = compute_gradient(p, observed_periods, observed_velocity, mode)
    
    return residual, grad
