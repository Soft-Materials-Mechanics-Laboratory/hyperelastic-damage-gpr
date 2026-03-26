import numpy as np

"""
Helper functions for monotonicity-and-non-negativity-constrained GPR model.
"""
# # === RBF Kernel and Derivatives (Alternative to Matern 5/2; not used in this work) ===
# def rbf_kernel(x1, x2, lengthscale, variance):
#     x1 = np.atleast_2d(x1)
#     x2 = np.atleast_2d(x2)
#     dists = cdist(x1, x2, 'sqeuclidean')
#     return variance * np.exp(-0.5 * dists / lengthscale**2)

# def rbf_kernel_dx(x1, x2, lengthscale, variance):
#     x1 = np.atleast_2d(x1)
#     x2 = np.atleast_2d(x2)
#     diff = x1[:, None, :] - x2[None, :, :]
#     base = rbf_kernel(x1, x2, lengthscale, variance)
#     return (-(diff.squeeze(-1)) / lengthscale**2) * base

# def rbf_kernel_dxdx(x1, x2, lengthscale, variance):
#     diff = x1[:, None, :] - x2[None, :, :]
#     sq_dist = np.sum(diff**2, axis=2)
#     base = variance * np.exp(-0.5 * sq_dist / lengthscale**2)
#     term1 = 1 / lengthscale**2
#     term2 = (diff.squeeze(-1)**2) / lengthscale**4
#     return (term1 - term2) * base

# === Matern 5/2 Kernel and Derivatives ===
def matern52_kernel(x1, x2, lengthscale, variance):
    """
    Matern 5/2 kernel.
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    r = np.linalg.norm(x1[:, None] - x2[None, :], axis=-1) + 1e-12
    sqrt5_r = np.sqrt(5) * r / lengthscale
    return variance * (1 + sqrt5_r + 5 * r**2 / (3 * lengthscale**2)) * np.exp(-sqrt5_r)

def matern52_kernel_dx(x1, x2, lengthscale, variance):
    """
    First derivative of the Matern 5/2 kernel.
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    diff = x1[:, None, :] - x2[None, :, :]
    r = np.linalg.norm(diff, axis=-1) + 1e-12
    sqrt5_r = np.sqrt(5) * r / lengthscale
    exp_term = np.exp(-sqrt5_r)
    coef = -5 * variance / (3 * lengthscale**2) # The minus sign here is not present in the formula written in the manuscript. This is because the paper presents the formula for dk(W,W')/dW', whereas the equation here is the derivative wrt the first argument, i.e., dk(W,W')/dW. Note, dk(W,W')/dW = -dk(W,W')/dW'. Therefore, both are equivalent and correct.
    return coef * diff.squeeze(-1) * (1 + sqrt5_r) * exp_term

def matern52_kernel_dxdx(x1, x2, lengthscale, variance):
    """
    Second derivative of the Matern 5/2 kernel.
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    diff = x1[:, None, :] - x2[None, :, :]
    r = np.linalg.norm(diff, axis=-1) + 1e-12
    sqrt5_r = np.sqrt(5) * r / lengthscale
    exp_term = np.exp(-sqrt5_r)
    term1 = 5 / (3 * lengthscale**2)
    term2 = 25 * diff.squeeze(-1)**2 / (3 * lengthscale**4)
    return variance * exp_term * (term1 * (1 + sqrt5_r) - term2)

# === GPR Functions ===
def neg_log_marginal_likelihood(params, X, Y, noise):
    """
    Negative log-marginal likelihood function.
    """
    lengthscale, variance = np.exp(params)
    K = matern52_kernel(X, X, lengthscale, variance) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K))) # Increased the nugget vaue for stability.
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
    log_likelihood = -0.5 * Y.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(X) * np.log(2*np.pi)
    return -log_likelihood.flatten()[0]

def nonnegativity_constraint(params, X_train, Y_train, X_constrain, noise):
    """
    Non-negativity constraint.
    """
    lengthscale, variance = np.exp(params)
    K = matern52_kernel(X_train, X_train, lengthscale, variance) + noise * np.eye(len(X_train))
    Ks = matern52_kernel(X_constrain, X_train, lengthscale, variance)
    Kss = matern52_kernel(X_constrain, X_constrain, lengthscale, variance)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
    mean = Ks @ alpha # Note: The actual formula is mean = K(z,z*)^T @ K(z,z)^-1 @ Y. Here we are using K_s and not K_s^T because the first argument in the Ks (above) is X_constrain. That is, we are getting K_s(z*,z), which is the transpose of K_s(z,z*).
    v = np.linalg.solve(L, Ks.T)
    var = np.diag(Kss - v.T @ v).reshape(-1, 1)
    constraint_value = mean
    return constraint_value.flatten()

def derivative_constraint(params, X_train, Y_train, X_constrain, noise):
    """
    Monotonivity constraint.
    """
    lengthscale, variance = np.exp(params)
    K = matern52_kernel(X_train, X_train, lengthscale, variance) + noise * np.eye(len(X_train))
    Ks = matern52_kernel_dx(X_constrain, X_train, lengthscale, variance)
    Kss = matern52_kernel_dxdx(X_constrain, X_constrain, lengthscale, variance)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
    mean_dy_dx = Ks @ alpha # Note: The actual formula is mean = K(z,z*)^T @ K(z,z)^-1 @ Y. Here we are using K_s and not K_s^T because the first argument in the Ks (above) is X_constrain. That is, we are getting K_s(z*,z), which is the transpose of K_s(z,z*).
    v = np.linalg.solve(L, Ks.T)
    var_dy_dx = np.diag(Kss - v.T @ v).reshape(-1, 1)
    constraint_value = mean_dy_dx #+ 2.0 * np.sqrt(var_dy_dx)
    return -constraint_value.flatten()

def penalized_obj(params, X, Y, Xc, lam, noise):
    """
    Penalty-based objective funtion of constrained GPR.
    """
    base = neg_log_marginal_likelihood(params, X, Y, noise = noise)
    g_pos = nonnegativity_constraint(params, X, Y, Xc, noise = noise)
    g_mon = derivative_constraint(params, X, Y, Xc, noise = noise)
    # Quadratic penalty on violations only
    lam_pos, lam_mon = lam, lam
    penalty = lam_pos * np.sum(np.minimum(0.0, g_pos)**2) + lam_mon * np.sum(np.minimum(0.0, g_mon)**2)
    return base + penalty
    
# === Predictive mean ===
def predict_mu(Xtrain, ytrain, Xtest, l, v, noise):
    """
    GPR mean prediction.
    """
    K = matern52_kernel(Xtrain, Xtrain, l, v) + noise * np.eye(len(Xtrain))
    Ks = matern52_kernel(Xtest, Xtrain, l, v)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytrain))
    return (Ks @ alpha).flatten() # Note: The actual formula is mean = K(z,z*)^T @ K(z,z)^-1 @ Y. Here we are using K_s and not K_s^T because the first argument in the Ks (above) is X_test or X_constrain. That is, we are getting K_s(z*,z), which is the transpose of K_s(z,z*).

# === Predictive variance ===
def predict_var(Xtrain, ytrain, Xtest, l, v, noise):
    """
    GPR variance prediction.
    """
    K = matern52_kernel(Xtrain, Xtrain, l, v) + noise * np.eye(len(Xtrain))
    Ks = matern52_kernel(Xtest, Xtrain, l, v)
    Kss = matern52_kernel(Xtest, Xtest, l, v)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytrain))
    v_temp = np.linalg.solve(L, Ks.T)
    return (np.diag(Kss - v_temp.T @ v_temp)).flatten()
