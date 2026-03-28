import numpy as np
import scipy
from typing import Optional
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import simpson, trapezoid, quad
from .experimental_kinematics import ref_deviatoric, invariants, right_cauchy_green

def hyperelastic_coefficients(C, S_h):
    """
    Function to compute the isochoric hyperelastic response functions of the 2nd PK stress, given the right Cauchy-Green
    deformation tensors and the corresponding isochoric hyperelastic 2nd PK stresses (both in the form of a 3x3 matrices)
    are provided as a list.
    """
    gamma1, gamma2 = [], []
    
    for i in range(len(C)):
        # Setting up the linear least squares optimization problem, A @ x = b
        C_temp = np.array(C[i])
        Dev_1_temp = ref_deviatoric(np.eye(3), C_temp) # This is G2 (Second component of the tensor basis).
        J_temp = 1
        C_bar_temp = (J_temp ** (-2/3)) * C_temp        
        Dev_C_bar_temp = ref_deviatoric(C_bar_temp, C_temp) # This is G3 (Third component of the tensor basis).
        S_temp = np.array(S_h[i])

        # Calculating the components of A considering the Lagrange multiplier (uniaxial tension)
        C_inv_temp = np.linalg.inv(C_temp)
        A_1 = Dev_1_temp - ((Dev_1_temp[2,2] / C_inv_temp[2,2]) * C_inv_temp)
        A_2 = Dev_C_bar_temp - ((Dev_C_bar_temp[2,2] / C_inv_temp[2,2]) * C_inv_temp)
        
        A = np.vstack((np.ravel(A_1), np.ravel(A_2))).T
        b = np.ravel(J_temp ** (2/3) * S_temp).T
        x = scipy.optimize.lsq_linear(A, b, tol=1e-12, max_iter = 10000).x
        gamma1.append(x[0])
        gamma2.append(x[1])
        
    return np.array(gamma1), np.array(gamma2)

    
def damage_coefficients(S_h, S):
    """
    Function to compute the damage factor Chi (d(psi)/dW) of the 2nd PK stress, given the hyperelastic / intact 2nd PK
    stress, and the corresponding total 2nd PK stresses provided as a list. Here, individual stresses are in the form of
    3x3 matrices.
    """
    xi = []
    for i in range(len(S_h)):
        S_h_temp = np.array(S_h[i])
        S_temp = np.array(S[i])
        
        A = np.ravel(S_h_temp).T
        A = A.reshape(-1,1)
        b = np.ravel(S_temp).T

        x = scipy.optimize.lsq_linear(A, b, tol=1e-12, max_iter = 10000).x
        xi.append(x)
        
    return np.array(xi)

    
def strain_energy_density(C, S_h):
    """
    Function to compute strain energy density by integrating hyperelastic stress, S_h (or total stress, S), with respect to
    dC (increment in the right Cauchy-Green deformation tensor) using the Trapezoidal rule.
    W = integration of (S_h/2):dC from I to C.
    """
    W = []
    W_temp = 0
    W.append(W_temp)
    for i in range(1,len(C)):
        dC = C[i] - C[i-1]
        W_temp = W_temp + np.tensordot(((S_h[i]+S_h[i-1])/4),dC) # Trapezoidal Rule
        W.append(W_temp)
    return W


def energy_spline_adaptive(C_list, S_list, t: Optional[np.ndarray] = None, atol=1e-10, rtol=1e-10):
    """
    Build cubic splines for each independent component of C(t) and S(t), then integrate
    f(t) = 0.5 * S(t) : dC/dt(t) over t0..tN with adaptive quadrature.
    This is an alternative Spline + Adaptive Quadrature method to integrate S over C to get the strain energy density W.
    Returns cumulative W as a list.
    """
    assert CubicSpline is not None and quad is not None, "SciPy is required for the spline+adaptive method."
    C = np.array(C_list, dtype=float)
    S = np.array(S_list, dtype=float)
    N = C.shape[0]

    if t is None:
        t = _arc_length_param(C)
    else:
        t = np.asarray(t, dtype=float)
        assert t.shape == (N,)
    assert np.all(np.diff(t) > 0)

    idxs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    C_spl = []
    S_spl = []
    for (i, j) in idxs:
        C_spl.append(CubicSpline(t, C[:, i, j], bc_type="natural"))
        S_spl.append(CubicSpline(t, S[:, i, j], bc_type="natural"))

    def f(tt: float) -> float:
        val = 0.0
        for k in range(6):
            val += S_spl[k](tt) * C_spl[k].derivative()(tt)
        return 0.5 * val

    W = [0.0]
    for i in range(1, N):
        Wi, _ = quad(f, float(t[i - 1]), float(t[i]), epsabs=atol, epsrel=rtol, limit=200)
        W.append(W[-1] + Wi)
    return W


def energy_gauss_per_interval(C_list, S_list, t: Optional[np.ndarray] = None, order: int = 2):
    """
    Per-interval Gauss-Legendre quadrature of f(t)=0.5*S(t):dC/dt over each [t_i, t_{i+1}].
    This is an alternative Gauss-Legendre per interval method to integrate S over C to get the strain energy density W.
    Returns cumulative W as a list.
    """
    assert order in (2, 3)
    C = np.array(C_list, dtype=float)
    S = np.array(S_list, dtype=float)
    N = C.shape[0]

    if t is None:
        t = _arc_length_param(C)
    else:
        t = np.asarray(t, dtype=float)
        assert t.shape == (N,)
    assert np.all(np.diff(t) > 0)

    if order == 2:
        xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        w = np.array([1.0, 1.0])
    else:
        xi = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        w = np.array([5/9, 8/9, 5/9])

    W = [0.0]
    for i in range(1, N):
        t0, t1 = t[i - 1], t[i]
        dt = t1 - t0
        dC = C[i] - C[i - 1]
        dCdt = dC / dt

        S0, S1 = S[i - 1], S[i]
        dS = S1 - S0
        accum = 0.0
        tg = 0.5 * (t1 + t0) + 0.5 * dt * xi
        for g in range(len(w)):
            alpha = (tg[g] - t0) / dt
            Sg = S0 + alpha * dS
            #accum += w[g] * _frobenius_dot(Sg, dCdt)
            accum += w[g] * float(np.tensordot(Sg, dCdt))
        Wi = 0.5 * (0.5 * dt) * accum
        W.append(W[-1] + Wi)
    return W