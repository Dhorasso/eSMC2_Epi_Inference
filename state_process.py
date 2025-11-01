"""
SEIR model with stochastic transmission and Ensemble Kalman Filter (EnKF) update.
"""

import numpy as np
from numpy.linalg import LinAlgError
from scipy.special import gammaln


def state_transition(
    theta: np.ndarray,
    current_state_particles: np.ndarray,
    state_names: list[str],
    theta_names: list[str],
    t_start: int,
    t_end: int,
    obs: float,
    dt: float = 1.0
) -> tuple[np.ndarray, float]:
    """
    Propagate ensemble states forward in time and apply EnKF correction.

    Returns:
        x_corr: Corrected ensemble (num_particles x num_compartments)
        LIK: Unbiased log-likelihood
    """
    t_points = np.arange(t_start, t_end + dt, dt)
    num_steps = len(t_points)
    num_particles, num_compartments = current_state_particles.shape

    # Preallocate prediction array
    x_pred = np.empty((num_steps, num_particles, num_compartments), dtype=float)
    x_pred[0] = current_state_particles

    # Evolve ensemble
    for i in range(1, num_steps):
        x_pred[i] = Forecast_step(x_pred[i - 1], theta, theta_names, dt)

    # EnKF correction using last predicted ensemble
    x_corr, LIK = Update_step(x_pred[-1], obs, theta, theta_names, dt)

    return x_corr, LIK


def Forecast_step(
    x: np.ndarray,
    theta: np.ndarray,
    theta_names: list[str],
    dt: float,
    forecast: bool = False
) -> np.ndarray:
    """
    Single-step state transition for ensemble x.
    """
    S, E, I, R, NI, B = x.T  # Unpack columns

    def get_param(name: str, default: float) -> float:
        return float(theta[theta_names.index(name)]) if name in theta_names else default

    alpha = get_param("alpha", 1.0)
    gamma = get_param("gamma", 1.0)
    phi = get_param("phi", 1e-5)
    rho = get_param("rho", 1.0)
    nu_beta = get_param("nu_beta", 0.15)

    N = S + E + I + R

    # Deterministic flows
    new_exposed = np.exp(B) * S * I / N * dt
    new_infected = alpha * E * dt
    new_recovered = gamma * I * dt

    S_next = S - new_exposed
    E_next = E + new_exposed - new_infected
    I_next = I + new_infected - new_recovered
    R_next = R + new_recovered
    NI_next = rho * new_infected

    # Stochastic transmission term
    B_next = B + nu_beta * np.random.default_rng().standard_normal(size=B.shape[0]) * np.sqrt(dt)

    if forecast:
        B_next = B
        mu = NI_next
        n = np.where(phi > 0, 1.0 / phi, 1e9)
        p = 1.0 / (1.0 + phi * mu)
        NI_next = np.random.negative_binomial(
            np.maximum(1, np.round(n)).astype(int),
            p.clip(1e-12, 1 - 1e-12),
            size=x.shape[0]
        )

    # Assemble next state
    X_next = np.empty_like(x)
    X_next[:, 0] = S_next
    X_next[:, 1] = E_next
    X_next[:, 2] = I_next
    X_next[:, 3] = R_next
    X_next[:, 4] = NI_next
    X_next[:, 5] = B_next

    # Ensure non-negative compartments
    X_next[:, :5] = np.maximum(X_next[:, :5], 1e-5)

    return X_next


def log_c_const(d: int, v: int) -> float:
    """
    Compute log of constant c(d, v) from Ghurye & Olkin (1969).
    """
    log_num = -0.5 * d * v * np.log(2.0) - 0.25 * d * (d - 1) * np.log(np.pi)
    log_denom = np.sum([gammaln(0.5 * (v - i + 1)) for i in range(1, d + 1)])
    return log_num - log_denom


def unbiased_logpdf(y: float, mu_N: float, Sigma_N: float, N: int) -> float:
    """
    Unbiased estimator of log Gaussian density.
    """
    if N <= 4:
        return -np.inf

    denom = 1.0 - 1.0 / N
    diff = y - mu_N
    M_N = (N - 1.0) * Sigma_N
    inner = M_N - diff ** 2 / denom

    if M_N <= 0.0 or inner <= 0.0:
        return -np.inf

    log_term = (
        -0.5 * np.log(2.0 * np.pi)
        + log_c_const(1, N - 2)
        - log_c_const(1, N - 1)
        - 0.5 * np.log(denom)
        - 0.5 * (N - 3.0) * np.log(M_N)
        + 0.5 * (N - 4.0) * np.log(inner)
    )
    return float(log_term)


def Update_step(
    x_pred: np.ndarray,
    obs: float,
    theta: np.ndarray,
    theta_names: list[str],
    dt: float = 1.0
) -> tuple[np.ndarray, float]:
    """
    Ensemble Kalman Filter update step.
    """
    N, D = x_pred.shape
    phi = float(theta[theta_names.index("phi")]) if "phi" in theta_names else 0.0

    H = np.array([[0, 0, 0, 0, 1, 0]])  # Observation operator
    y_pred = x_pred @ H.T
    y_pred = y_pred.reshape(N, 1)

    y_mean = float(np.mean(y_pred))
    V_t = max(0.1, y_mean * (1 + phi * y_mean))

    y_obs = obs + np.random.normal(0.0, np.sqrt(V_t), size=(N, 1))

    x_mean = np.mean(x_pred, axis=0)
    dx = x_pred - x_mean
    dy = y_pred - y_mean

    Pxy = dx.T @ dy / (N - 1.0)
    Pyy = float((dy.T @ dy) / (N - 1.0)) + V_t
    Pyy_safe = max(Pyy, 1e-12)
    K = Pxy / Pyy_safe

    x_next = x_pred + (y_obs - y_pred) @ K.T
    x_next[:, :5] = np.maximum(x_next[:, :5], 1e-5)

    LIK = unbiased_logpdf(obs, y_mean, Pyy, N)

    return x_next, LIK




































    
# ######################################################################################################
# #####  Fonction to propagate the state forward  ######################################################
# #####################################################################################################

# import numpy as np
# import pandas as pd
# from scipy.stats import binom, norm,  multivariate_normal, gamma,truncnorm
# from numpy.random import binomial, normal
# from epi_model import stochastic_seir_model, deterministic_seir_model
# from scipy.special import gamma
# from numpy.linalg import det, LinAlgError
# from scipy.special import gammaln 


# def state_transition(model, theta, current_state_particles, state_hist, state_names, theta_names, t_start, t_end, obs, dt=1):
#     """
#     Solve a stochastic disease model using vectorized computation.

#     Parameters:
#     - model: Model function
#     - theta: Parameter array
#     -  state_hist: Initial state array (num_particles x num_compartments)
#     - state_names: Names of compartments
#     - theta_names: Names of parameters
#     - t_start: Start time
#     - t_end: End time
#     - dt: Time step (default is 1)

#     Returns:
#     - results_df: DataFrame containing results at the last time step
#     """
#     # Time points
#     t_points = np.arange(t_start, t_end + dt, dt)
#     num_steps = len(t_points)

#     # Initialize results array
#     t= t_end
#     num_particles, num_compartments =  state_hist[0].shape
#     x_pred = np.zeros((num_steps, num_particles, num_compartments))
    
#     x_pred[0] =  current_state_particles  # Set initial state
#     LIK=0
#     for i in range(1, num_steps):
#         x_pred[i] = f(x_pred[i-1], state_hist, theta, theta_names, t, dt)

#     x_corr, LIK= enkf_step(x_pred[- 1], state_hist, obs, theta, theta_names, t)
#     return  x_corr, LIK


# def trans_denst(X_next, X_prev, theta, theta_names, dt=1):
#     """
#     Vectorized discrete-time stochastic SEIR compartmental model.
#     """

#     X=f(X_prev, theta, theta_names, dt=1)
#     # pxt = p_se+p_ei+p_ir+p_bt
#     N = X.shape[0]
#     u = np.mean(X, axis=0)
#     S =  (X-u).T @ (X-u) / (N - 1)
#     S += 1e-6 * np.eye(S.shape[0])
#     return multivariate_normal.logpdf(X_next, mean=u, cov=S, allow_singular=True) 
#     # Ensure all compartments remain non-negative
#     # return proposa_denst(X_p)





# def proposa_denst(X):
#     N = X.shape[0]
#     u = np.mean(X, axis=0)
#     S =  (X-u).T @ (X-u) / (N - 1)
#     S += 1e-6 * np.eye(S.shape[0])
#     return multivariate_normal.logpdf(X, mean=u, cov=S, allow_singular=True) 



# def gamma_delay_distribution(mean, sd, max_delay=3):
#     """Discretize a gamma distribution with given mean and sd."""
#     shape = (mean / sd)**2
#     scale = (sd**2) / mean
#     x = np.arange(0, max_delay + 1)
#     probs = gamma.cdf(x + 1, a=shape, scale=scale) - gamma.cdf(x, a=shape, scale=scale)
#     return probs / probs.sum()

# def apply_reporting_delay(incidence_1d):
#     delay_probs = gamma_delay_distribution(2, 1,3)
#     reported = np.convolve(incidence_1d, delay_probs, mode='full')[:len(incidence_1d)]
#     return reported

# def apply_reporting_delay_to_all(exposure_hist):
#     # exposure_hist shape: (timesteps, particles)
#     num_timesteps, num_particles = exposure_hist.shape
#     reported_all = np.zeros_like(exposure_hist)
#     for p in range(num_particles):
#         reported_all[:, p] = apply_reporting_delay(exposure_hist[:, p])
#     return reported_all[-1]


# def f(x, state_hist, theta, theta_names, t, dt=1):
#     """
#     x: current states, shape (num_particles, num_states)
#     state_hist: full state history array, shape (time, num_particles, num_states)
#     theta: parameter vector, shape (num_params,)
#     theta_names: list of parameter names
#     t: current time step index
#     dt: time step size
#     """
#     S, E, I, R, NI= x.T
#     N = S + E + I + R

#     param = dict(zip(theta_names, theta))
#     B= param['B']
#     alpha = param['alpha']
#     gamma = param['gamma']
#     rho = param['rho']
#     nu_beta = param.get('nu_beta', 0.1)

#     new_exposed =B* S * I / N * dt
#     new_infected = alpha * E*dt
#     new_recovered = gamma * I*dt

#     S_next = S - new_exposed
#     E_next = E + new_exposed - new_infected
#     I_next = I + new_infected - new_recovered
#     R_next = R + new_recovered
#     NI_next = rho*new_infected 
#     # B_next = B + nu_beta * np.random.normal(0, 1, size=B.shape) * dt

#     # X_next = np.column_stack((S_next, E_next, I_next, R_next, NI_next, B_next))
#     # state_hist[t] = X_next

#     # # Vectorized reporting delay application
#     # exposure_hist = state_hist[:t+1, :, 4]  # full history of new infections
#     # delayed_NI = apply_reporting_delay_to_all(exposure_hist)  # shape (timesteps, particles)
#     # NI_next = delayed_NI # delayed new infections at current timestep, shape (num_particles,)

#     X_next = np.column_stack((S_next, E_next, I_next, R_next, NI_next))
    
#     # Apply np.maximum only to the first 5 columns
#     X_next[:, :5] = np.maximum(X_next[:, :5], 1e-5)
    
#     return X_next



# def log_c_const(d, v):
#     """
#     Compute the log of the constant c(d, v) from Ghurye & Olkin (1969).
#     """

#     log_num = -0.5 * d * v * np.log(2) - 0.25 * d * (d - 1) * np.log(np.pi)
#     log_denom = np.sum([gammaln(0.5 * (v - i + 1)) for i in range(1, d + 1)])
    
#     return log_num - log_denom



# def unbiased_logpdf(y, mu_N, Sigma_N, N):
#     """
#     Unbiased estimator of the log Gaussian density from Ghurye & Olkin (1969).
    
#     Parameters:
#     - y: observed vector
#     - mu_N: sample mean vector
#     - Sigma_N: sample covariance matrix (divided by N-1)
#     - N: sample size
    
#     Returns:
#     - An unbiased estimate of the log Gaussian density, or -np.inf if invalid.
    
#     Requirements:
#     - N > d + 3
#     """
#     y = np.atleast_1d(y)
#     mu_N = np.atleast_1d(mu_N)
#     d = len(y)

#     if N <= d + 3:
#         raise ValueError("Not enough samples for valid estimator")   

#     M_N = (N - 1) * Sigma_N
#     diff = y - mu_N
#     denom = 1 - 1 / N

#     try:
#         inner = M_N - np.outer(diff, diff) / denom
#         # Check positive definiteness of inner matrix
#         eigvals_inner = np.linalg.eigvalsh(inner)
#         eigvals_M = np.linalg.eigvalsh(M_N)
#         if np.any(eigvals_inner <= 0) or np.any(eigvals_M <= 0):
#             return -np.inf

#         # Evaluate log terms safely
#         log_term = (
#             -d / 2 * np.log(2 * np.pi)
#             + log_c_const(d, N-2)
#             -log_c_const(d, N-1)
#             - (d / 2) * np.log(denom)
#             - ((N - d - 2) / 2) * np.log(det(M_N))
#             + ((N - d - 3) / 2) * np.log(det(inner))
#         )
#         return log_term

#     except LinAlgError:
#         return -np.inf


# def enkf_step(x_pred, state_hist, obs, theta, theta_names, t, dt=1):
#     N, D = x_pred.shape

#     # Forecast step
    
#     param = dict(zip(theta_names, theta))

#     # Observation operator
#     H = np.array([[0, 0, 0, 0, 1]])  # Shape: (1, D)

#     # Project forecast ensemble to observation space
#     y_pred = x_pred @ H.T  # Shape: (N, 1)
#     y_mean = np.mean(y_pred, axis=0)   # Shape: (1,)
#     V_t = np.maximum(0.1, y_mean)      # Poisson-inspired variance proxy

#     y_obs = obs + np.random.normal(0, np.sqrt(V_t), size=(N, 1))  # Simulated noisy obs

#     x_mean = np.mean(x_pred, axis=0)   # Shape: (D,)
#     dx = x_pred - x_mean               # Shape: (N, D)
#     dy = y_pred - y_mean               # Shape: (N, 1)

#     Pxy = dx.T @ dy / (N - 1)          # Shape: (D, 1)
#     Pyy = (dy.T @ dy) / (N - 1) + V_t  # Shape: (1, 1)

#     # Kalman gain
#     K = Pxy @ np.linalg.inv(Pyy)

#     # Update step
#     innovation = y_obs - y_pred        # Shape: (N, 1)
#     x_next = x_pred + innovation @ K.T

#     # Ensure non-negative compartments
#     x_next[:, :5] = np.maximum(x_next[:, :5], 1e-5)

#     # === Unbiased log-likelihood computation ===
#     mu_N = y_mean
#     Sigma_N = Pyy  # This is an empirical covariance + noise term
#     LIK = unbiased_logpdf(obs, mu_N, Sigma_N, N)
#     # LIK = norm.logpdf(obs, loc=y_mean.squeeze(), scale=np.sqrt(Pyy.squeeze()))

#     return x_next, LIK