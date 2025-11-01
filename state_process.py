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
