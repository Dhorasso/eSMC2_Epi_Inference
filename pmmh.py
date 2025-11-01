################################################################################
# Particle Marginal Metropolis-Hastings (PMMH) kernel and log-prior functions
################################################################################

import numpy as np
from scipy.stats import multivariate_normal
from ssm_prior_draw import *
from Filter import EnK_Filter

################################################################################
# PMMH Kernel
################################################################################

def PMMH_kernel(Z, current_theta_particles, current_state_particles, theta_names,
                observed_data, state_names, initial_theta_info, initial_state_info,
                num_state_particles, theta_mean_current, theta_covariance_current,
                resampling_method, m, pmmh_moves, c, n_jobs=10):
    """
    Perform Particle Marginal Metropolis-Hastings (PMMH) update for a given particle.

    Parameters
    ----------
    Z : ndarray
        Incremental log likelihood matrix [num_particles, num_timesteps].
    current_theta_particles : ndarray
        Current particles of theta parameters.
    current_state_particles : ndarray
        Current state particles.
    theta_names : list
        Names of theta parameters.
    observed_data : pd.DataFrame
        Observed data.
    state_names : list
        Names of state variables.
    initial_theta_info : dict
        Prior info and transformations for theta parameters.
    initial_state_info : dict
        Prior info and transformations for state variables.
    num_state_particles : int
        Number of state particles in the particle filter.
    theta_mean_current : ndarray
        Current mean of theta particles.
    theta_covariance_current : ndarray
        Current covariance of theta particles.
    resampling_method : callable
        Resampling function for particle filter.
    m : int
        Index of the current theta particle.
    pmmh_moves : int
        Number of PMMH proposal moves.
    c : float
        Scaling factor for covariance in proposal.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    dict
        Updated particle info: 'Z_m', 'log_prior_theta', 'state', 'theta', 'acc'.
    """

    acc = 0
    I = 1e-5 * np.eye(theta_covariance_current.shape[0])
    theta_covariance_current = c * theta_covariance_current + I

    state_current = current_state_particles
    theta_current = current_theta_particles[m]

    log_prior_current = log_prior(initial_theta_info, theta_current)
    Z_current = np.sum(Z, axis=1)[m]
    Z_m_current = Z[m, :]

    for i in range(pmmh_moves):
        # Propose new theta
        if theta_mean_current.shape[0] == 1:
            theta_proposal = np.random.normal(theta_mean_current, np.sqrt(theta_covariance_current[0, 0]))
        else:
            theta_proposal = np.random.multivariate_normal(theta_mean_current, theta_covariance_current)

        log_prior_proposal = log_prior(initial_theta_info, theta_proposal)

        if np.isfinite(log_prior_proposal):
            # Evaluate log-posterior
            current = Z_current + log_prior_current

            # Particle filter for proposal
            ini_state = initial_one_state(initial_state_info, num_state_particles)
            ini_current_state = np.array(ini_state['currentStateParticles'])
            untrans_theta_proposal = untransform_theta(theta_proposal, initial_theta_info)

            EnKF_results = EnK_Filter(
                state_names, ini_current_state, initial_state_info, untrans_theta_proposal,
                theta_names, observed_data, num_state_particles, resampling_method, n_jobs=n_jobs
            )

            Z_m_proposal = EnKF_results['incLogLike']
            Z_proposal = np.sum(Z_m_proposal)
            state_proposal = EnKF_results['particle_state']
            proposal = Z_proposal + log_prior_proposal

            # Include multivariate normal log-density for MH step
            proposal += log_multivariate_normal_pdf(theta_current, theta_mean_current, theta_covariance_current)
            current += log_multivariate_normal_pdf(theta_proposal, theta_mean_current, theta_covariance_current)

            # Acceptance probability
            ratio = proposal - current
            alpha = np.exp(ratio)

            if np.isfinite(alpha) and np.random.uniform() < min(1, alpha):
                Z_current = Z_proposal
                Z_m_current = Z_m_proposal
                state_current = state_proposal
                theta_current = theta_proposal
                log_prior_current = log_prior_proposal
                acc += 1

    return {
        'Z_m': Z_m_current,
        'log_prior_theta': log_prior_current,
        'state': state_current,
        'theta': theta_current,
        'acc': acc / pmmh_moves
    }

################################################################################
# Log multivariate normal PDF (without normalization constant)
################################################################################

def log_multivariate_normal_pdf(x, mean, cov):
    diff = x - mean
    if mean.shape[0] == 1:
        cov_inv = 1 / cov[0, 0]
        return -0.5 * (diff ** 2 * cov_inv)
    else:
        cov_inv = np.linalg.inv(cov)
        return -0.5 * np.dot(diff.T, np.dot(cov_inv, diff))

################################################################################
# Log prior with optional transformations and Jacobian adjustment
################################################################################

def log_prior(initial_theta_info, theta):
    total_log_prior = 0

    for i, (param_name, info) in enumerate(initial_theta_info.items()):
        value = theta[i]
        transf = info.get('transf', 'none')
        dist = info['prior']

        if transf == 'log':
            theta_original = np.exp(value)
            # jacobian_adjustment = value
        elif transf == 'logit':
            theta_original = 1 / (1 + np.exp(-value))
            # jacobian_adjustment = np.log(theta_original) + np.log(1 - theta_original)
        else:
            theta_original = value

        log_prior_val = dist.logpdf(theta_original)
        total_log_prior += log_prior_val  # + jacobian_adjustment

    return total_log_prior
