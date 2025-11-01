################################################################################
# Inner Ensemble Kalman Filter (EnK_Filter)
################################################################################

import numpy as np
import pandas as pd
import gc
from state_process import*
from resampling import resampling_style
from ssm_prior_draw import  untransform_theta
from joblib import Parallel, delayed

def EnK_Filter(state_names, current_state_particles, initial_state_info, 
               theta, theta_names, observed_data, num_state_particles, 
               resampling_method, add=0, end=False, forecast_days=0, n_jobs=10):
    """
    Inner Ensemble Kalman Filter (EnKF) for state estimation.

    Parameters
    ----------
    state_names : list
        Names of the state variables.
    current_state_particles : ndarray
        Current ensemble of state particles.
    initial_state_info : dict
        Dictionary of initial state priors and transformations.
    theta : ndarray
        Model parameters.
    theta_names : list
        Names of model parameters.
    observed_data : pd.DataFrame
        Observed data to update against.
    num_state_particles : int
        Number of particles in the ensemble.
    resampling_method : callable
        Function to resample particles.
    add : int, optional
        Additional offset for time indexing.
    end : bool, optional
        Whether to store trajectories.
    forecast_days : int, optional
        Number of forecast steps beyond observed data.
    n_jobs : int, optional
        Number of parallel jobs for trajectory storage.

    Returns
    -------
    dict
        Dictionary containing:
            - 'incLogLike': incremental log-likelihood
            - 'particle_state': updated particles
            - 'traj_state': trajectory of states (if end=True)
    """

    num_timesteps = len(observed_data)
    traj_state = [{key: [] for key in ['time'] + state_names} for _ in range(num_state_particles)]
    inc_log_likelihood = np.zeros(num_timesteps)

    for t in range(num_timesteps + forecast_days):
        t_start, t_end = (0, 0) if t == 0 else (t - 1, t)

        # Forecast / update
        if t < num_timesteps:
            x_corr, LIK = state_transition(
                theta, current_state_particles, state_names, theta_names, 
                t_start, t_end, observed_data[min(t, num_timesteps-1)]
            )
        else:
            x_corr = Forecast_step(current_state_particles, theta, theta_names, forecast=True)

        current_state_particles = x_corr

        # Compute log-likelihood increment
        if t < num_timesteps:
            zt = max(np.exp(LIK), 1e-12)
            inc_log_likelihood[t] = np.log(zt)

        # Store trajectories at the end
        if end:
            traj_state = Parallel(n_jobs=n_jobs)(
                delayed(lambda traj, j: pd.DataFrame(
                    {'time': list(traj['time']) + [t],
                     **{name: list(traj[name]) + [untransform_theta(current_state_particles[j], initial_state_info)[i]] 
                        for i, name in enumerate(state_names)}}
                ))(traj, j) 
                for j, traj in enumerate(traj_state)
            )

    gc.collect()

    return {
        'incLogLike': inc_log_likelihood,
        'particle_state': current_state_particles,
        'traj_state': traj_state
    }
