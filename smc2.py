################################################################################
# eSMC² Algorithm Implementation
# - Supports Ensemble Kalman filter inner step
# - PMMH rejuvenation
################################################################################
import time

import numpy as np
import pandas as pd
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
from ssm_prior_draw import*
from state_process import state_transition
from scipy.stats import poisson
from Filter import EnK_Filter
from pmmh import PMMH_kernel
from resampling import resampling_style
from multiprocessing import Pool
import os, math






def ESMC_squared(
    initial_state_info, initial_theta_info, observed_data, num_state_particles,
    num_theta_particles, resampling_threshold=0.5, 
    pmmh_moves=5, c=0.5, n_jobs=10, resampling_method='stratified',
    real_time=False, esmc2_prevResults=None, forecast_days=0, show_progress=True
):
    """
    Run Ensemble SMC² for parameter and state estimation.

    Parameters
    ----------
    initial_state_info : dict
        Prior info and transformations for states.
    initial_theta_info : dict
        Prior info and transformations for parameters.
    observed_data : pd.DataFrame
        Observed dataset.
    num_state_particles : int
        Number of particles for inner filter.
    num_theta_particles : int
        Number of outer theta particles.
    resampling_threshold : float
        ESS threshold for resampling.
    pmmh_moves : int
        Number of PMMH rejuvenation moves.
    c : float
        Covariance scaling for PMMH proposals.
    n_jobs : int
        Number of parallel jobs.
    resampling_method : str
        Resampling method.
    real_time : bool
        Whether continuing from previous SMC² results.
    esmc2_prevResults : dict
        Previous eSMC² results.
    forecast_days : int
        Number of forecast steps beyond observed data.
    show_progress : bool
        Whether to show tqdm progress bar.

    Returns
    -------
    dict
        Contains log model evidence, trajectories, ESS, acceptance rates, and CPU time.
    """

    start_time = time.time()
    num_timesteps = len(observed_data)
    
    Z_arr = np.zeros((num_theta_particles, num_timesteps))
    likelihood_increment = np.ones(num_theta_particles)
    log_model_evid = np.zeros(num_timesteps)
    theta_weights = np.ones((num_theta_particles, num_timesteps)) / num_theta_particles
    ESS_theta = np.zeros(num_timesteps)
    acceptance_rate = np.zeros(num_timesteps)

    # Initialize theta and state particles
    initialization_theta = initial_theta(initial_theta_info, num_theta_particles)
    current_theta_particles = initialization_theta['currentThetaParticles']
    theta_names = initialization_theta['thetaName']
    
    initialization_state = initial_state(initial_state_info, num_theta_particles, num_state_particles)
    current_state_particles_all = initialization_state['currentStateParticles']
    state_names = initialization_state['stateName']

    total_cores = os.cpu_count()
    n_jobs = max(4, math.floor(total_cores * 0.75))
    
    if real_time and esmc2_prevResults is not None:
        current_theta_particles = esmc2_prevResults['current_theta_particles']
        current_state_particles_all = esmc2_prevResults['current_state_particles_all']

    traj_theta = [{key: [] for key in ['time'] + theta_names} for _ in range(num_theta_particles)]
    
    if show_progress:
        progress_bar = tqdm(total=num_timesteps, desc="SMC² Progress")

    for t in range(num_timesteps):
        t_start, t_end = (0, 0) if t == 0 else (t - 1, t)

        # Process all theta particles in parallel
        def process_particle_theta(m):
            trans_theta = current_theta_particles[m]
            theta = untransform_theta(trans_theta, initial_theta_info)
            current_state_particles = current_state_particles_all[m]

            x_corr, LIK = state_transition(
                theta, current_state_particles, state_names, theta_names, t_start, t_end, observed_data.iloc[t]
            )
            current_state_particles = x_corr
            likelihood_increment_theta = max(np.exp(LIK), 1e-12)
            return {'state_particles': current_state_particles, 'likelihood': likelihood_increment_theta, 'theta': trans_theta}

        particles_update_theta = Parallel(n_jobs=n_jobs)(
            delayed(process_particle_theta)(m) for m in range(num_theta_particles)
        )

        current_state_particles_all = np.array([p['state_particles'] for p in particles_update_theta])
        current_theta_particles = np.array([p['theta'] for p in particles_update_theta])
        likelihood_increment = np.array([p['likelihood'] for p in particles_update_theta])
        Z_arr[:, t] = np.log(likelihood_increment)
        Z = Z_arr[:, :t + 1]

        # Update weights and ESS
        theta_weights[:, t] = theta_weights[:, max(0, t - 1)] * likelihood_increment
        log_model_evid[t] = log_model_evid[max(0, t - 1)] + np.log(Evidence(theta_weights[:, max(0, t - 1)], likelihood_increment))
        theta_weights[:, t] /= np.sum(theta_weights[:, t])
        ESS_theta[t] = 1 / np.sum(theta_weights[:, t] ** 2)

        # Resampling and PMMH rejuvenation
        if ESS_theta[t] < resampling_threshold * num_theta_particles:
            resampled_indices_theta = resampling_style(theta_weights[:, t], resampling_method)
            Z = Z[resampled_indices_theta]
            theta_mean = np.average(current_theta_particles, axis=0, weights=theta_weights[:, t])
            theta_covariance = np.cov(current_theta_particles.T, ddof=0, aweights=theta_weights[:, t])
            theta_weights[:, t] = np.ones(num_theta_particles) / num_theta_particles
            current_theta_particles = current_theta_particles[resampled_indices_theta]
            current_state_particles_all = current_state_particles_all[resampled_indices_theta]

            # PMMH update in parallel
            new_particles = Parallel(n_jobs=n_jobs)(
                delayed(PMMH_kernel)(
                    Z, current_theta_particles, current_state_particles_all[m], theta_names,
                    observed_data.iloc[:t + 1], state_names, initial_theta_info, initial_state_info,
                    num_state_particles, theta_mean, theta_covariance, resampling_method, m,  
                    pmmh_moves, c, n_jobs
                ) for m in range(num_theta_particles)
            )

            current_theta_particles = np.array([new['theta'] for new in new_particles])
            current_state_particles_all = np.array([new['state'] for new in new_particles])
            acceptance_rate[t] = np.mean([new['acc'] for new in new_particles])
            Z = np.array([new['Z_m'] for new in new_particles])

        # Update theta trajectories
        traj_theta = Parallel(n_jobs=n_jobs)(
            delayed(lambda traj, j: pd.DataFrame(
                {'time': list(traj['time']) + [t],
                 **{name: list(traj[name]) + [untransform_theta(current_theta_particles[j], initial_theta_info)[i]]
                    for i, name in enumerate(theta_names)}}
            ))(traj, j) for j, traj in enumerate(traj_theta)
        )

        # Final particle filter step for last timestep
        if t == num_timesteps - 1:
            if real_time:
                current_state = esmc2_prevResults['current_state_particles']
            else:
                ini_state = initial_one_state(initial_state_info, num_state_particles)
                current_state = np.array(ini_state['currentStateParticles'])
            
            theta_mean = np.mean(current_theta_particles, axis=0)
            dists = np.linalg.norm(current_theta_particles - theta_mean, axis=1)
            idx = np.argsort(dists)[:100]
            theta_samples = current_theta_particles[idx, :]
            
            def run_pf_for_theta(theta_raw, theta_id):
                theta = untransform_theta(theta_raw, initial_theta_info)
                EnKF_results = EnK_Filter(
                    state_names, current_state, initial_state_info, theta, theta_names,
                    observed_data, num_state_particles, resampling_method,
                    forecast_days=forecast_days, add=1, end=True, n_jobs=1
                )
                traj_state = EnKF_results['traj_state']
                return [df.assign(theta_id=theta_id) for df in traj_state]

            all_traj_states = Parallel(n_jobs=n_jobs)(
                delayed(run_pf_for_theta)(theta_samples[i], i) for i in range(len(theta_samples))
            )
            traj_state_F = [traj for sublist in all_traj_states for traj in sublist]

        if show_progress:
            progress_bar.update(1)

    if show_progress:
        progress_bar.close()

    cpu_time = time.time() - start_time
    return {
        'log_modelevidence': log_model_evid,
        'margLogLike': log_model_evid[-1],
        'trajState': traj_state_F,
        'trajtheta': traj_theta,
        'ESS': ESS_theta,
        'acc': acceptance_rate,
        'cpu_time': cpu_time
    }

################################################################################
# Helper: Weighted evidence
################################################################################

def Evidence(theta_weights, like):
    """
    Compute weighted evidence at a given time step.
    """
    return np.average(like, weights=theta_weights)