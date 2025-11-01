########################################################################################
# This script contains code to handle constraint prameters and draw initail state and 
# parameter particles
##########################################################################################


import numpy as np
from scipy.stats import norm, lognorm, truncnorm, gamma, invgamma


#################################################################################
###### Function to transform/ untransform constrain parametres #####################

def logit(x):
    return np.log(x / (1 - x))

def inv_logit(x):
    return 1 / (1 + np.exp(-x))

def transform_theta(theta, initial_theta_info):
    transformed_theta = np.zeros_like(theta)
    for i, (param, info) in enumerate(initial_theta_info.items()):
        trans = info.get('transf', 'none')
        if trans == 'log':
            transformed_theta[i] = np.log(theta[i])
        elif trans == 'logit':
            transformed_theta[i] = logit(theta[i])
        else:
            transformed_theta[i] = theta[i]
    return transformed_theta

def untransform_theta(theta, initial_theta_info):
    untransformed_theta = np.zeros_like(theta)
    for i, (param, info) in enumerate(initial_theta_info.items()):
        trans = info.get('transf', 'none')
        if trans == 'log':
            untransformed_theta[i] = np.exp(theta[i])
        elif trans == 'logit':
            untransformed_theta[i] = inv_logit(theta[i])
        else:
            untransformed_theta[i] = theta[i]
    return untransformed_theta


#################################################################################
###### Functions to draw initial state and parameter particles #################

def initial_one_state(initial_state_info, num_state_particles):
    state_names = list(initial_state_info.keys())
    current_state_particles = np.zeros((num_state_particles, len(state_names)))

    for i in range(num_state_particles):
        state_values = [initial_state_info[state]['prior'].rvs() for state in state_names]
        state_values = transform_theta(state_values, initial_state_info)
        current_state_particles[i] = state_values

    return {
        'currentStateParticles': current_state_particles,
        'stateName': state_names,
    }

def initial_state(initial_state_info, num_theta_particles, num_state_particles):
    state_names = list(initial_state_info.keys())
    current_state_particles_all = np.zeros((num_theta_particles, num_state_particles, len(state_names)))

    for j in range(num_theta_particles):
        for i in range(num_state_particles):
            state_values = [initial_state_info[state]['prior'].rvs() for state in state_names]
            state_values = transform_theta(state_values, initial_state_info)
            current_state_particles_all[j, i, :] = state_values

    return {
        'currentStateParticles': current_state_particles_all,
        'stateName': state_names,
    }

def initial_theta(initial_theta_info, num_theta_particles):
    theta_names = list(initial_theta_info.keys())
    current_theta_particles = np.zeros((num_theta_particles, len(theta_names)))

    for i in range(num_theta_particles):
        theta_values = [initial_theta_info[param]['prior'].rvs() for param in theta_names]
        theta_values = transform_theta(theta_values, initial_theta_info)
        current_theta_particles[i] = theta_values

    return {
        'currentThetaParticles': current_theta_particles,
        'thetaName': theta_names,
    }