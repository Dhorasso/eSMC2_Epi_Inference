#####################################################################################
# Application of eSMC² for mpox
# This script loads mpox data and applies the eSMC² algorithm.
#####################################################################################

# === Import necessary libraries ===
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm, uniform, norm
from smc2 import ESMC_squared 
import matplotlib.pyplot as plt
# Load data
monkeypox = pd.read_csv('monkeypox_US.csv')

# Rename and prepare columns
monkeypox.rename(columns={'Epi_date_v3': 'Date'}, inplace=True)
monkeypox['Date'] = pd.to_datetime(monkeypox['Date'])

# Filter to include data up to Dec 8, 2022
monkeypox = monkeypox[monkeypox['Date'] <= '2022-12-08']

plt.figure(figsize=(10, 4))
plt.plot(monkeypox['Date'], monkeypox['Cases'], color='dodgerblue', lw=2)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Daily Confirmed Cases', fontsize=14)
plt.title('U.S. Monkeypox (Mpox) Daily Cases', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




#####################################################################################
# STEP 2: Define priors for states and parameters
#####################################################################################

# --- Initial state priors (state_info) ---
state_info = {
   'S': {'prior': truncnorm((0-329999990)/1e-1, (330000000-329999990)/1e-1, loc=329999990, scale=1e-1), 'transf': 'none'},
    'E': {'prior': uniform(loc=0, scale=0), 'transf': 'none'},
    'I': {'prior': truncnorm((1-10)/1e-1, (15-10)/1e-1, loc=10, scale=1e-1), 'transf': 'none'},
    'R': {'prior': uniform(loc=0, scale=0), 'transf': 'none'},
    'Z': {'prior': uniform(loc=1, scale=1), 'transf': 'none'},
    'B': {'prior': uniform(loc=0.2, scale=0.3-0.2), 'transf': 'log'},
}

# --- Parameter priors (theta_info) ---
theta_info = {
    'alpha': {'prior': truncnorm((1/21-1/7)/0.05, (1/3-1/7)/0.05, loc=1/7, scale=0.05), 'transf': 'none'},
    'gamma': {'prior': uniform(loc=1/28, scale=1/14-1/28), 'transf': 'none'},
    'nu_beta': {'prior': uniform(loc=0, scale=0.3), 'transf': 'none'},
    'phi': {'prior': uniform(loc=0, scale=0.05), 'transf': 'none'},
}

np.random.seed(123)

esmc2_results = ESMC_squared(
    initial_state_info=state_info,
    initial_theta_info=theta_info,
    observed_data=monkeypox['Cases'],
    num_state_particles=200,
    num_theta_particles=1000,
)


# Print the Marginal log-likelihood
print("Marginal log-likelihood:", esmc2_results['margLogLike'])



#####################################################################################
# STEP 4:   State visualization
#####################################################################################

from smc_visualization import*
import seaborn as sns


ci_levels = [50, 75, 90, 95]  # CI levels to plot
mean_color ='blue'
ci_color = 'gray'       
window = 1 
fontsize_axis = 15

# Extract particle trajectories
matrix_state = trace_smc(esmc2_results['trajState'])
matrix_theta = trace_smc(esmc2_results['trajtheta'])

# Compute effective reproduction number Rₜ
gamma = np.mean(matrix_theta['gamma'][:, -1])
matrix_state['Rt'] = matrix_state['B'] / gamma
fig, axs = plt.subplots(3, 1, figsize=(10, 4*3))

# New infections
plot_smc(matrix_state['Z'], Date=monkeypox['Date'], ax=axs[0], mean_color=mean_color, ci_color=ci_color, label='Mean', window=window, ci_levels=ci_levels)
axs[0].scatter(monkeypox['Date'], monkeypox['Cases'], color='orange', edgecolor='k', s=20, label='Observed')
axs[0].set_xlabel('Date', fontsize=fontsize_axis)
axs[0].set_ylabel('Daily cases', fontsize=fontsize_axis)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=16)
axs[0].legend()

# Transmission rate βₜ
plot_smc(matrix_state['B'], Date=monkeypox['Date'], ax=axs[1], mean_color=mean_color, ci_color=ci_color, label='Mean', window=window, ci_levels=ci_levels)
axs[1].set_xlabel('Date', fontsize=fontsize_axis)
axs[1].set_ylabel(r'Transmission rate $\beta_t$', fontsize=fontsize_axis)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=16)


# Effective reproduction number Rₜ
plot_smc(matrix_state['Rt'], ax=axs[2] , Date=monkeypox['Date'], mean_color=mean_color, ci_color=ci_color, label='Mean', window=window, ci_levels=ci_levels)
axs[2].axhline(1, color='r', linestyle='--', lw=2)
axs[2].set_xlabel('Date', fontsize=fontsize_axis)
axs[2].set_ylabel(r'Reproduction number $R_0(t)$', fontsize=fontsize_axis)
axs[2].tick_params(axis='x', labelsize=14)
axs[2].tick_params(axis='y', labelsize=16)


for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
plt.tight_layout()
plt.show()


#####################################################################################
# STEP 4:   Parameter visualization
#####################################################################################


import matplotlib.gridspec as gridspec
from collections import OrderedDict

# --- User settings ---

mean_color = 'blue'
ci_color = 'gray'
ci_levels = [50, 75, 90, 95]
fontsize_axis = 20
fontsize_label = 25
fontsize_legend = 20

theta_labels = {
    'alpha': r'$\alpha$',
    'gamma': r'$\gamma$',
    'nu_beta': r'$\nu_{\beta}$',
    'phi': r'$\phi$',
}

# Extract priors using LaTeX labels as keys
priors = {theta_labels[k]: v['prior'] for k, v in theta_info.items()}

# Lower and upper bounds of priors
prior_bounds = {key: priors[key].ppf(0.999) for key in priors.keys()}
prior_supports = {key: priors[key].support()[0] for key in priors.keys()}  # <-- min bound

N = len(matrix_theta)
fig = plt.figure(figsize=(14, 4 * N))
gs = gridspec.GridSpec(N, 2, width_ratios=[3.7, 1.3])
handles, labels = [], []

axs = np.empty((N, 2), dtype=object)
for i in range(N):
    axs[i, 0] = fig.add_subplot(gs[i, 0])
    axs[i, 1] = fig.add_subplot(gs[i, 1])

for i, key in enumerate(matrix_theta.keys()):
    label_key = theta_labels[key]

    plot_smc(matrix_theta[key], ax=axs[i, 0], Date=monkeypox['Date'],
             mean_color=mean_color, ci_color=ci_color,
             label='eSMC²', ci_levels=ci_levels)

    axs[i, 0].set_xlabel('Date', fontsize=fontsize_axis)
    axs[i, 0].set_ylabel(label_key, fontsize=fontsize_label)
    axs[i, 0].tick_params(axis='x', labelsize=14)
    axs[i, 0].tick_params(axis='y', labelsize=16)

    if 'true_theta' in globals() and i < len(true_theta):
        axs[i, 0].axhline(y=true_theta[i], color='orange', linestyle='--', lw=3, label='Ground truth')

    # --- Posterior KDE ---
    data = matrix_theta[key][:, -1]
    sns.kdeplot(data, ax=axs[i, 1], color=ci_color, lw=2, label='Posterior')

    if 'true_theta' in globals() and i < len(true_theta):
        axs[i, 1].axvline(true_theta[i], color='orange', linestyle='--', lw=3, label='Ground truth')

    # --- Prior PDF ---
    x_min = prior_supports[label_key]
    x_max = max(data.max(), prior_bounds[label_key])
    x_vals = np.linspace(x_min, x_max, 500)
    axs[i, 1].plot(x_vals, priors[label_key].pdf(x_vals),
                   color='green', lw=3, linestyle='--', label='Prior')

    axs[i, 1].set_xlabel(label_key, fontsize=fontsize_label)
    axs[i, 1].set_ylabel('Density', fontsize=fontsize_axis)
    axs[i, 1].tick_params(axis='x', labelsize=14)
    axs[i, 1].tick_params(axis='y', labelsize=16)

    for ax in [axs[i, 0], axs[i, 1]]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.2)

    h, l = axs[i, 1].get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='lower center',
           fontsize=fontsize_legend, ncol=4, frameon=False,
           markerscale=3, handlelength=3, handletextpad=0.8)

plt.tight_layout(h_pad=2, w_pad=3, rect=[0, 0.05, 1, 1])
plt.show()

