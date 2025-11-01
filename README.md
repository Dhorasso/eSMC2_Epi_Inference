# eSMC²: Ensemble SMC² for Epidemiological Models

This repository contains the implementation of the **Ensemble SMC² (eSMC²)** algorithm for sequential Bayesian inference in state-space epidemiological models, as described in our paper:

"Accelerated Bayesian inference for state-space epidemiological models with Ensemble SMC²"

---

## Features
- Joint inference of hidden states and epidemiological parameters.
- Ensemble-based approximation (EnKF) for scalable likelihood evaluation.
- Supports discrete, count-based epidemic data.
- Demonstrated on synthetic datasets and 2022 U.S. monkeypox incidence data.

---

## Installation
Clone the repository:

```bash
git clone https://github.com/Dhorasso/eSMC2_Epi_Inference.git
cd eSMC2_Epi_Inference
```

## Example Usage


#####  Import necessary modules
```python
#####################################################################################
# Application of eSMC² for Experiment 1
# This script loads simulated data and applies the eSMC² algorithm.
#####################################################################################

# === Import necessary libraries ===
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm, uniform, norm
from smc2 import ESMC_squared  # your implementation
from smc_visualization import*
import matplotlib.pyplot as plt
```
##### Load your data

```python
#####################################################################################
# STEP 1: Load pre-simulated data
#####################################################################################

# Load data generated previously (columns: time, obs, beta_t)
true_theta = [1/2, 1/7]  # sigma, gamma
simulated_data = pd.read_csv("simulated_data1.csv")

## Visulation

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 4))

# --- Left axis: daily incidence (obs) ---
ax1.scatter(simulated_data['time'], simulated_data['obs'], color='dodgerblue', edgecolor='white', s=80)
ax1.set_xlabel('Days', fontsize=16)
ax1.set_ylabel('Daily incidence', color='dodgerblue', fontsize=16)
ax1.tick_params(axis='y', labelcolor='dodgerblue', labelsize=14)
# Increase x-axis tick label size
ax1.tick_params(axis='x', labelsize=14)  # change 18 to any fontsize you want

ax1.grid(True, alpha=0.3)

# --- Right axis: transmission rate (beta_t) ---
ax2 = ax1.twinx()
ax2.plot(simulated_data['time'], simulated_data['beta_t'], color='orange', lw=2)
ax2.set_ylabel(r'Transmission rate $\beta_t$', color='orange', fontsize=16)
ax2.tick_params(axis='y', labelcolor='orange', labelsize=14)

ax2.tick_params(axis='x',  labelsize=140)

# --- Optional: legends ---
# Combine legends from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

plt.tight_layout()
plt.show()
```


