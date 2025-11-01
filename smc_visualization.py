import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DayLocator, DateFormatter
from matplotlib.ticker import AutoMinorLocator

def trace_smc(Traject):
    """
    Convert SMC² trajectories into a dictionary of matrices for each state.

    Parameters:
    - Traject: list of pd.DataFrame
        Trajectories obtained from SMC_squared

    Returns:
    - dict: keys = state names, values = matrix (particles x time)
    """
    matrix_dict = {}
    state_names = list(Traject[0].columns[1:])
    
    for state in state_names:
        state_matrices = [df[state].values.reshape(1, -1) for df in Traject]
        combined_matrix = np.concatenate(state_matrices, axis=1)
        reshaped_matrix = combined_matrix.reshape(-1, Traject[0].shape[0])
        matrix_dict[state] = reshaped_matrix
    
    return matrix_dict




def plot_smc(matrix, ax, mean_color='b', ci_color='dodgerblue', label='Mean', Date=None, window=1, ci_levels=[95]):
    """
    Plot SMC² results with mean and user-specified credible intervals.

    Parameters
    ----------
    matrix : np.ndarray or list
        Matrix of particles x time for one state.
    ax : matplotlib.axes.Axes
        Axes object to draw on.
    color : str
        Color for mean line and CI shading.
    label : str
        Label for mean line.
    Date : array-like
        Optional x-axis values.
    window : int
        Rolling window for smoothing.
    ci_levels : list of int
        List of credible interval levels to display (e.g., [50, 75, 90, 95]).
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    # Compute mean
    mean = pd.Series(np.nanmean(matrix, axis=0)).rolling(window=window, min_periods=1).mean().values

    # X-axis
    x = Date if Date is not None else np.arange(matrix.shape[1])

    # Plot each CI
    for ci in sorted(ci_levels, reverse=True):
        lower = np.nanpercentile(matrix, 50 - ci/2, axis=0)
        upper = np.nanpercentile(matrix, 50 + ci/2, axis=0)
        ax.fill_between(x, lower, upper, color=ci_color, alpha=1 * (1-ci / 100)+0.08) 



    # Plot mean
    ax.plot(x, mean, color=mean_color, lw=3, label=label)

    # Aesthetics
    ax.grid(True, alpha=0.4)
    if Date is not None:
        ax.minorticks_on()
        ax.xaxis.set_major_locator(MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter('%b %y'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(DayLocator(bymonthday=16))
        ax.grid(True, which='minor', linestyle='--', linewidth=0.4)
