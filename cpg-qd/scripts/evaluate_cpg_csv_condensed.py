"For analysis of batch 04"
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from cpg_convergence.experiment_utils.cpg_condensed import aggregate_convergence_csv, \
    pareto_mask, pareto_mask_fast, plot_convergence_scatter, plot_scatter_steps_conv_pxx_vs_log_induced_norm, \
    create_interactive_convergence_plot, plot_scatter_log_spectral_gap_vs_log_induced_norm

exp_id = "bxx_ryy" # enter existing exp_id here, e.g. "b04_r01"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "experiments", exp_id, "numerical_metrics", "convergence_results.csv")
CSV_PATH_AGGREGATED = os.path.join(BASE_DIR, "experiments", exp_id, "numerical_metrics", "convergence_results_aggregated.csv")

# aggregate_convergence_csv(CSV_PATH) # Uncomment if aggregated file does not yet exist
df = pd.read_csv(CSV_PATH_AGGREGATED)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print(f"n_oscillators range: {df['n_oscillators'].min()} - {df['n_oscillators'].max()}")
print(f"n_couplings range: {df['n_couplings'].min()} - {df['n_couplings'].max()}")

# print(df['n_oscillators'].value_counts().sort_index())
print(f"Number of unique n_oscillators values: {df['n_oscillators'].nunique()}")

# ========== Plotting ===============

create_interactive_convergence_plot(
    CSV_PATH_AGGREGATED,
    port=8060
)

plot_convergence_scatter(
    df,
    ratio_selection=[1, "max"],
    # ratio_selection=["max"],
    # ratio_selection=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    only_converged=True,
    use_log_scale=False,
    show=False,
    atol_zero=1e-6,
    save_path=os.path.join(BASE_DIR, "experiments", exp_id, "plots", "convergence_scatter.png"),
)


plot_scatter_steps_conv_pxx_vs_log_induced_norm(
    df,
    x_col="mean_step_conv_p90",
    frac_not_conv_col="mean_fraction_not_converged",
    show=False,
    save_path=os.path.join(BASE_DIR, "experiments", exp_id, "plots", "scatter_step_conv_p90_vs_log_induced_norm.png")
    )


plot_scatter_log_spectral_gap_vs_log_induced_norm(
    df,
    ratio_selection=[1, "max"],
    only_converged=False,
    atol_zero=1e-6,
    show=True,
    save_path=os.path.join(BASE_DIR, "experiments", exp_id, "plots", "scatter_log_spectral_gap_vs_log_induced_norm.png")
    )
