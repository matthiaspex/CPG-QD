"""Script to evaluate the csv results from the bs_cpg simulations,
specifically batch 03, and analyze variability.
"""
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from cpg_convergence.experiment_utils.bs_variability import aggregate_genotype_repetitions, \
    plot_std_distribution_grid, pairwise_variance_tests, load_variability_values, \
    test_variance_two_sets, plot_mean_distribution_grid
    


exp_ids = ["bxx_ryy", "bxx_ryy", "bxx_ryy"]  # enter existing exp_ids here, e.g. ["b04_r01", "b04_r02"]
save_path_prefix = "" # Enter a prefix for the saved plot paths, e.g. "batch04_comparison"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATHS = [os.path.join(BASE_DIR, "experiments", exp_id, "numerical_metrics", "bs_simulation_results.csv")
             for exp_id in exp_ids]
AGGREGATED_PATHS = None  # optional: provide precomputed aggregated paths here

TMP_DIR = os.path.join(BASE_DIR, "tmp")



AGGREGATED_PATHS = aggregate_genotype_repetitions(CSV_PATHS)
print("\nAggregation complete. Generated files:")
for path in AGGREGATED_PATHS:
    print(f"  {path}")



if AGGREGATED_PATHS is None:
    AGGREGATED_PATHS = [path.replace(".csv", "_aggregated.csv") for path in CSV_PATHS]


plot_std_distribution_grid(
    aggregated_paths=AGGREGATED_PATHS,
    normalize=False,
    bins=50,
    min_generation=50,
    max_generation=None,
    show=False,
    path=os.path.join(TMP_DIR, save_path_prefix, "std_distribution_grid.png")
)

plot_std_distribution_grid(
    aggregated_paths=AGGREGATED_PATHS,
    normalize=True,
    bins=50,
    min_generation=50,
    max_generation=None,
    show=False,
    path=os.path.join(TMP_DIR, save_path_prefix, "cv_distribution_grid.png")
)

plot_mean_distribution_grid(
    aggregated_paths=AGGREGATED_PATHS,
    bins=50,
    min_generation=50,
    max_generation=None,
    show=False,
    path=os.path.join(TMP_DIR, save_path_prefix, "mean_distribution_grid.png")
)

# # Compare variance of fitness std (or CV) between first two experiments
# a = load_variability_values(AGGREGATED_PATHS[0], "fitness", normalize=True)
# b = load_variability_values(AGGREGATED_PATHS[1], "fitness", normalize=True)
# print(test_variance_two_sets(a, b, center="median"))  # Levene recommended

# # Pairwise across all paths for 'fitness'
# res_df = pairwise_variance_tests(AGGREGATED_PATHS, metric="fitness", normalize=True)
# print(res_df)


print("Evaluation complete.")