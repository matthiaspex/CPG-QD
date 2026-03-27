"""Script to evaluate the csv results from the bs_cpg simulations,
specifically batch 02, and generate performance plots.
"""
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cpg_convergence.experiment_utils.bs_performance import plot_mean_fitness_over_generations, \
    plot_qd_metrics_over_generations, plot_mean_with_bootstrap_ci_over_generations, run_ttests, \
    plot_ci_t_test_paper

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

exp_ids = ["bxx_ryy", "bxx_ryy", "bxx_ryy"]  # enter existing exp_ids here, e.g. ["b04_r01", "b04_r02"]
save_path_prefix = "" # Enter a prefix for the saved plot paths, e.g. "batch04_comparison"


# qd_metrics_selection = ["qd_score", "coverage"]
qd_metrics_selection = ["qd_score", "coverage", "max_fitness"]


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATHS = [os.path.join(BASE_DIR, "experiments", exp_id, "numerical_metrics", "bs_simulation_results.csv")
             for exp_id in exp_ids]

SAVE_DIR = os.path.join(BASE_DIR, "results", save_path_prefix)


plot_mean_fitness_over_generations(CSV_PATHS, show=False, path=os.path.join(SAVE_DIR, "fitness.png"))
plot_qd_metrics_over_generations(CSV_PATHS, show=False,
                                 path=os.path.join(SAVE_DIR, "qd_metrics.png"))
plot_mean_with_bootstrap_ci_over_generations(CSV_PATHS,
                                             metrics="fitness",
                                             n_boot=1000,
                                             ci=95,
                                             show=False,
                                             path=os.path.join(SAVE_DIR, "fitness_bootstrap_ci.png"))

plot_mean_with_bootstrap_ci_over_generations(CSV_PATHS,
                                             metrics=qd_metrics_selection,
                                             n_boot=1000,
                                             ci=95,
                                             show=False,
                                             path=os.path.join(SAVE_DIR, "qd_metrics_bootstrap_ci.png"))


plot_ci_t_test_paper(CSV_PATHS,
                        metrics=["coverage", "qd_score"],
                        n_boot=1000,
                        ci=95,
                        show=False,
                        path=os.path.join(SAVE_DIR, "qd_metrics_ci_t_test_paper_biggest_font_slim.png"),
                        fontsize_title=32,   #28
                        fontsize_labels=28,  #24
                        fontsize_legend=24,  #20
                        fontsize_textbox=23, #20
                        fontsize_ticks=24,   #20
                        enable_grid=False,
                        disable_title=True,
                        increased_v_space=True,
                        dpi=600,
                        save_as_tif=False,
                        save_as_cmyk=False
                        )


run_ttests(CSV_PATHS, metrics=qd_metrics_selection, one_sided=True)



