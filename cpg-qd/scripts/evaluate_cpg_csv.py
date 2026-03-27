import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Type 1 font
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import time

from cpg_convergence.experiment_utils.cpg import aggregate_convergence_csv, create_interactive_convergence_plot, \
    pareto_mask, pareto_mask_fast, plot_induced_norm_vs_spectral_gap, plot_weight_vs_norms, \
    plot_scatter_steps_conv_pxx_vs_log_induced_norm, show_fully_converged_partition, \
    get_impossible_morphologies, plot_coupling_density_vs_norms, plot_couplings_vs_weights, \
    plot_norms_vs_density_and_weight, plot_spearman_and_log_pearson, plot_sgap_vs_ind_norm, \
    get_sgap_ind_norm_convergence_ranges, plot_sgap_vs_ind_norm_with_convergence_square, \
    plot_density_vs_sgap_over_ind_norm, summarize_successful_morphologies_by_method
exp_id = "bxx_ryy" # enter existing exp_id here, e.g. "b04_r01"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "experiments", exp_id, "numerical_metrics", "convergence_results.csv")
CSV_PATH_AGGREGATED = os.path.join(BASE_DIR, "experiments", exp_id, "numerical_metrics", "convergence_results_aggregated.csv")

# aggregate_convergence_csv(CSV_PATH) # Uncomment if aggregated file does not yet exist
df = pd.read_csv(CSV_PATH_AGGREGATED)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# df['n_oscillators'] = df['n_oscillators'] + df['ring_size']

print(f"n_oscillators range: {df['n_oscillators'].min()} - {df['n_oscillators'].max()}")
print(f"n_couplings range: {df['n_couplings'].min()} - {df['n_couplings'].max()}")

# print(df['n_oscillators'].value_counts().sort_index())
print(f"Number of unique n_oscillators values: {df['n_oscillators'].nunique()}")



# ==============================
# # Filter for specific experimental conditions
# df = pd.read_csv(CSV_PATH)
# df_filtered = df[(df['morphology_type'] == "equal_arms") & 
#                  (df['morphology_parameter'] == 11) & 
#                  (df['weight_coupling'] == 50)]

# # Group by method and create plots for each
# for method in df_filtered['methodology'].unique():
#     df_method = df_filtered[df_filtered['methodology'] == method]
#     plt.scatter(df_method['steps_to_convergence'], df_method['fraction_not_converged'], alpha=0.5, label=method)

# plt.xlabel('Index')
# plt.ylabel('Convergence')
# plt.legend()
# plt.show()


# # ==============================
# # # Compute and visualize Pareto front
# df_aggregated = pd.read_csv(CSV_PATH_AGGREGATED)
# # mean_steps = df_aggregated['std_steps_to_convergence'].mean()
# mean_fraction = df_aggregated['std_fraction_not_converged'].mean()
# # print(f"Mean std_steps_to_convergence: {mean_steps}")
# print(f"Mean std_fraction_not_converged: {mean_fraction}")

# # Create 2x2 subplot grid for different percentiles
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle('Pareto Fronts for Different Step Convergence Percentiles', fontsize=16, fontweight='bold')

# percentiles = ['p50', 'p75', 'p90', 'p100']
# percentile_names = ['50th percentile', '75th percentile', '90th percentile', '100th percentile']

# for idx, (percentile, percentile_name) in enumerate(zip(percentiles, percentile_names)):
#     row = idx // 2
#     col = idx % 2
#     ax = axes[row, col]
    
#     # Column names
#     x_col = f'mean_step_conv_{percentile}'
#     y_col = 'mean_fraction_not_converged'
    
#     # Compute Pareto front for this percentile
#     points = df_aggregated[[x_col, y_col]].values
#     pareto_front = pareto_mask_fast(points)
#     df_pareto = df_aggregated[pareto_front]
    
#     print(f"\nPareto front size for {percentile}: {len(df_pareto)}")
    
#     # Plot all points
#     ax.scatter(df_aggregated[x_col], df_aggregated[y_col], 
#                alpha=0.5, s=50, color='blue', label='All points')
    
#     # Plot Pareto front
#     ax.scatter(df_pareto[x_col], df_pareto[y_col], 
#                color='red', s=150, marker='*', 
#                edgecolors='gold', linewidths=2,
#                label='Pareto front', zorder=5)
    
#     # Set labels and title
#     ax.set_xlabel(f'Mean Step Convergence ({percentile_name})', fontsize=11)
#     ax.set_ylabel('Mean Fraction Not Converged', fontsize=11)
#     ax.set_title(f'Step Convergence {percentile_name}', fontsize=12, fontweight='bold')
#     ax.legend(loc='best')
#     ax.grid(True, alpha=0.3)
    
#     # Set consistent axis ranges
#     ax.set_xlim(0, 200)
#     ax.set_ylim(0, 1.0)

# plt.tight_layout()
# plt.show()


# ==============================
# # Analyze spectral gap and induced norm values
# plot_induced_norm_vs_spectral_gap(df)

# plot_weight_vs_norms(df)

# plot_scatter_steps_conv_pxx_vs_log_induced_norm(df, x_col="mean_step_conv_p90",)


# show_fully_converged_partition(df)

# print(get_impossible_morphologies(df))

# # methods = None
# # methods = ["base", "cobweb", "fully_connected", "leader_follower"]
# methods = ["base", "cobweb", "fully_connected"]
# # methods = ["fully_connected"]
# plot_coupling_density_vs_norms(df, methods=methods)
# plot_couplings_vs_weights(df, methods=methods, normalize_x=True, normalize_y=False, alpha_unsuccessful=0.3)
# plot_norms_vs_density_and_weight(df, methods=methods)

# spearman_corr, pearson_log_corr = plot_spearman_and_log_pearson(df, methods=methods)
# print("Spearman columns:", spearman_corr.columns.tolist())
# print("Pearson (log) columns:", pearson_log_corr.columns.tolist())

# plot_sgap_vs_ind_norm(df)

# start_time_range = time.time()
# range_dict = get_sgap_ind_norm_convergence_ranges(df, atol=1e-6)
# print(f"Time to compute convergence ranges: {time.time() - start_time_range:.2f} seconds")
# print(f"Spectral gap convergence range: median - {range_dict['median']['sgap']}, mean - {range_dict['mean']['sgap']}")
# print(f"Induced norm convergence range: median - {range_dict['median']['ind_norm']}, mean - {range_dict['mean']['ind_norm']}")
# # store range_dict as pickle
# import pickle
# with open(os.path.join(BASE_DIR, "experiments", exp_id, "numerical_metrics", "sgap_ind_norm_convergence_ranges.pkl"), "wb") as f:
#     pickle.dump(range_dict, f)


# load range_dict from pickle
import pickle
with open(os.path.join(BASE_DIR, "experiments", exp_id, "numerical_metrics", "sgap_ind_norm_convergence_ranges.pkl"), "rb") as f:
    range_dict = pickle.load(f)
plot_sgap_vs_ind_norm_with_convergence_square(df,
                                  ranges=range_dict,
                                  mean_enabled=False,
                                  median_enabled=False,
                                  combined_enabled=True,
                                  show=True,
                                  path=os.path.join(BASE_DIR, "experiments", exp_id, "plots", "sgap_vs_ind_norm_with_convergence_square_combined.png"),
                                  atol=1e-6,
                                  fontsize_title=16,
                                  fontsize_labels=14,
                                  fontsize_legend=12,
                                  fontsize_textbox=12,
                                  include_weight_arrow=True,
                                  enable_grid=False,
                                  )
sys.exit()
plot_density_vs_sgap_over_ind_norm(df, show=True, path=os.path.join(BASE_DIR, "experiments", exp_id, "plots", "density_vs_log_sgap_over_ind_norm.png"))



# percentage of successes per morphology
print(f"\nSmallest n_oscillators value: {df['n_oscillators'].min()}")
print(f"Largest n_oscillators value: {df['n_oscillators'].max()}")
print(f"Number of unique ring_size and arm_size pairs: {df.groupby(['ring_size', 'arm_size']).ngroups}")

print("\nSummary of successful morphologies by methodology:")
summary_df = summarize_successful_morphologies_by_method(df)
print(summary_df)




# # ==============================
# # Default path for testing
# create_interactive_convergence_plot(CSV_PATH_AGGREGATED, port=8050)
