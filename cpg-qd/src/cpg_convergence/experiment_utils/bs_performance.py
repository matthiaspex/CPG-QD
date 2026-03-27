import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy.stats import ttest_ind

def plot_mean_fitness_over_generations(csv_paths, show=False, path=None):
    """
    For each CSV in csv_paths:
      - compute mean fitness per (run_id, generation) across genotype entries
      - plot generation vs. mean fitness for each run_id
      - use one color per CSV file; all runs from that CSV share the color (alpha=0.7)
      - legend shows exp_id, method, ring_size, arm_size, weight_coupling (assumed constant per CSV)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.colormaps.get_cmap("tab10")

    # Collect exp_ids for the title
    exp_ids_list = []

    for i, csv_path in enumerate(csv_paths):
        if not os.path.isfile(csv_path):
            print(f"Warning: missing CSV '{csv_path}', skipping.")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: empty CSV '{csv_path}', skipping.")
            continue

        required = {"run_id", "generation", "fitness"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        agg = (
            df.groupby(["run_id", "generation"], as_index=False)["fitness"]
              .mean()
              .sort_values(["run_id", "generation"])
        )

        meta_fields = ["method", "ring_size", "arm_size", "weight_coupling"]
        meta = {}
        for field in meta_fields:
            if field in df.columns:
                unique_vals = df[field].dropna().unique()
                if len(unique_vals) != 1:
                    raise ValueError(f"{csv_path}: expected a single value for {field}, found {unique_vals}")
                meta[field] = unique_vals[0]

        # Derive exp_id from path or fallback to index
        exp_id = os.path.basename(os.path.dirname(os.path.dirname(csv_path))) or f"exp_{i}"
        exp_ids_list.append(exp_id)
        meta_label = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else os.path.basename(csv_path)
        label = f"{exp_id}: {meta_label}"

        color = cmap(i % cmap.N)
        run_ids = agg["run_id"].unique()
        for j, run_id in enumerate(run_ids):
            run_df = agg[agg["run_id"] == run_id]
            run_label = label if j == 0 else None  # one legend entry per CSV
            ax.plot(
                run_df["generation"],
                run_df["fitness"],
                color=color,
                alpha=0.7,
                label=run_label,
            )

    # Create title with all exp_ids
    title = "Relative Performance: " + " - ".join(exp_ids_list) if exp_ids_list else "Relative Performance"
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean fitness")
    ax.set_ylim(bottom=0)
    # Place legend below the x-axis (closer) and reserve top space for title
    legend = ax.legend(title="Configuration", loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(bottom=0.16, top=0.90)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_extra_artists=(legend,), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_qd_metrics_over_generations(csv_paths, show=False, path=None):
    """
    Like plot_mean_fitness_over_generations but plots mean per-generation:
      qd_score, coverage, max_fitness
    Produces 3 horizontal subplots (one per metric). One color per CSV; all runs
    from that CSV share the color (alpha=0.7). Legend shows exp_id and meta.
    """
    metrics = ["qd_score", "coverage", "max_fitness"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    if len(metrics) == 1:
        axes = [axes]
    cmap = plt.colormaps.get_cmap("tab10")

    exp_ids_list = []
    legend = None

    for i, csv_path in enumerate(csv_paths):
        if not os.path.isfile(csv_path):
            print(f"Warning: missing CSV '{csv_path}', skipping.")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: empty CSV '{csv_path}', skipping.")
            continue

        required_columns = {"run_id", "generation"} | set(metrics)
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        agg = (
            df.groupby(["run_id", "generation"], as_index=False)[metrics]
              .mean()
              .sort_values(["run_id", "generation"])
        )

        meta_fields = ["method", "ring_size", "arm_size", "weight_coupling"]
        meta = {}
        for field in meta_fields:
            if field in df.columns:
                unique_vals = df[field].dropna().unique()
                if len(unique_vals) != 1:
                    raise ValueError(f"{csv_path}: expected a single value for {field}, found {unique_vals}")
                meta[field] = unique_vals[0]

        exp_id = os.path.basename(os.path.dirname(os.path.dirname(csv_path))) or f"exp_{i}"
        exp_ids_list.append(exp_id)
        meta_label = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else os.path.basename(csv_path)
        label = f"{exp_id}: {meta_label}"

        color = cmap(i % cmap.N)
        run_ids = agg["run_id"].unique()
        for j, run_id in enumerate(run_ids):
            run_df = agg[agg["run_id"] == run_id]
            run_label = label if j == 0 else None  # one legend entry per CSV
            for ax_idx, metric in enumerate(metrics):
                ax = axes[ax_idx]
                ax.plot(
                    run_df["generation"],
                    run_df[metric],
                    color=color,
                    alpha=0.7,
                    label=run_label if ax_idx == 0 else None,  # add legend labels only on first subplot
                )

    # Titles and labels
    for ax_idx, metric in enumerate(metrics):
        axes[ax_idx].set_title(metric.replace("_", " ").title())
        axes[ax_idx].set_xlabel("Generation")
        axes[ax_idx].set_ylabel("Mean " + metric.replace("_", " "))
        axes[ax_idx].set_ylim(bottom=0)
        axes[ax_idx].grid(True, alpha=0.3)

    title = "Relative Performance: " + " - ".join(exp_ids_list) if exp_ids_list else "Relative Performance"
    fig.suptitle(title)

    # Create a single legend from the first axis (closer to the plots)
    legend = axes[0].legend(title="Configuration", loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.18, top=0.92)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if legend is not None:
            fig.savefig(path, dpi=150, bbox_extra_artists=(legend,), bbox_inches="tight")
        else:
            fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_mean_with_bootstrap_ci_over_generations(
    csv_paths,
    metrics=None,
    n_boot=1000,
    ci=95,
    show=False,
    path=None,
):
    """
    Plot mean +/- bootstrap 95% CI over generations for one or more metrics.

    Parameters
    ----------
    csv_paths : list[str]
        One CSV per experimental condition (one color per CSV).
    metrics : list[str] or None
        Metrics to plot. If None, defaults to ["qd_score","coverage","max_fitness"].
    n_boot : int
        Number of bootstrap resamples to compute CI (default 1000).
    ci : float
        Confidence level in percent (default 95).
    show : bool
        If True, call plt.show().
    path : str or None
        If provided, save the figure to this path.
    """
    # Normalize metrics argument: accept None, a single string, or an iterable
    if metrics is None:
        metrics = ["qd_score", "coverage", "max_fitness"]
    elif isinstance(metrics, str):
        metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    else:
        # ensure it's a list (e.g., tuple -> list)
        metrics = list(metrics)

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    cmap = plt.colormaps.get_cmap("tab10")

    exp_ids_list = []
    legend_handles = []
    legend_labels = []
    legend_info = []  # Store tuples of (color, exp_id, method, weight_coupling)

    alpha_pct = (100.0 - float(ci)) / 2.0
    lower_pct = alpha_pct
    upper_pct = 100.0 - alpha_pct

    y_range_max = np.zeros(len(metrics))
    for i, csv_path in enumerate(csv_paths):
        if not os.path.isfile(csv_path):
            print(f"Warning: missing CSV '{csv_path}', skipping.")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: empty CSV '{csv_path}', skipping.")
            continue

        required_columns = {"run_id", "generation"} | set(metrics)
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        # Reduce genotype-level entries by averaging per (run_id, generation)
        agg = (
            df.groupby(["run_id", "generation"], as_index=False)[metrics]
              .mean()
              .sort_values(["run_id", "generation"])
        )

        meta_fields = ["method", "ring_size", "arm_size", "weight_coupling"]
        meta = {}
        for field in meta_fields:
            if field in df.columns:
                unique_vals = df[field].dropna().unique()
                if len(unique_vals) != 1:
                    raise ValueError(f"{csv_path}: expected a single value for {field}, found {unique_vals}")
                meta[field] = unique_vals[0]

        exp_id = os.path.basename(os.path.dirname(os.path.dirname(csv_path))) or f"exp_{i}"
        exp_ids_list.append(exp_id)
        meta_label = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else os.path.basename(csv_path)
        label = f"{exp_id}: {meta_label}"

        color = cmap(i % cmap.N)

        # Unique sorted generations present in this file
        generations = np.sort(agg["generation"].unique())

        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx]

            means = []
            lowers = []
            uppers = []
            gens_for_plot = []

            for g in generations:
                vals = agg.loc[agg["generation"] == g, metric].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    # skip generation if no runs present for this metric
                    continue

                # observed mean across runs
                obs_mean = float(np.mean(vals))
                means.append(obs_mean)
                gens_for_plot.append(g)

                # bootstrap: sample runs with replacement and compute mean
                # if only one run, bootstrap will produce identical means
                idx = np.random.randint(0, vals.size, size=(n_boot, vals.size))
                sample_means = vals[idx].mean(axis=1)
                lower = np.percentile(sample_means, lower_pct)
                upper = np.percentile(sample_means, upper_pct)
                lowers.append(lower)
                uppers.append(upper)

            if len(gens_for_plot) == 0:
                ax.set_visible(False)
                continue

            gens_arr = np.array(gens_for_plot)
            means_arr = np.array(means)
            lowers_arr = np.array(lowers)
            uppers_arr = np.array(uppers)

            line_label = label if ax_idx == 0 else None
            ax.plot(gens_arr, means_arr, color=color, alpha=0.9, label=line_label)
            ax.fill_between(gens_arr, lowers_arr, uppers_arr, color=color, alpha=0.25)

            ax.set_xlabel("Generation")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_ylim(bottom=0)
            y_range_max[ax_idx] = max(y_range_max[ax_idx], np.max(uppers_arr) * 1.05)
            ax.set_ylim(top=y_range_max[ax_idx])
            ax.grid(True, alpha=0.3)

        # collect legend info for each experiment
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2))
        legend_labels.append(label)
        legend_info.append((color, exp_id, meta.get("method", "-"), meta.get("weight_coupling", "-")))

    title = "Relative Performance (mean ± CI): " + " - ".join(exp_ids_list) if exp_ids_list else "Relative Performance"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.22, top=0.85, wspace=0.3)
    # Add custom legend below the figure
    if legend_info:
        import matplotlib.patches as mpatches
        legend_patches = []
        legend_labels = []
        for color, exp_id, method, weight_coupling in legend_info:
            label = f"{exp_id} | method: {method} | weight_coupling: {weight_coupling}"
            patch = mpatches.Patch(color=color, label=label)
            legend_patches.append(patch)
            legend_labels.append(label)
        # Place the color legend below the figure
        fig.legend(
            handles=legend_patches,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.07),
            ncol=1 if len(legend_patches) < 3 else 2,
            fontsize=10,
            title="Experiment Legend",
            title_fontsize=11,
            frameon=False
        )
    # Ensure subtitles for every axis are displayed
    for ax_idx, metric in enumerate(metrics):
        axes[ax_idx].set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)




def plot_ci_t_test_paper(
    csv_paths,
    metrics=["coverage", "qd_score"],
    n_boot=1000,
    ci=95,
    show=False,
    path=None,
    fontsize_title: int = 12,
    fontsize_legend: int = 10,
    fontsize_labels: int = 11,
    fontsize_textbox: int = 10,
    fontsize_ticks: int = 10,
    enable_grid: bool = True,
    disable_title: bool = False,
    increased_v_space: bool = False,
    save_as_tif: bool = False,
    save_as_cmyk: bool = False,
    dpi: int = 300,
):
    """
    Plot mean +/- bootstrap 95% CI over generations for one or more metrics.

    Parameters
    ----------
    csv_paths : list[str]
        One CSV per experimental condition (one color per CSV).
    metrics : list[str] or None
        Metrics to plot. If None, defaults to ["qd_score","coverage","max_fitness"].
    n_boot : int
        Number of bootstrap resamples to compute CI (default 1000).
    ci : float
        Confidence level in percent (default 95).
    show : bool
        If True, call plt.show().
    path : str or None
        If provided, save the figure to this path.
    fontsize_title : int
        Font size for titles (default 12).
    fontsize_legend : int
        Font size for legend (default 10).
    fontsize_labels : int
        Font size for axis labels (default 11).
    fontsize_textbox : int
        Font size for textboxes (default 10).
    fontsize_ticks : int
        Font size for tick labels (default 10).
    enable_grid : bool
        Whether to enable grid (default True).
    disable_title : bool
        Whether to disable the title (default False).
    increased_v_space : bool
        Whether to increase vertical spacing (default False).
    save_as_tif : bool
        If True, save figure as TIF instead of PNG (default False).
    save_as_cmyk : bool
        If True, convert figure to CMYK color space before saving (default False).
    dpi : int
        DPI for saving the figure (default 300).
    """
    # Normalize metrics argument: accept None, a single string, or an iterable
    if metrics is None:
        metrics = ["qd_score", "coverage", "max_fitness"]
    elif isinstance(metrics, str):
        metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    else:
        # ensure it's a list (e.g., tuple -> list)
        metrics = list(metrics)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 7 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    # Define color mapping for CPG types
    cpg_color_map = {
        "Slow CPG": "orange",
        "Fast/Stable CPG": "green",
        "Unstable CPG": "red"
    }

    exp_ids_list = []
    legend_handles = []
    legend_labels = []
    legend_info = []  # Store tuples of (color, cpg_type)
    cpg_indices = {}  # Map cpg_type to indices in csv_paths

    alpha_pct = (100.0 - float(ci)) / 2.0
    lower_pct = alpha_pct
    upper_pct = 100.0 - alpha_pct

    y_range_max = np.zeros(len(metrics))
    for i, csv_path in enumerate(csv_paths):
        if not os.path.isfile(csv_path):
            print(f"Warning: missing CSV '{csv_path}', skipping.")
            continue
        
        pattern = r'b(\d+)_r(\d+)' # recognize "bxx_ryy" patterns
        match = re.search(pattern, csv_path)
        df = pd.read_csv(csv_path)
        run_index = int(match.group(2))
        if (run_index%10)%3 == 0:
            cpg_type = "Slow CPG"
        elif (run_index%10)%3 == 1:
            cpg_type = "Fast/Stable CPG"
        else:
            cpg_type = "Unstable CPG"
        
        # Track cpg_type to index mapping
        if cpg_type not in cpg_indices:
            cpg_indices[cpg_type] = []
        cpg_indices[cpg_type].append(i)
        
        # Assign color based on CPG type
        color = cpg_color_map.get(cpg_type, "gray")
        
        if df.empty:
            print(f"Warning: empty CSV '{csv_path}', skipping.")
            continue

        required_columns = {"run_id", "generation"} | set(metrics)
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        # Reduce genotype-level entries by averaging per (run_id, generation)
        agg = (
            df.groupby(["run_id", "generation"], as_index=False)[metrics]
              .mean()
              .sort_values(["run_id", "generation"])
        )

        meta_fields = ["method", "ring_size", "arm_size", "weight_coupling"]
        meta = {}
        for field in meta_fields:
            if field in df.columns:
                unique_vals = df[field].dropna().unique()
                if len(unique_vals) != 1:
                    raise ValueError(f"{csv_path}: expected a single value for {field}, found {unique_vals}")
                meta[field] = unique_vals[0]

        exp_id = os.path.basename(os.path.dirname(os.path.dirname(csv_path))) or f"exp_{i}"
        exp_ids_list.append(exp_id)
        meta_label = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else os.path.basename(csv_path)
        label = f"{exp_id}: {meta_label}"

        # Unique sorted generations present in this file
        generations = np.sort(agg["generation"].unique())

        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx]

            means = []
            lowers = []
            uppers = []
            gens_for_plot = []

            for g in generations:
                vals = agg.loc[agg["generation"] == g, metric].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    # skip generation if no runs present for this metric
                    continue

                # observed mean across runs
                obs_mean = float(np.mean(vals))
                means.append(obs_mean)
                gens_for_plot.append(g)

                # bootstrap: sample runs with replacement and compute mean
                # if only one run, bootstrap will produce identical means
                idx = np.random.randint(0, vals.size, size=(n_boot, vals.size))
                sample_means = vals[idx].mean(axis=1)
                lower = np.percentile(sample_means, lower_pct)
                upper = np.percentile(sample_means, upper_pct)
                lowers.append(lower)
                uppers.append(upper)

            if len(gens_for_plot) == 0:
                ax.set_visible(False)
                continue

            gens_arr = np.array(gens_for_plot)
            means_arr = np.array(means)
            lowers_arr = np.array(lowers)
            uppers_arr = np.array(uppers)

            line_label = label if ax_idx == 0 else None
            ax.plot(gens_arr, means_arr, color=color, alpha=0.9, label=line_label)
            ax.fill_between(gens_arr, lowers_arr, uppers_arr, color=color, alpha=0.25)

            ax.set_xlabel("Generation", fontsize=fontsize_labels)
            if metric == "qd_score":
                ylabel = "total archive fitness"
            elif metric == "coverage":
                ylabel = "percentage"
            else:
                ylabel = metric.replace("_", " ").title()
            ax.set_ylabel(ylabel, fontsize=fontsize_labels)
            ax.set_ylim(bottom=0)
            y_range_max[ax_idx] = max(y_range_max[ax_idx], np.max(uppers_arr) * 1.05)
            ax.set_ylim(top=y_range_max[ax_idx])
            ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
            if enable_grid:
                ax.grid(True, alpha=0.3)

        # collect legend info for each experiment
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2))
        legend_labels.append(label)
        legend_info.append((color, cpg_type))

    # Run t-tests to get p-values for textboxes
    p_values = {}
    if all(cpg_type in cpg_indices for cpg_type in ["Fast/Stable CPG", "Slow CPG", "Unstable CPG"]):
        # Get the indices for each CPG type
        fast_idx = cpg_indices.get("Fast/Stable CPG", [None])[0]
        slow_idx = cpg_indices.get("Slow CPG", [None])[0]
        unstable_idx = cpg_indices.get("Unstable CPG", [None])[0]
        
        # Create subset of csv_paths for t-tests
        test_paths = [csv_paths[idx] for idx in [fast_idx, slow_idx, unstable_idx] if idx is not None]
        
        # Load curves and compute AUC for t-tests
        all_aucs = []
        for csv_path in test_paths:
            curves = load_curves(csv_path, metrics)
            aucs, _ = compute_auc_and_final(curves, metrics)
            all_aucs.append(aucs)
        
        # Compute p-values for each metric
        for metric_idx, metric in enumerate(metrics):
            if len(all_aucs) >= 3:
                # all_aucs[0] = Fast/Stable, all_aucs[1] = Slow, all_aucs[2] = Unstable
                _, p_fast_vs_slow, _ = pairwise_ttest(all_aucs[0][metric], all_aucs[1][metric], alternative='greater')
                _, p_fast_vs_unstable, _ = pairwise_ttest(all_aucs[0][metric], all_aucs[2][metric], alternative='greater')
                p_values[metric] = (p_fast_vs_slow, p_fast_vs_unstable)

    title = "Relative Performance (mean ± CI): " + " - ".join(exp_ids_list) if exp_ids_list else "Relative Performance"
    if not disable_title:
        fig.suptitle(title, fontsize=fontsize_title, fontweight="bold")
    bottom = 0.16 if not increased_v_space else 0.29
    hspace = 0.5 if not increased_v_space else 1.05
    left = 0.08
    right = 0.95
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=0.95, hspace=hspace)
    
    # Add textboxes between subfigures with actual p-values
    for i in range(len(metrics)):
        # Calculate vertical position between subplots
        num_metrics = len(metrics)
        if i == 0:
            y_pos = 0.540 if not increased_v_space else 0.56# customly defined
        elif i == 1:
            y_pos = 0.068 if not increased_v_space else 0.12# customly defined
        else:
            y_pos = 0.5 - (i * 0.45)  # fallback calculation
        
        # Get p-values for this metric
        if metrics[i] in p_values:
            p_fast_vs_slow, p_fast_vs_unstable = p_values[metrics[i]]
            p_slow_str = f"{p_fast_vs_slow:.2e}"
            p_unstable_str = f"{p_fast_vs_unstable:.2e}"
        else:
            p_slow_str = "[placeholder]"
            p_unstable_str = "[placeholder]"
        
        textbox_content = (
            f"AUC(Fast/Stable CPG) > AUC(Slow CPG)\n"
            f"  ↳  (p = {p_slow_str})\n"
            f"AUC(Fast/Stable CPG) > AUC(Unstable CPG)\n"
            f"  ↳  (p = {p_unstable_str})"
        )
        
        fig.text(0, #left, # + (1-left)/2 - 0.045,
                 y_pos, textbox_content, ha='left', va='bottom',
                fontsize=fontsize_textbox,
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    
    # Add custom legend below the figure
    if legend_info:
        import matplotlib.patches as mpatches
        legend_patches = []
        legend_labels = []
        for color, cpg_type in legend_info:
            label = f"{cpg_type}"
            patch = mpatches.Patch(color=color, label=label)
            legend_patches.append(patch)
            legend_labels.append(label)
        # Place the color legend below the figure
        fig.legend(
            handles=legend_patches,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(left + (1-left)/2 - 0.045, -0.02),
            ncol=2,
            fontsize=fontsize_legend,
            title="Experiment Legend",
            title_fontsize=fontsize_legend + 1,
            frameon=False
        )
    # Ensure subtitles for every axis are displayed
    for ax_idx, metric in enumerate(metrics):
        axes[ax_idx].set_title(metric.replace("_", " ").title(), fontsize=fontsize_title)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Determine file extension based on save_as_tif flag
        if save_as_tif:
            # Remove existing extension and add .tif
            path_base = os.path.splitext(path)[0]
            save_path = path_base + ".tif"
            file_format = "tiff"
        else:
            save_path = path
            file_format = "png"
        
        # Apply CMYK conversion if requested
        if save_as_cmyk:
            # Set CMYK color conversion for PDF/postscript output
            matplotlib.rcParams['pdf.compression'] = 0
            matplotlib.rcParams['image.cmap'] = 'gray'
            # For CMYK without TIF, save as PDF for better CMYK support
            if not save_as_tif:
                save_path = os.path.splitext(save_path)[0] + ".pdf"
                file_format = "pdf"
        
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=file_format)
    if show:
        plt.show()
    plt.close(fig)




def load_curves(csv_path, metrics):
    df = pd.read_csv(csv_path)
    # Group by run_id, generation, take mean per metric
    agg = df.groupby(["run_id", "generation"], as_index=False)[metrics].mean()
    curves = {}
    for run_id, run_df in agg.groupby("run_id"):
        curves[run_id] = {}
        for metric in metrics:
            gen = run_df["generation"].to_numpy()
            vals = run_df[metric].to_numpy()
            # Sort by generation
            idx = np.argsort(gen)
            gen = gen[idx]
            vals = vals[idx]
            curves[run_id][metric] = (gen, vals)
    return curves

def compute_auc_and_final(curves, metrics):
    aucs = {metric: [] for metric in metrics}
    finals = {metric: [] for metric in metrics}
    for run_id in curves:
        for metric in metrics:
            gen, vals = curves[run_id][metric]
            auc = np.trapz(vals, gen)
            aucs[metric].append(auc)
            finals[metric].append(vals[-1])
    return aucs, finals

def pairwise_ttest(data1, data2, alternative='two-sided'):
    """
    Perform Welch's t-test between two groups.
    
    Parameters
    ----------
    data1 : array-like
        First group data
    data2 : array-like
        Second group data
    alternative : str
        Type of test: 'two-sided', 'greater', or 'less'
        - 'two-sided': tests if data1 != data2
        - 'greater': tests if data1 > data2
        - 'less': tests if data1 < data2
    
    Returns
    -------
    tuple
        (mean_diff, p_val, test_type)
    """
    # Welch's t-test (unequal variance)
    t_stat, p_val = ttest_ind(data1, data2, equal_var=False, alternative=alternative)
    mean_diff = np.mean(data1) - np.mean(data2)
    return mean_diff, p_val, alternative

def run_ttests(csv_paths, metrics, one_sided=False):
    """
    Load curves and compute AUC/final for each condition, then run pairwise t-tests.
    
    Parameters
    ----------
    csv_paths : list[str]
        List of CSV file paths
    metrics : list[str]
        Metrics to test
    one_sided : bool
        If True, performs one-sided tests in both directions for each pair.
        If False, performs two-sided tests (default).
    """
    # Load curves and compute AUC/final for each condition
    all_aucs = []
    all_finals = []
    exp_labels = []
    for csv_path in csv_paths:
        curves = load_curves(csv_path, metrics)
        aucs, finals = compute_auc_and_final(curves, metrics)
        all_aucs.append(aucs)
        all_finals.append(finals)
        exp_id = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        exp_labels.append(exp_id)

    print("Pairwise t-test results (mean diff, p-value):\n")
    for metric in metrics:
        print(f"\nMetric: {metric}")
        print("Comparison        \t\tAUC mean diff \tAUC_p-value \tFinal_mean_diff \tFinal_p-value")
        for i in range(len(csv_paths)):
            for j in range(i+1, len(csv_paths)):
                if one_sided:
                    # Test both directions
                    auc_diff_greater, auc_p_greater, _ = pairwise_ttest(all_aucs[i][metric], all_aucs[j][metric], alternative='greater')
                    auc_diff_less, auc_p_less, _ = pairwise_ttest(all_aucs[i][metric], all_aucs[j][metric], alternative='less')
                    final_diff_greater, final_p_greater, _ = pairwise_ttest(all_finals[i][metric], all_finals[j][metric], alternative='greater')
                    final_diff_less, final_p_less, _ = pairwise_ttest(all_finals[i][metric], all_finals[j][metric], alternative='less')
                    
                    comp = f"{exp_labels[i]} > {exp_labels[j]}"
                    print(f"{comp:<30}\t{auc_diff_greater:.4f}   \t{auc_p_greater:.4g}   \t{final_diff_greater:.4f}    \t\t{final_p_greater:.4g}")
                    
                    comp = f"{exp_labels[i]} < {exp_labels[j]}"
                    print(f"{comp:<30}\t{auc_diff_less:.4f}   \t{auc_p_less:.4g}   \t{final_diff_less:.4f}    \t\t{final_p_less:.4g}")
                else:
                    # Two-sided test
                    auc_diff, auc_p, _ = pairwise_ttest(all_aucs[i][metric], all_aucs[j][metric], alternative='two-sided')
                    final_diff, final_p, _ = pairwise_ttest(all_finals[i][metric], all_finals[j][metric], alternative='two-sided')
                    comp = f"{exp_labels[i]} vs {exp_labels[j]}"
                    print(f"{comp:<30}\t{auc_diff:.4f}   \t{auc_p:.4g}   \t{final_diff:.4f}    \t\t{final_p:.4g}")

