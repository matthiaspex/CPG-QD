import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import scipy.stats as stats


def aggregate_genotype_repetitions(csv_paths, output_suffix="_aggregated"):
    """
    For each CSV in csv_paths:
      - Group by (method, ring_size, arm_size, weight_coupling, run_id, generation, genotype_id)
      - Compute mean and std of fitness, sine_total_displacement, cosine_total_displacement,
        disk_elevation, ground_contact_fraction across genotype_rep
      - Save a new CSV with aggregated results (20x fewer rows)

    Parameters
    ----------
    csv_paths : list[str]
        Paths to input CSV files.
    output_suffix : str
        Suffix to append to output file names (before .csv extension).

    Returns
    -------
    list[str]
        Paths to the generated aggregated CSV files.
    """
    output_paths = []

    groupby_cols = [
        "method", "ring_size", "arm_size", "weight_coupling",
        "run_id", "generation", "genotype_id"
    ]
    
    metrics = [
        "fitness", "sine_total_displacement", "cosine_total_displacement",
        "disk_elevation", "ground_contact_fraction",
        "assistive_score", "bilateral_contralateral_score", "bilateral_score",
        "contralateral_score", "bilateral_score_grf", "contralateral_score_grf"
    ]
    
    # Columns that should remain constant within each group
    constant_cols = ["n_oscillators", "n_couplings", "spectral_gap", "induced_norm"]

    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Warning: missing CSV '{csv_path}', skipping.")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: empty CSV '{csv_path}', skipping.")
            continue

        # Verify required columns
        required = set(groupby_cols + metrics + ["genotype_rep"])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        print(f"Processing {csv_path}...")
        print(f"  Original rows: {len(df)}")

        # Aggregate: mean and std for each metric
        agg_dict = {metric: ["mean", "std"] for metric in metrics}
        
        # Keep constant columns (take first value)
        for col in constant_cols:
            if col in df.columns:
                agg_dict[col] = "first"

        agg_df = df.groupby(groupby_cols, as_index=False).agg(agg_dict)

        # Flatten multi-level column names
        agg_df.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in agg_df.columns
        ]

        print(f"  Aggregated rows: {len(agg_df)}")

        # Generate output path
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}{output_suffix}{ext}"
        agg_df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
        output_paths.append(output_path)

    return output_paths

def plot_std_distribution_grid(aggregated_paths, normalize=False, bins=50, min_generation=0, max_generation=None, show=False, path=None):
    """
    Plot histograms of variability across all generation×genotype_id rows from aggregated CSVs.

    For each aggregated CSV:
      - Extract metric_std (or metric_std/metric_mean if normalize=True) for:
        fitness, sine_total_displacement, cosine_total_displacement, disk_elevation, ground_contact_fraction
      - Overlay histograms per metric across files (same axes, different colors)
      - Legend below the figure with: exp_id, method, ring_size, arm_size, weight_coupling

    Parameters
    ----------
    aggregated_paths : list[str]
        Paths to aggregated CSV files (with columns like '<metric>_mean' and '<metric>_std').
    normalize : bool
        If True, plot coefficient of variation (std/mean). Else plot raw std.
    bins : int
        Number of bins for histograms.
    min_generation : int
        Only include rows with generation >= min_generation. Default is 0 (all generations).
    max_generation : int or None
        Only include rows with generation <= max_generation. If None, no upper limit.
    show : bool
        If True, display the plot via plt.show().
    path : str or None
        If provided, save the figure to this path.
    """
    metrics = [
        "fitness",
        "sine_total_displacement",
        "cosine_total_displacement",
        "disk_elevation",
        "ground_contact_fraction",
        "assistive_score",
        "bilateral_contralateral_score",
        "bilateral_score",
        "contralateral_score",
        "bilateral_score_grf",
        "contralateral_score_grf",
    ]

    # Use a 3x4 grid to accommodate 11 metrics (one slot will remain hidden)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=False)
    cmap = plt.colormaps.get_cmap("tab10")

    # Storage for overlay data per metric and legend entries
    data_per_metric = {m: [] for m in metrics}
    legend_labels = []
    legend_colors = []
    exp_ids_list = []

    for i, agg_path in enumerate(aggregated_paths):
        if not os.path.isfile(agg_path):
            print(f"Warning: missing aggregated CSV '{agg_path}', skipping.")
            continue

        df = pd.read_csv(agg_path)
        if df.empty:
            print(f"Warning: empty aggregated CSV '{agg_path}', skipping.")
            continue

        # Filter by generation threshold
        if 'generation' in df.columns:
            df = df[df['generation'] >= min_generation]
            if max_generation is not None:
                df = df[df['generation'] <= max_generation]
            if df.empty:
                range_str = f"generation >= {min_generation}" + (f" and <= {max_generation}" if max_generation is not None else "")
                print(f"Warning: no data for {range_str} in '{agg_path}', skipping.")
                continue

        # Derive exp_id from path (experiments/<exp_id>/numerical_metrics/...)
        exp_id = os.path.basename(os.path.dirname(os.path.dirname(agg_path))) or f"exp_{i}"
        exp_ids_list.append(exp_id)

        # Build label with constants per file
        meta_fields = ["method", "ring_size", "arm_size", "weight_coupling"]
        meta = {}
        for field in meta_fields:
            if field in df.columns:
                unique_vals = df[field].dropna().unique()
                if len(unique_vals) == 1:
                    meta[field] = unique_vals[0]
                else:
                    meta[field] = f"varies({','.join(map(str, unique_vals[:3]))}{'...' if len(unique_vals) > 3 else ''})"
        label = f"{exp_id}: " + ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else exp_id

        color = cmap(i % cmap.N)
        legend_labels.append(label)
        legend_colors.append(color)

        # Collect per-metric arrays
        for m in metrics:
            std_col = f"{m}_std"
            mean_col = f"{m}_mean"
            if std_col not in df.columns or (normalize and mean_col not in df.columns):
                print(f"Warning: '{std_col}'{', ' + mean_col if normalize else ''} missing in {agg_path}, skipping metric '{m}'.")
                continue

            std_vals = df[std_col].to_numpy(dtype=float)
            if normalize:
                mean_vals = df[mean_col].to_numpy(dtype=float)
                # Safe division using abs(mean); filter near-zero and invalid values
                with np.errstate(divide='ignore', invalid='ignore'):
                    vals = std_vals / np.abs(mean_vals)
                mask = np.isfinite(vals) & (np.abs(mean_vals) > 1e-10)
                vals = vals[mask]
            else:
                vals = std_vals[np.isfinite(std_vals)]

            data_per_metric[m].append((vals, color))

    # Plot histograms in a 3x4 grid
    axes_flat = axes.ravel()
    for idx, m in enumerate(metrics):
        ax = axes_flat[idx]
        series = data_per_metric[m]
        if not series:
            ax.set_visible(False)
            continue

        # Compute shared bin edges across all overlays for this metric
        all_vals = np.concatenate([vals for vals, _ in series if len(vals) > 0]) if any(len(vals) > 0 for vals, _ in series) else np.array([])
        if all_vals.size > 0:
            # Trim 2.5% from each tail
            p_low, p_high = np.percentile(all_vals, [2.5, 97.5])
            vmin, vmax = float(p_low), float(p_high)
            if vmin == vmax:
                # Expand a tiny bit if degenerate
                vmin -= 1e-9
                vmax += 1e-9
            edges = np.linspace(vmin, vmax, bins)
        else:
            edges = bins  # fallback

        # Overlay histograms with filled, semi-transparent areas (trimmed to 2.5-97.5 percentile range)
        for vals, color in series:
            if len(vals) == 0:
                continue
            # Filter to percentile range
            if all_vals.size > 0:
                trimmed_vals = vals[(vals >= p_low) & (vals <= p_high)]
            else:
                trimmed_vals = vals
            ax.hist(trimmed_vals, bins=edges, histtype="stepfilled", color=color, alpha=0.5, edgecolor=color, linewidth=0.5)

        label_suffix = " (CV = std/mean)" if normalize else " (std)"
        ax.set_title(f"{m.replace('_', ' ').title()}{label_suffix}", fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Hide any remaining unused subplots (there will be 12 slots for 11 metrics)
    if len(metrics) < len(axes_flat):
        for k in range(len(metrics), len(axes_flat)):
            axes_flat[k].set_visible(False)

    # Suptitle with exp_ids
    title_prefix = "Coefficient of Variation (CV)" if normalize else "Standard Deviation"
    title_ids = " - ".join(exp_ids_list) if exp_ids_list else "(no experiments)"
    if min_generation > 0 or max_generation is not None:
        gen_parts = []
        if min_generation > 0:
            gen_parts.append(f"gen >= {min_generation}")
        if max_generation is not None:
            gen_parts.append(f"gen <= {max_generation}")
        gen_suffix = f" | {', '.join(gen_parts)}"
    else:
        gen_suffix = " | all generations"
    suptitle = fig.suptitle(f"{title_prefix}: {title_ids}{gen_suffix}", fontsize=13, y=0.98)

    # Build a single legend below the figure
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2) for c in legend_colors]
    legend = fig.legend(
        handles, legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        title="Configuration",
        fontsize=9
    )

    # Leave space for legend under axes and suptitle on top, add spacing between subplots
    fig.subplots_adjust(bottom=0.20, top=0.92, hspace=0.35, wspace=0.28)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_extra_artists=(legend, suptitle), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)



def plot_mean_distribution_grid(aggregated_paths, bins=50, min_generation=0, max_generation=None, show=False, path=None):
    """
    Plot histograms of mean values across all generation×genotype_id rows from aggregated CSVs.

    For each aggregated CSV:
      - Extract metric_mean for: fitness, sine_total_displacement, cosine_total_displacement, 
        disk_elevation, ground_contact_fraction
      - Overlay histograms per metric across files (same axes, different colors)
      - Legend below the figure with: exp_id, method, ring_size, arm_size, weight_coupling

    Parameters
    ----------
    aggregated_paths : list[str]
        Paths to aggregated CSV files (with columns like '<metric>_mean' and '<metric>_std').
    bins : int
        Number of bins for histograms.
    min_generation : int
        Only include rows with generation >= min_generation. Default is 0 (all generations).
    max_generation : int or None
        Only include rows with generation <= max_generation. If None, no upper limit.
    show : bool
        If True, display the plot via plt.show().
    path : str or None
        If provided, save the figure to this path.
    """
    metrics = [
        "fitness",
        "sine_total_displacement",
        "cosine_total_displacement",
        "disk_elevation",
        "ground_contact_fraction",
        "assistive_score",
        "bilateral_contralateral_score",
        "bilateral_score",
        "contralateral_score",
        "bilateral_score_grf",
        "contralateral_score_grf",
    ]

    # Use a 3x4 grid to accommodate 11 metrics (one slot will remain hidden)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=False)
    cmap = plt.colormaps.get_cmap("tab10")

    # Storage for overlay data per metric and legend entries
    data_per_metric = {m: [] for m in metrics}
    legend_labels = []
    legend_colors = []
    exp_ids_list = []

    for i, agg_path in enumerate(aggregated_paths):
        if not os.path.isfile(agg_path):
            print(f"Warning: missing aggregated CSV '{agg_path}', skipping.")
            continue

        df = pd.read_csv(agg_path)
        if df.empty:
            print(f"Warning: empty aggregated CSV '{agg_path}', skipping.")
            continue

        # Filter by generation threshold
        if 'generation' in df.columns:
            df = df[df['generation'] >= min_generation]
            if max_generation is not None:
                df = df[df['generation'] <= max_generation]
            if df.empty:
                range_str = f"generation >= {min_generation}" + (f" and <= {max_generation}" if max_generation is not None else "")
                print(f"Warning: no data for {range_str} in '{agg_path}', skipping.")
                continue

        # Derive exp_id from path (experiments/<exp_id>/numerical_metrics/...)
        exp_id = os.path.basename(os.path.dirname(os.path.dirname(agg_path))) or f"exp_{i}"
        exp_ids_list.append(exp_id)

        # Build label with constants per file
        meta_fields = ["method", "ring_size", "arm_size", "weight_coupling"]
        meta = {}
        for field in meta_fields:
            if field in df.columns:
                unique_vals = df[field].dropna().unique()
                if len(unique_vals) == 1:
                    meta[field] = unique_vals[0]
                else:
                    meta[field] = f"varies({','.join(map(str, unique_vals[:3]))}{'...' if len(unique_vals) > 3 else ''})"
        label = f"{exp_id}: " + ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else exp_id

        color = cmap(i % cmap.N)
        legend_labels.append(label)
        legend_colors.append(color)

        # Collect per-metric arrays (mean values)
        for m in metrics:
            mean_col = f"{m}_mean"
            if mean_col not in df.columns:
                print(f"Warning: '{mean_col}' missing in {agg_path}, skipping metric '{m}'.")
                continue

            mean_vals = df[mean_col].to_numpy(dtype=float)
            vals = mean_vals[np.isfinite(mean_vals)]

            data_per_metric[m].append((vals, color))

    # Plot histograms in a 3x4 grid
    axes_flat = axes.ravel()
    for idx, m in enumerate(metrics):
        ax = axes_flat[idx]
        series = data_per_metric[m]
        if not series:
            ax.set_visible(False)
            continue

        # Compute shared bin edges across all overlays for this metric
        all_vals = np.concatenate([vals for vals, _ in series if len(vals) > 0]) if any(len(vals) > 0 for vals, _ in series) else np.array([])
        if all_vals.size > 0:
            # Trim 2.5% from each tail
            p_low, p_high = np.percentile(all_vals, [2.5, 97.5])
            vmin, vmax = float(p_low), float(p_high)
            if vmin == vmax:
                # Expand a tiny bit if degenerate
                vmin -= 1e-9
                vmax += 1e-9
            edges = np.linspace(vmin, vmax, bins)
        else:
            edges = bins  # fallback

        # Overlay histograms with filled, semi-transparent areas (trimmed to 2.5-97.5 percentile range)
        for vals, color in series:
            if len(vals) == 0:
                continue
            # Filter to percentile range
            if all_vals.size > 0:
                trimmed_vals = vals[(vals >= p_low) & (vals <= p_high)]
            else:
                trimmed_vals = vals
            ax.hist(trimmed_vals, bins=edges, histtype="stepfilled", color=color, alpha=0.5, edgecolor=color, linewidth=0.5)

        ax.set_title(f"{m.replace('_', ' ').title()} (mean)", fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Hide any remaining unused subplots (there will be 12 slots for 11 metrics)
    if len(metrics) < len(axes_flat):
        for k in range(len(metrics), len(axes_flat)):
            axes_flat[k].set_visible(False)

    # Suptitle with exp_ids
    title_ids = " - ".join(exp_ids_list) if exp_ids_list else "(no experiments)"
    if min_generation > 0 or max_generation is not None:
        gen_parts = []
        if min_generation > 0:
            gen_parts.append(f"gen >= {min_generation}")
        if max_generation is not None:
            gen_parts.append(f"gen <= {max_generation}")
        gen_suffix = f" | {', '.join(gen_parts)}"
    else:
        gen_suffix = " | all generations"
    suptitle = fig.suptitle(f"Mean Values: {title_ids}{gen_suffix}", fontsize=13, y=0.98)

    # Build a single legend below the figure
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2) for c in legend_colors]
    legend = fig.legend(
        handles, legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        title="Configuration",
        fontsize=9
    )

    # Leave space for legend under axes and suptitle on top, add spacing between subplots
    fig.subplots_adjust(bottom=0.20, top=0.92, hspace=0.35, wspace=0.28)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_extra_artists=(legend, suptitle), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def load_variability_values(agg_path, metric, normalize=False, min_generation=0, max_generation=None):
    """
    Load std (or CV=std/mean) values for a metric from an aggregated CSV.
    Returns a 1D numpy array with finite values only.
    
    Parameters
    ----------
    agg_path : str
        Path to aggregated CSV file.
    metric : str
        Metric name (e.g., 'fitness').
    normalize : bool
        If True, return CV (std/|mean|). Else return std.
    min_generation : int
        Only include rows with generation >= min_generation.
    max_generation : int or None
        Only include rows with generation <= max_generation. If None, no upper limit.
    """
    if not os.path.isfile(agg_path):
        raise FileNotFoundError(agg_path)
    df = pd.read_csv(agg_path)
    
    # Filter by generation threshold
    if 'generation' in df.columns:
        df = df[df['generation'] >= min_generation]
        if max_generation is not None:
            df = df[df['generation'] <= max_generation]
    std_col = f"{metric}_std"
    mean_col = f"{metric}_mean"
    if std_col not in df.columns or (normalize and mean_col not in df.columns):
        raise ValueError(f"Missing columns for metric '{metric}' in {agg_path}")
    std_vals = df[std_col].to_numpy(dtype=float)
    if normalize:
        mean_vals = df[mean_col].to_numpy(dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = std_vals / np.abs(mean_vals)
        mask = np.isfinite(vals) & (np.abs(mean_vals) > 1e-10)
        return vals[mask]
    return std_vals[np.isfinite(std_vals)]

def test_variance_two_sets(a, b, center='median'):
    """
    Test whether variances differ between two independent samples.
    Returns a dict with Levene (robust), Bartlett (normality-assuming),
    and parametric F-test (two-sided; normality-assuming).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        raise ValueError("Need at least 2 finite observations in each set.")

    # Robust Levene (Brown–Forsythe if center='median')
    lev_stat, lev_p = stats.levene(a, b, center=center)

    # Bartlett (assumes normality; sensitive to non-normality)
    bart_stat, bart_p = stats.bartlett(a, b)

    # Classical two-sample F-test (assumes normality)
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    F = va / vb
    df1, df2 = a.size - 1, b.size - 1
    cdf = stats.f.cdf(F, df1, df2)
    f_p_two_sided = 2 * min(cdf, 1 - cdf)
    f_p_two_sided = min(1.0, max(0.0, f_p_two_sided))

    return {
        "levene_stat": float(lev_stat), "levene_p": float(lev_p), "levene_center": center,
        "bartlett_stat": float(bart_stat), "bartlett_p": float(bart_p),
        "f_stat": float(F), "f_df": (int(df1), int(df2)), "f_p_two_sided": float(f_p_two_sided),
        "var_a": float(va), "var_b": float(vb), "n_a": int(a.size), "n_b": int(b.size),
    }

def pairwise_variance_tests(aggregated_paths, metric, normalize=False, center='median', min_generation=0, max_generation=None):
    """
    Run variance-difference tests for all pairs of aggregated CSVs on a given metric.
    Returns a DataFrame with p-values and summary stats for each pair.
    
    Parameters
    ----------
    aggregated_paths : list[str]
        Paths to aggregated CSV files.
    metric : str
        Metric name to test.
    normalize : bool
        If True, test CV instead of raw std.
    center : str
        Levene test center ('median' or 'mean').
    min_generation : int
        Only include rows with generation >= min_generation.
    max_generation : int or None
        Only include rows with generation <= max_generation. If None, no upper limit.
    """
    rows = []
    def label_from_path(p):
        return os.path.basename(os.path.dirname(os.path.dirname(p)))

    for i in range(len(aggregated_paths)):
        for j in range(i + 1, len(aggregated_paths)):
            pa, pb = aggregated_paths[i], aggregated_paths[j]
            a = load_variability_values(pa, metric, normalize=normalize, min_generation=min_generation, max_generation=max_generation)
            b = load_variability_values(pb, metric, normalize=normalize, min_generation=min_generation, max_generation=max_generation)
            res = test_variance_two_sets(a, b, center=center)
            rows.append({
                "A": label_from_path(pa), "B": label_from_path(pb),
                "metric": metric, "normalize": normalize,
                "levene_p": res["levene_p"], "bartlett_p": res["bartlett_p"], "f_p": res["f_p_two_sided"],
                "var_a": res["var_a"], "var_b": res["var_b"], "n_a": res["n_a"], "n_b": res["n_b"],
            })

    return pd.DataFrame(rows)

