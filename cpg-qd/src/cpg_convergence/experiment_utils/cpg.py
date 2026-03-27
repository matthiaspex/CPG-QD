import os
import sys

import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp
from typing import Literal, List, Tuple, Any, Dict, Optional, Union
import chex
import math

from plotly import express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import seaborn as sns

from scipy.stats import chi2



def generate_ring_setup_variations(rng: jax.random.PRNGKey,
                                   cpg_type: Literal["ring", "equal_arms", "varying_arms"],
                                   nvar: int,
                                   range_of_variation: Tuple[int, int],
                                   mu_gamma: float = None
                                   ) -> Tuple[List[chex.Array], List[Any]]:
    """
    Generate a list of ring setup variations based on the specified CPG type and parameters.
    The variational parameter is either ring_size (ring), segments_per_arm (equal arms), or cv (varying arms).
    Options:
    - "ring": Generates variations for a ring without arms. Uniformly sampled within range
    - "equal_arms": Generates variations for a ring of 5 oscillators with arms of equal length.
                    Length of the arms uniformly sampled within range.
    - "varying_arms": Generates variations for a ring of 5 oscillators with arms of varying lengths.
                      Coefficient of variation (cv) sampled uniformly within range, arm lengths sampled from gamma distribution.
    Parameters:
    - rng: JAX random key for reproducibility.
    - cpg_type: Type of CPG configuration ("ring", "equal_arms", "varying_arms").
    - nvar: Number of variations to generate.
    - range_of_variation: Tuple specifying the (min, max) range for sampling ring_size, arm_lengths or cv.
                If min = max, the value is fixed.
    - mu_gamma: Mean for gamma distribution (only for "varying_arms").
    """
    ring_setup_variations = []
    parameter_variation_list = []

    if cpg_type == "ring":
        for _ in range(nvar):
            rng, rng_sample = jax.random.split(rng)
            ring_size = int(jax.random.randint(
                rng_sample, shape=(), minval=range_of_variation[0], maxval=range_of_variation[1] + 1
            ))
            ring_setup = jnp.array([0] * ring_size)
            ring_setup_variations.append(ring_setup)
            parameter_variation_list.append(ring_size)

    elif cpg_type == "equal_arms":
        for _ in range(nvar):
            rng, rng_sample = jax.random.split(rng)
            arm_length = int(jax.random.randint(
                rng_sample, shape=(), minval=range_of_variation[0], maxval=range_of_variation[1] + 1
            ))
            ring_setup = jnp.array(5 * [arm_length])
            ring_setup_variations.append(ring_setup)
            parameter_variation_list.append(arm_length)

    elif cpg_type == "varying_arms":
        if mu_gamma is None:
            raise ValueError("mu_gamma must be provided for 'varying_arms' CPG type.")
        for _ in range(nvar):
            rng, rng_sample = jax.random.split(rng)
            cv = jax.random.uniform(
                rng_sample, shape=(), minval=range_of_variation[0], maxval=range_of_variation[1]
            )
            ring_setup = gamma_with_cv(rng_sample, mu=mu_gamma, cv=cv, sample_size=5) # sample 5 arm lengths
            ring_setup = jnp.round(ring_setup).astype(int) # round to nearest integer
            ring_setup_variations.append(ring_setup)
            parameter_variation_list.append(cv)

    else:
        raise ValueError(f"Unknown cpg_type: {cpg_type}")
    
    return ring_setup_variations, parameter_variation_list
    

def gamma_with_cv(rng: jax.random.PRNGKey, mu: float, cv: float, sample_size: int=5) -> int:
    """
    Sample from a gamma distribution with given mean (mu) and coefficient of variation (cv).
    The coefficient of variation is defined as the ratio of the standard deviation to the mean.
    """
    alpha = 1.0 / (cv ** 2)
    beta = alpha / mu
    samples = jax.random.gamma(rng, alpha, shape=(sample_size,)) / beta
    return samples



def create_csv(file_path: str, header: List[str]) -> None:
    """Create a CSV file with the specified header."""
    if os.path.exists(file_path):
        raise FileExistsError(f"File already exists: {file_path}")
    with open(file_path, 'w') as f:
        f.write(','.join(header) + '\n')

def add_csv_entry(file_path: str, entry: Dict[str, Union[str, float, int]]) -> None:
    """Add an entry to the CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    # Extract header from the CSV file
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')

    # Check if all keys in entry are in header
    for key in entry.keys():
        if key not in header:
            raise ValueError(f"Key '{key}' in entry is not in the CSV header.")
    
    with open(file_path, 'a') as f:
        row = []
        for col in header:
            value = entry.get(col, "")
            row.append(str(value))
        f.write(','.join(row) + '\n')


def write_metadata_dict_to_txt(
    metadata: dict,
    file_path: str
):
    """
    Write the metadata dictionary to a text file in a readable format.
    Each key-value pair will be written on a new line.
    """
    assert file_path.endswith('.txt'), "File path must end with .txt"
    with open(file_path, 'w') as f:
        f.write("Metadata Information:\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def create_experiment_directory(
        base_dir: str,
        experiment_id: str
):
    """
    Create a directory structure for the experiment.
    The structure will be:
    base_dir/experiments/experiment_name/numerical_metrics/
        base_dir/experiments/experiment_name/plots/
        base_dir/experiments/experiment_name/run/
        base_dir/experiments/experiment_name/videos/
    Return
        numerical_metrics_dir: the path to the numerical_metrics directory
        plots_dir: the path to the plots directory
        run_dir: the path to the run directory
        videos_dir: the path to the videos directory
    """
    experiment_dir = os.path.join(base_dir, "experiments", experiment_id)
    if not os.path.exists(experiment_dir):
        numerical_metrics_dir = os.path.join(experiment_dir, "numerical_metrics")
        plots_dir = os.path.join(experiment_dir, "plots")
        run_dir = os.path.join(experiment_dir, "run")
        videos_dir = os.path.join(experiment_dir, "videos")
        dirs = [numerical_metrics_dir, plots_dir, run_dir, videos_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            # Create the .gitkeep file inside it, this makes sure that the directory is tracked by git
            # empty directories are not tracked by git
            gitkeep_path = os.path.join(d, '.gitkeep')
            with open(gitkeep_path, 'w') as f:
                pass  # Create an empty file
        print(f"Experiment directory structure created at {experiment_dir}")

    else:
        raise FileExistsError(f"Experiment directory {experiment_dir} already exists.\n\tPlease choose a different name or remove the existing directory.")

    return numerical_metrics_dir, plots_dir, run_dir, videos_dir



def aggregate_convergence_csv(csv_path: str) -> None:
    """
    Aggregate the new-format convergence CSV by experimental setup:
    (method, ring_size, arm_size, weight_coupling).

    For each setup, compute mean, std (ddof=1), and median for:
      - step_conv_p50, step_conv_p75, step_conv_p90, step_conv_p100
      - fraction_not_converged

    Also compute pairwise uncertainty score between each step_conv_pXX and fraction_not_converged:
      uncertainty = sqrt( (std_step/mean_step)^2 + (std_frac/mean_frac)^2 )
      with NaN/Inf handled via np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)

    Writes: "<dir>/convergence_results_aggregated.csv"
    """
    df = pd.read_csv(csv_path)

    groupby_cols = ["method", "ring_size", "arm_size", "weight_coupling"]
    const_cols = ["n_oscillators", "n_couplings", "spectral_gap", "induced_norm"]
    step_cols = ["step_conv_p50", "step_conv_p75", "step_conv_p90", "step_conv_p100"]
    frac_col = "fraction_not_converged"

    # Ensure numeric types
    for c in const_cols + step_cols + [frac_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    grouped = df.groupby(groupby_cols, dropna=False)

    rows = []
    for keys, grp in grouped:
        out = dict(zip(groupby_cols, keys))
        out["n_runs"] = int(grp.shape[0])

        # Carry constants (first non-null)
        for const in const_cols:
            if const in grp.columns:
                vals = grp[const].dropna().unique()
                out[const] = vals[0] if len(vals) > 0 else np.nan

        # Stats for each metric
        # Mean / std / median
        for c in step_cols + [frac_col]:
            x = grp[c].astype(float)
            out[f"mean_{c}"] = float(x.mean())
            out[f"std_{c}"] = float(x.std(ddof=1))
            out[f"median_{c}"] = float(x.median())

        # Pairwise uncertainty scores with fraction_not_converged (use current definition)
        mean_frac = out[f"mean_{frac_col}"]
        std_frac = out[f"std_{frac_col}"]
        cv_frac = std_frac / mean_frac if mean_frac not in (0.0, None) else np.nan
        cv_frac = np.nan_to_num(cv_frac, nan=0.0, posinf=0.0, neginf=0.0)

        for c in step_cols:
            mean_step = out[f"mean_{c}"]
            std_step = out[f"std_{c}"]
            cv_step = std_step / mean_step if mean_step not in (0.0, None) else np.nan
            cv_step = np.nan_to_num(cv_step, nan=0.0, posinf=0.0, neginf=0.0)

            uncertainty = float(np.sqrt(cv_step**2 + cv_frac**2))
            out[f"uncertainty_{c}_{frac_col}"] = uncertainty

        rows.append(out)

    aggregated = pd.DataFrame(rows).sort_values(groupby_cols).reset_index(drop=True)

    # Save aggregated CSV next to input
    output_path = os.path.join(os.path.dirname(csv_path), "convergence_results_aggregated.csv")
    aggregated.to_csv(output_path, index=False)
    print(f"Aggregated CSV saved to: {output_path}")
    print(f"Shape: {aggregated.shape}")
    if len(aggregated) > 0:
        print(aggregated.head())


# ==============================
# Pareto Plots

def load_aggregated_data(csv_path: str) -> pd.DataFrame:
    """Load the aggregated convergence results CSV."""
    return pd.read_csv(csv_path)


def pareto_mask(points):
    """
    points: array of shape (N, 2)
            columns are [time_to_convergence, fraction_not_converged]
    returns: boolean array, True if Pareto-optimal
    """
    N = points.shape[0]
    is_pareto = np.ones(N, dtype=bool)

    for i in range(N):
        if not is_pareto[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            if (
                points[j, 0] <= points[i, 0] and
                points[j, 1] <= points[i, 1] and
                (
                    points[j, 0] < points[i, 0] or
                    points[j, 1] < points[i, 1]
                )
            ):
                is_pareto[i] = False
                break

    return is_pareto


def pareto_mask_fast(points: np.ndarray) -> np.ndarray:
    """
    Fast O(N log N) Pareto front detection for 2D minimization.
    
    Args:
        points: array of shape (N, 2)
                [time_to_convergence, fraction_not_converged]
    
    Returns:
        boolean mask, True if Pareto-optimal
    """
    N = points.shape[0]
    indices = np.arange(N)
    
    # Sort by first objective (time), then second objective (fraction)
    sorted_idx = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sorted_idx]
    
    is_pareto_sorted = np.ones(N, dtype=bool)
    min_frac = np.inf
    
    for i in range(N):
        if sorted_points[i, 1] < min_frac:
            min_frac = sorted_points[i, 1]
        else:
            is_pareto_sorted[i] = False
    
    # Map back to original order
    is_pareto = np.zeros(N, dtype=bool)
    is_pareto[sorted_idx] = is_pareto_sorted
    
    return is_pareto

# def create_covariance_ellipse(mean_x, mean_y, std_x, std_y, cov, n_std=2.0, n_points=50):
#     """
#     Create points for a covariance ellipse.
    
#     Args:
#         mean_x, mean_y: Center coordinates
#         std_x, std_y: Standard deviations
#         cov: Covariance between x and y
#         n_std: Number of standard deviations for ellipse size
#         n_points: Number of points for the ellipse
    
#     Returns:
#         x, y: Arrays of ellipse boundary points
#     """
#     # Covariance matrix
#     cov_matrix = np.array([[std_x**2, cov],
#                            [cov, std_y**2]])
    
#     # Eigenvalues and eigenvectors
#     eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
#     # Sort by eigenvalue
#     order = eigenvalues.argsort()[::-1]
#     eigenvalues = eigenvalues[order]
#     eigenvectors = eigenvectors[:, order]
    
#     # Angle of rotation
#     angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
#     # Width and height of ellipse (scaled by chi-square for 2D, 95% confidence)
#     chi2_val = chi2.ppf(0.95, 2)
#     width = 2 * n_std * np.sqrt(chi2_val * eigenvalues[0])
#     height = 2 * n_std * np.sqrt(chi2_val * eigenvalues[1])
    
#     # Generate ellipse points
#     t = np.linspace(0, 2 * np.pi, n_points)
#     ellipse_x = (width / 2) * np.cos(t)
#     ellipse_y = (height / 2) * np.sin(t)
    
#     # Rotate and translate
#     R = np.array([[np.cos(angle), -np.sin(angle)],
#                   [np.sin(angle), np.cos(angle)]])
#     ellipse_points = R @ np.vstack([ellipse_x, ellipse_y])
    
#     x = ellipse_points[0, :] + mean_x
#     y = ellipse_points[1, :] + mean_y
    
#     return x, y

def clip_and_normalize_uncertainty_for_metric(df: pd.DataFrame, uncertainty_col: str,
                                               p_low: float = 5, p_high: float = 95) -> pd.DataFrame:
    """
    Clip uncertainty scores to percentiles and normalize to [0.2, 1.0] for alpha values.
    Higher uncertainty -> more transparent (lower alpha).
    Handles NaN values by replacing them with the median.
    """
    df = df.copy()
    
    if uncertainty_col not in df.columns:
        df['alpha'] = 0.7
        return df
    
    # Replace NaN in uncertainty with median
    median_unc = df[uncertainty_col].median()
    df[uncertainty_col] = df[uncertainty_col].fillna(median_unc)
    
    lower = np.percentile(df[uncertainty_col], p_low)
    upper = np.percentile(df[uncertainty_col], p_high)
    
    if upper == lower:
        df['alpha'] = 0.6
    else:
        df['uncertainty_clipped'] = np.clip(df[uncertainty_col], lower, upper)
        # Higher uncertainty -> more transparent
        df['alpha'] = 1.0 - 0.8 * (df['uncertainty_clipped'] - lower) / (upper - lower)
    
    return df


def create_interactive_convergence_plot(csv_path: str, port: int = 8050):
    """
    Interactive Plotly/Dash visualization for convergence results (new CSV format).
    
    Features:
    - Tab 1: Regular plot with all metrics
      - Select method (checkbox, multiple allowed)
      - Sliders for ring_size and arm_size
      - Range slider for weight_coupling
      - Select x-axis metric (radio: step_conv_p50/p75/p90/p100)
      - Select position metric (radio: mean or median)
      - Y-axis: fraction_not_converged
      - Weight coupling: marker size + connecting lines
      - Opacity: uncertainty score for selected x-metric
      - Point count display for debugging
    - Tab 2: Pareto front analysis
      - Highlights points on Pareto front
      - Same controls as Tab 1
    """
    df = load_aggregated_data(csv_path)
    
    methods = sorted(df['method'].unique())
    method_colors = {m: c for m, c in zip(methods,
                                          ['#1f77b4', '#ff7f0e', '#2ca02c',
                                           '#d62728', '#9467bd', '#8c564b'])}
    
    step_metrics = ['step_conv_p50', 'step_conv_p75', 'step_conv_p90', 'step_conv_p100']
    position_options = ['mean', 'median']
    
    # Get unique values for sliders
    ring_values = sorted(df['ring_size'].dropna().unique())
    arm_values = sorted(df['arm_size'].dropna().unique())
    weight_values = sorted(df['weight_coupling'].dropna().unique())
    
    ring_marks = {i: f"{ring_values[i]}" for i in range(len(ring_values))}
    arm_marks = {i: f"{arm_values[i]}" for i in range(len(arm_values))}
    
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        dbc.Row([dbc.Col(html.H1("CPG Convergence Analysis",
                                 className="text-center mb-4"))]),
        dbc.Row([
            dbc.Col([
                html.H4("Controls", className="mb-3"),
                
                # Methods checkbox
                html.Label("Methods:", className="fw-bold mt-3"),
                dcc.Checklist(
                    id='method-checklist',
                    options=[{'label': m, 'value': m} for m in methods],
                    value=methods,
                    labelStyle={'display': 'block', 'margin': '5px'}
                ),
                
                html.Hr(),
                
                # Ring Size range slider
                html.Label("Ring Size Range:", className="fw-bold mt-3"),
                dcc.RangeSlider(
                    id='ring-size-slider',
                    min=0, max=len(ring_values)-1, step=1, 
                    value=[0, len(ring_values)-1],
                    marks=ring_marks,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                
                html.Hr(),
                
                # Arm Size range slider
                html.Label("Arm Size Range:", className="fw-bold mt-3"),
                dcc.RangeSlider(
                    id='arm-size-slider',
                    min=0, max=len(arm_values)-1, step=1, 
                    value=[0, len(arm_values)-1],
                    marks=arm_marks,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                
                html.Hr(),
                
                # Weight Coupling range slider
                html.Label("Weight Coupling Range:", className="fw-bold mt-3"),
                dcc.RangeSlider(
                    id='weight-coupling-range-slider',
                    min=0, max=len(weight_values)-1, step=1, 
                    value=[0, len(weight_values)-1],
                    marks={i: f"{weight_values[i]:.2f}" for i in range(len(weight_values))},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                
                html.Hr(),
                
                # X-axis metric selection (radio)
                html.Label("X-Axis Metric:", className="fw-bold mt-3"),
                dcc.RadioItems(
                    id='xaxis-metric-radio',
                    options=[{'label': m, 'value': m} for m in step_metrics],
                    value=step_metrics[0],
                    labelStyle={'display': 'block', 'margin': '5px'}
                ),
                
                html.Hr(),
                
                # Position metric selection (radio)
                html.Label("Position (Mean/Median):", className="fw-bold mt-3"),
                dcc.RadioItems(
                    id='position-metric-radio',
                    options=[{'label': pos, 'value': pos} for pos in position_options],
                    value=position_options[0],
                    labelStyle={'display': 'block', 'margin': '5px'}
                ),
                
            ], width=3, className="bg-light p-3"),
            
            dbc.Col([
                dcc.Tabs(id='tabs', value='tab-1', children=[
                    dcc.Tab(label='All Data', value='tab-1', children=[
                        html.Div(id='point-count', className="text-muted small mb-2"),
                        dcc.Graph(id='convergence-plot', style={'height': '85vh'},
                                  config={'displayModeBar': True})
                    ]),
                    dcc.Tab(label='Pareto Front', value='tab-2', children=[
                        html.Div(id='pareto-point-count', className="text-muted small mb-2"),
                        dcc.Graph(id='pareto-plot', style={'height': '85vh'},
                                  config={'displayModeBar': True})
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)
    
    # Main plot callback - All Data Tab
    @app.callback(
        [Output('convergence-plot', 'figure'),
         Output('point-count', 'children')],
        [Input('method-checklist', 'value'),
         Input('ring-size-slider', 'value'),
         Input('arm-size-slider', 'value'),
         Input('weight-coupling-range-slider', 'value'),
         Input('xaxis-metric-radio', 'value'),
         Input('position-metric-radio', 'value')],
        prevent_initial_call=False
    )
    def update_plot(selected_methods, ring_range, arm_range, weight_range, xaxis_metric, position_metric):
        
        # Get selected ranges from slider indices
        ring_min_idx = ring_range[0]
        ring_max_idx = ring_range[1]
        selected_ring_min = ring_values[ring_min_idx] if ring_min_idx < len(ring_values) else ring_values[0]
        selected_ring_max = ring_values[ring_max_idx] if ring_max_idx < len(ring_values) else ring_values[-1]
        
        arm_min_idx = arm_range[0]
        arm_max_idx = arm_range[1]
        selected_arm_min = arm_values[arm_min_idx] if arm_min_idx < len(arm_values) else arm_values[0]
        selected_arm_max = arm_values[arm_max_idx] if arm_max_idx < len(arm_values) else arm_values[-1]
        
        # Get weight coupling range
        weight_min_idx = weight_range[0]
        weight_max_idx = weight_range[1]
        selected_weight_min = weight_values[weight_min_idx] if weight_min_idx < len(weight_values) else weight_values[0]
        selected_weight_max = weight_values[weight_max_idx] if weight_max_idx < len(weight_values) else weight_values[-1]
        
        # Filter by methods and ranges
        filtered = df[
            (df['method'].isin(selected_methods)) &
            (df['ring_size'] >= selected_ring_min) &
            (df['ring_size'] <= selected_ring_max) &
            (df['arm_size'] >= selected_arm_min) &
            (df['arm_size'] <= selected_arm_max) &
            (df['weight_coupling'] >= selected_weight_min) &
            (df['weight_coupling'] <= selected_weight_max)
        ].copy()
        
        if len(filtered) == 0:
            return go.Figure().add_annotation(text="No data for selected parameters"), "0 points"
        
        # Get n_oscillators from filtered data
        n_oscillators = filtered['n_oscillators'].iloc[0] if len(filtered) > 0 else 'N/A'
        
        # Compute opacity based on uncertainty score for selected xaxis metric
        uncertainty_col = f"uncertainty_{xaxis_metric}_fraction_not_converged"
        filtered = clip_and_normalize_uncertainty_for_metric(filtered, uncertainty_col)
        
        # Build figure
        fig = go.Figure()
        weights = sorted(filtered['weight_coupling'].unique())
        weight_to_size = {w: 10 + 30 * (i / max(len(weights) - 1, 1))
                          for i, w in enumerate(weights)}
        
        # X and Y column names
        x_col = f"{position_metric}_{xaxis_metric}"
        y_col = f"{position_metric}_fraction_not_converged"
        
        total_points = 0
        
        # Plot each unique combination of (method, ring_size, arm_size) separately
        # so lines only connect dots with same setup but different weights
        for method in selected_methods:
            method_data = filtered[filtered['method'] == method]
            
            if len(method_data) == 0:
                continue
            
            if x_col not in method_data.columns or y_col not in method_data.columns:
                continue
            
            # Get unique (ring_size, arm_size) combinations for this method
            unique_morphologies = method_data[['ring_size', 'arm_size']].drop_duplicates()
            
            first_trace = True
            for _, morph_row in unique_morphologies.iterrows():
                ring_sz = morph_row['ring_size']
                arm_sz = morph_row['arm_size']
                
                # Filter to this specific morphology and sort by weight
                morph_data = method_data[
                    (method_data['ring_size'] == ring_sz) &
                    (method_data['arm_size'] == arm_sz)
                ].sort_values('weight_coupling')
                
                if len(morph_data) == 0:
                    continue
                
                total_points += len(morph_data)
                
                # Build hover text with spectral_gap and induced_norm
                hover_texts = []
                for _, row in morph_data.iterrows():
                    spec_gap = row.get('spectral_gap', 'N/A')
                    ind_norm = row.get('induced_norm', 'N/A')
                    if isinstance(spec_gap, float):
                        spec_gap = f"{spec_gap:.4f}"
                    if isinstance(ind_norm, float):
                        ind_norm = f"{ind_norm:.4f}"
                    
                    hover_text = (
                        f"<b>Method:</b> {row['method']}<br>"
                        f"<b>Ring Size:</b> {int(row['ring_size'])}<br>"
                        f"<b>Arm Size:</b> {int(row['arm_size'])}<br>"
                        f"<b>Weight:</b> {row['weight_coupling']}<br>"
                        f"<b>X:</b> {row[x_col]:.2f}<br>"
                        f"<b>Y:</b> {row[y_col]:.3f}<br>"
                        f"<b>Spectral Gap:</b> {spec_gap}<br>"
                        f"<b>Induced Norm:</b> {ind_norm}"
                    )
                    hover_texts.append(hover_text)
                
                # Use legendgroup to group all traces for this method
                # Only show legend for the first trace of each method
                fig.add_trace(go.Scatter(
                    x=morph_data[x_col],
                    y=morph_data[y_col],
                    mode='markers+lines',
                    name=method,
                    legendgroup=method,
                    showlegend=first_trace,
                    marker=dict(
                        size=[weight_to_size[w] for w in morph_data['weight_coupling']],
                        color=method_colors[method],
                        opacity=morph_data['alpha'].tolist(),
                        line=dict(width=1, color='white')
                    ),
                    line=dict(color=method_colors[method], width=2),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>'
                ))
                first_trace = False
        
        # Layout
        fig.update_layout(
            title=f"{xaxis_metric} vs Fraction Not Converged<br>ring_size=[{selected_ring_min},{selected_ring_max}], arm_size=[{selected_arm_min},{selected_arm_max}], n_oscillators={n_oscillators}",
            xaxis_title=xaxis_metric,
            yaxis_title="Fraction Not Converged",
            xaxis=dict(range=[0, 200], showgrid=True),
            yaxis=dict(range=[0, 1.0], showgrid=True),
            hovermode='closest',
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(b=120),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        point_count_text = f"<b>Total points plotted: {total_points}</b>"
        return fig, point_count_text
    
    # Pareto Front plot callback
    @app.callback(
        [Output('pareto-plot', 'figure'),
         Output('pareto-point-count', 'children')],
        [Input('method-checklist', 'value'),
         Input('ring-size-slider', 'value'),
         Input('arm-size-slider', 'value'),
         Input('weight-coupling-range-slider', 'value'),
         Input('xaxis-metric-radio', 'value'),
         Input('position-metric-radio', 'value')],
        prevent_initial_call=False
    )
    def update_pareto_plot(selected_methods, ring_range, arm_range, weight_range, xaxis_metric, position_metric):
        
        # Get selected ranges from slider indices
        ring_min_idx = ring_range[0]
        ring_max_idx = ring_range[1]
        selected_ring_min = ring_values[ring_min_idx] if ring_min_idx < len(ring_values) else ring_values[0]
        selected_ring_max = ring_values[ring_max_idx] if ring_max_idx < len(ring_values) else ring_values[-1]
        
        arm_min_idx = arm_range[0]
        arm_max_idx = arm_range[1]
        selected_arm_min = arm_values[arm_min_idx] if arm_min_idx < len(arm_values) else arm_values[0]
        selected_arm_max = arm_values[arm_max_idx] if arm_max_idx < len(arm_values) else arm_values[-1]
        
        # Get weight coupling range
        weight_min_idx = weight_range[0]
        weight_max_idx = weight_range[1]
        selected_weight_min = weight_values[weight_min_idx] if weight_min_idx < len(weight_values) else weight_values[0]
        selected_weight_max = weight_values[weight_max_idx] if weight_max_idx < len(weight_values) else weight_values[-1]
        
        # Filter by methods and ranges
        filtered = df[
            (df['method'].isin(selected_methods)) &
            (df['ring_size'] >= selected_ring_min) &
            (df['ring_size'] <= selected_ring_max) &
            (df['arm_size'] >= selected_arm_min) &
            (df['arm_size'] <= selected_arm_max) &
            (df['weight_coupling'] >= selected_weight_min) &
            (df['weight_coupling'] <= selected_weight_max)
        ].copy()
        
        if len(filtered) == 0:
            return go.Figure().add_annotation(text="No data for selected parameters"), "0 points"
        
        # Get n_oscillators from filtered data
        n_oscillators = filtered['n_oscillators'].iloc[0] if len(filtered) > 0 else 'N/A'
        
        # Compute Pareto front
        x_col = f"{position_metric}_{xaxis_metric}"
        y_col = f"{position_metric}_fraction_not_converged"
        
        points = filtered[[x_col, y_col]].values
        pareto_indices = pareto_mask_fast(points)
        
        filtered['is_pareto'] = pareto_indices
        
        # Compute opacity based on uncertainty score for selected xaxis metric
        uncertainty_col = f"uncertainty_{xaxis_metric}_fraction_not_converged"
        filtered = clip_and_normalize_uncertainty_for_metric(filtered, uncertainty_col)
        
        # Build figure
        fig = go.Figure()
        weights = sorted(filtered['weight_coupling'].unique())
        weight_to_size = {w: 10 + 30 * (i / max(len(weights) - 1, 1))
                          for i, w in enumerate(weights)}
        
        total_points = 0
        pareto_points = 0
        
        # Plot each unique combination of (method, ring_size, arm_size) separately
        # so lines only connect dots with same setup but different weights
        for method in selected_methods:
            method_data = filtered[filtered['method'] == method]
            
            if len(method_data) == 0:
                continue
            
            if x_col not in method_data.columns or y_col not in method_data.columns:
                continue
            
            # Get unique (ring_size, arm_size) combinations for this method
            unique_morphologies = method_data[['ring_size', 'arm_size']].drop_duplicates()
            
            first_trace = True
            for _, morph_row in unique_morphologies.iterrows():
                ring_sz = morph_row['ring_size']
                arm_sz = morph_row['arm_size']
                
                # Filter to this specific morphology and sort by weight
                morph_data = method_data[
                    (method_data['ring_size'] == ring_sz) &
                    (method_data['arm_size'] == arm_sz)
                ].sort_values('weight_coupling')
                
                if len(morph_data) == 0:
                    continue
                
                total_points += len(morph_data)
                pareto_data = morph_data[morph_data['is_pareto']]
                pareto_points += len(pareto_data)
                
                # Build hover text for all points
                hover_texts = []
                for _, row in morph_data.iterrows():
                    spec_gap = row.get('spectral_gap', 'N/A')
                    ind_norm = row.get('induced_norm', 'N/A')
                    if isinstance(spec_gap, float):
                        spec_gap = f"{spec_gap:.4f}"
                    if isinstance(ind_norm, float):
                        ind_norm = f"{ind_norm:.4f}"
                    
                    pareto_status = "Yes ⭐" if row['is_pareto'] else "No"
                    hover_text = (
                        f"<b>Method:</b> {row['method']}<br>"
                        f"<b>Ring Size:</b> {int(row['ring_size'])}<br>"
                        f"<b>Arm Size:</b> {int(row['arm_size'])}<br>"
                        f"<b>Weight:</b> {row['weight_coupling']}<br>"
                        f"<b>X:</b> {row[x_col]:.2f}<br>"
                        f"<b>Y:</b> {row[y_col]:.3f}<br>"
                        f"<b>Spectral Gap:</b> {spec_gap}<br>"
                        f"<b>Induced Norm:</b> {ind_norm}<br>"
                        f"<b>Pareto:</b> {pareto_status}"
                    )
                    hover_texts.append(hover_text)
                
                # Use legendgroup to group all traces for this method
                # Only show legend for the first trace of each method
                # Plot all points as circles
                fig.add_trace(go.Scatter(
                    x=morph_data[x_col],
                    y=morph_data[y_col],
                    mode='markers+lines',
                    name=method,
                    legendgroup=method,
                    showlegend=first_trace,
                    marker=dict(
                        size=[weight_to_size[w] for w in morph_data['weight_coupling']],
                        color=method_colors[method],
                        opacity=morph_data['alpha'].tolist(),
                        line=dict(width=1, color='white'),
                        symbol='circle'
                    ),
                    line=dict(color=method_colors[method], width=2),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>'
                ))
                first_trace = False
        
        # Add pareto stars overlay on top (same positions as dots)
        pareto_filtered = filtered[filtered['is_pareto']]
        if len(pareto_filtered) > 0:
            hover_texts_stars = []
            for _, row in pareto_filtered.iterrows():
                spec_gap = row.get('spectral_gap', 'N/A')
                ind_norm = row.get('induced_norm', 'N/A')
                if isinstance(spec_gap, float):
                    spec_gap = f"{spec_gap:.4f}"
                if isinstance(ind_norm, float):
                    ind_norm = f"{ind_norm:.4f}"
                
                hover_text = (
                    f"<b>Method:</b> {row['method']}<br>"
                    f"<b>Weight:</b> {row['weight_coupling']}<br>"
                    f"<b>X:</b> {row[x_col]:.2f}<br>"
                    f"<b>Y:</b> {row[y_col]:.3f}<br>"
                    f"<b>Spectral Gap:</b> {spec_gap}<br>"
                    f"<b>Induced Norm:</b> {ind_norm}<br>"
                    f"<b>Pareto:</b> Yes ⭐"
                )
                hover_texts_stars.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=pareto_filtered[x_col],
                y=pareto_filtered[y_col],
                mode='markers',
                name='Pareto-optimal',
                marker=dict(
                    size=[weight_to_size[w] + 5 for w in pareto_filtered['weight_coupling']],
                    color='gold',
                    opacity=0.7,
                    line=dict(width=2, color='gold'),
                    symbol='star'
                ),
                text=hover_texts_stars,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ))
        
        # Layout
        fig.update_layout(
            title=f"{xaxis_metric} vs Fraction Not Converged (Pareto Front)<br>ring_size=[{selected_ring_min},{selected_ring_max}], arm_size=[{selected_arm_min},{selected_arm_max}], n_oscillators={n_oscillators}",
            xaxis_title=xaxis_metric,
            yaxis_title="Fraction Not Converged",
            xaxis=dict(range=[0, 200], showgrid=True),
            yaxis=dict(range=[0, 1.0], showgrid=True),
            hovermode='closest',
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(b=120),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        pareto_point_count_text = f"<b>Total points: {total_points} | Pareto-optimal points: {pareto_points}</b>"
        return fig, pareto_point_count_text
    
    print(f"Starting Dash server on http://127.0.0.1:{port}")
    app.run(debug=True, port=port)



def plot_induced_norm_vs_spectral_gap(
    df: pd.DataFrame,
    log_scale: bool = True,
    title: str = "Spectral Gap vs Induced Norm",
    methods: Optional[List[str]] = None,
):
    """
    Scatter plot of spectral_gap vs induced_norm.
    Colors by 'method' (or legacy 'methodology'), sizes by 'n_oscillators' if present.
    """
    # Basic validation
    missing = [c for c in ["spectral_gap", "induced_norm"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    # Normalize legacy column name
    if "methodology" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"methodology": "method"})

    # Optional method filter
    if methods is not None:
        if "method" not in df.columns:
            raise ValueError("Column 'method' missing; cannot filter by methods.")
        df = df[df["method"].isin(methods)]
        if len(df) == 0:
            raise ValueError("No rows left after filtering by methods.")

    # Optional fields
    color_col = "method" if "method" in df.columns else None
    size_col = "n_oscillators" if "n_oscillators" in df.columns else None
    hover_cols = [c for c in ["method", "ring_size", "arm_size", "n_oscillators", "weight_coupling"] if c in df.columns]

    fig = px.scatter(
        df,
        x="spectral_gap",
        y="induced_norm",
        color=color_col,
        size=size_col,
        hover_data=hover_cols,
        title=title,
        labels={"spectral_gap": "Spectral Gap", "induced_norm": "Induced Norm"},
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color="white")))
    fig.update_layout(plot_bgcolor="white", legend_title_text="Method")
    
    if log_scale:
        fig.update_xaxes(type="log", showgrid=True, gridwidth=1, gridcolor="LightGray", zeroline=False)
        fig.update_yaxes(type="log", showgrid=True, gridwidth=1, gridcolor="LightGray", zeroline=False)
    else:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray", zeroline=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray", zeroline=False)
    
    fig.show()


def plot_weight_vs_norms(
    df: pd.DataFrame,
    x_min: float = 0.5,
    x_max: float = 5000.0,
    title: str = "Weights vs Spectral/Induced Norms (log-log)",
    methods: Optional[List[str]] = None,
) -> go.Figure:
    """
    Interactive plot:
    - X: weight_coupling (log scale, 0.5–5000)
    - Y: spectral norm (spectral_norm or spectral_gap) and induced_norm (log scale)
    - Color: method
    - Filter by methods if provided.
    """
    if "weight_coupling" not in df.columns:
        raise ValueError("Missing column 'weight_coupling'.")
    if "induced_norm" not in df.columns:
        raise ValueError("Missing column 'induced_norm'.")
    spectral_col = "spectral_norm" if "spectral_norm" in df.columns else (
        "spectral_gap" if "spectral_gap" in df.columns else None
    )
    if spectral_col is None:
        raise ValueError("Missing 'spectral_norm' or 'spectral_gap'.")

    if "methodology" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"methodology": "method"})
    if "method" not in df.columns:
        df["method"] = "unknown"

    # Filter by methods if provided
    if methods is not None:
        df = df[df["method"].isin(methods)]
        if len(df) == 0:
            raise ValueError("No rows left after filtering by methods.")

    df = df.copy()
    df["weight_coupling"] = pd.to_numeric(df["weight_coupling"], errors="coerce")
    df[spectral_col] = pd.to_numeric(df[spectral_col], errors="coerce")
    df["induced_norm"] = pd.to_numeric(df["induced_norm"], errors="coerce")
    df = df.dropna(subset=["weight_coupling", spectral_col, "induced_norm"])
    df = df[(df["weight_coupling"] >= x_min) & (df["weight_coupling"] <= x_max)]
    df = df[(df[spectral_col] > 0) & (df["induced_norm"] > 0)]
    if len(df) == 0:
        raise ValueError("No data within ranges and > 0 for log scales.")

    methods_unique = sorted(df["method"].unique())
    palette = px.colors.qualitative.Plotly
    method_colors = {m: palette[i % len(palette)] for i, m in enumerate(methods_unique)}

    fig = go.Figure()
    hover_cols = [c for c in ["ring_size", "arm_size", "n_oscillators", "n_couplings"] if c in df.columns]

    for m in methods_unique:
        d = df[df["method"] == m].sort_values("weight_coupling")
        if len(d) == 0:
            continue

        def mk_hover(series_y, label_y):
            return [
                (
                    f"<b>Method:</b> {row['method']}<br>"
                    f"<b>Weight:</b> {row['weight_coupling']:.4g}<br>"
                    f"<b>{label_y}:</b> {row[series_y]:.4g}"
                )
                for _, row in d.iterrows()
            ]

        fig.add_trace(go.Scatter(
            x=d["weight_coupling"], y=d[spectral_col],
            mode="markers+lines", name=f"{m} — spectral", legendgroup=m,
            marker=dict(symbol="circle", color=method_colors[m], size=8, line=dict(width=0.5, color="white")),
            line=dict(color=method_colors[m], width=2),
            text=mk_hover(spectral_col, "Spectral"), hovertemplate="%{text}<extra></extra>",
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=d["weight_coupling"], y=d["induced_norm"],
            mode="markers+lines", name=f"{m} — induced", legendgroup=m,
            marker=dict(symbol="square", color=method_colors[m], size=8, line=dict(width=0.5, color="white")),
            line=dict(color=method_colors[m], width=2, dash="dash"),
            text=mk_hover("induced_norm", "Induced"), hovertemplate="%{text}<extra></extra>",
            showlegend=True
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Weight Coupling (log)",
        yaxis_title="Norm (log)",
        plot_bgcolor="white",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
        hovermode="closest",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_xaxes(type="log", range=[math.log10(x_min), math.log10(x_max)], showgrid=True, gridcolor="LightGray")
    fig.update_yaxes(type="log", showgrid=True, gridcolor="LightGray")
    fig.show()



def plot_scatter_steps_conv_pxx_vs_log_induced_norm(
    df: pd.DataFrame,
    x_col: str = "mean_step_conv_p90",
    frac_not_conv_col: str = "mean_fraction_not_converged",
    induced_col: str = "induced_norm",
    xlim: tuple | None = (0, 200),
    ylim_induced: tuple | None = None,
    atol_zero: float = 1e-12,  # tolerance for "fully converged"
):
    """
    Two subplots:
    - Left: x = p90 steps to convergence, y = fraction_not_converged
    - Right: x = p90 steps to convergence, y = induced norm (log scale)
    Points are colored:
      - Green: fraction_not_converged == 0 (fully converged)
      - Red: otherwise
    """
    needed = [x_col, frac_not_conv_col, induced_col]
    df_plot = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

    # For log scale: keep only positive induced norms
    df_plot = df_plot[df_plot[induced_col] > 0]

    # Compute fraction converged (after filtering to keep alignment)
    fraction_not_converged = df_plot[frac_not_conv_col]

    # Color by full convergence (fraction_not_converged == 0)
    is_full = np.isclose(df_plot[frac_not_conv_col].to_numpy(), 0.0, atol=atol_zero)
    colors = np.where(is_full, "tab:green", "tab:red")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig.suptitle("p90 steps vs fraction not converged and induced norm (log y)", fontsize=14)

    # Left subplot: fraction not converged
    axes[0].scatter(df_plot[x_col], fraction_not_converged, s=40, alpha=0.7, c=colors)
    axes[0].set_xlabel("Mean step_conv_p90")
    axes[0].set_ylabel("Fraction not converged")
    axes[0].grid(True, alpha=0.3)

    # Right subplot: induced norm (log y)
    axes[1].scatter(df_plot[x_col], df_plot[induced_col], s=40, alpha=0.7, c=colors)
    axes[1].set_xlabel("Mean step_conv_p90")
    axes[1].set_ylabel("Induced norm (log)")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)

    # Optional ranges
    if xlim is not None:
        axes[0].set_xlim(*xlim)
        axes[1].set_xlim(*xlim)
    if ylim_induced is not None:
        axes[1].set_ylim(*ylim_induced)

    # Legend showing color meaning
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Fully converged (fraction_not_converged=0)",
               markerfacecolor="tab:green", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Not fully converged",
               markerfacecolor="tab:red", markersize=8),
    ]
    axes[0].legend(handles=handles, loc="best")
    axes[1].legend(handles=handles, loc="best")

    plt.tight_layout()
    plt.show()


def show_fully_converged_partition(
    df: pd.DataFrame,
    zero_col: Optional[str] = None,
    atol: float = 1e-12,
    subset_cols: Optional[List[str]] = None,
    sort_cols: Optional[List[str]] = None,
    html_path: Optional[str] = None,
    auto_open: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Filter the dataframe to setups that fully converged (fraction_not_converged == 0),
    render them to a styled HTML table, and open in a separate browser tab/window.

    Args:
        df: Input dataframe (e.g., aggregated CSV).
        zero_col: Column name for fraction_not_converged. If None, auto-detect
                  'mean_fraction_not_converged' or 'fraction_not_converged'.
        atol: Tolerance for "zero" comparison (handles tiny numerical noise).
        subset_cols: Optional list of columns to display. If None, a sensible default subset is used.
        sort_cols: Optional columns to sort the filtered result by.
        html_path: Optional output HTML path. If None, a temporary file is created.
        auto_open: If True, opens the HTML in the default browser.

    Returns:
        filtered_df, html_file_path
    """
    import webbrowser, tempfile

    # Auto-detect the fraction column
    if zero_col is None:
        if "mean_fraction_not_converged" in df.columns:
            zero_col = "mean_fraction_not_converged"
        elif "fraction_not_converged" in df.columns:
            zero_col = "fraction_not_converged"
        else:
            raise ValueError("No fraction_not_converged column found (expected 'mean_fraction_not_converged' or 'fraction_not_converged').")

    # Ensure numeric and build mask for full convergence
    ser = pd.to_numeric(df[zero_col], errors="coerce")
    mask = np.isfinite(ser) & np.isclose(ser.to_numpy(), 0.0, atol=atol)
    filtered = df.loc[mask].copy()

    # Optional sorting
    if sort_cols:
        present = [c for c in sort_cols if c in filtered.columns]
        if present:
            filtered = filtered.sort_values(present)

    # Choose columns to display
    if subset_cols is None:
        preferred_cols = [
            "method", "ring_size", "arm_size", "n_oscillators", "n_couplings",
            "weight_coupling",
            "mean_step_conv_p50", "mean_step_conv_p75", "mean_step_conv_p90", "mean_step_conv_p100",
            "median_step_conv_p50", "median_step_conv_p75", "median_step_conv_p90", "median_step_conv_p100",
            "spectral_gap", "induced_norm",
            zero_col
        ]
        subset_cols = [c for c in preferred_cols if c in filtered.columns]
        # Fallback to all columns if none of the preferred are present
        if not subset_cols:
            subset_cols = filtered.columns.tolist()

    display_df = filtered[subset_cols]

    # Build styled HTML
    table_html = display_df.to_html(index=False, border=0)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fully Converged Setups</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 16px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
  thead th {{ position: sticky; top: 0; background: #f7f7f7; z-index: 2; }}
  tr:nth-child(even) {{ background-color: #fafafa; }}
  .info {{ margin-bottom: 10px; color: #555; }}
</style>
</head>
<body>
<div class="info">
  <b>Rows:</b> {len(display_df)} &nbsp;&nbsp; <b>Columns:</b> {len(display_df.columns)}
  &nbsp;&nbsp; <b>Filtered by:</b> {zero_col} == 0 (±{atol})
</div>
{table_html}
</body>
</html>
"""

    # Write to file
    if html_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        html_path = tmp.name
        tmp.close()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Open in browser
    if auto_open:
        webbrowser.open(f"file://{os.path.abspath(html_path)}", new=2)

    return display_df, html_path


def get_impossible_morphologies(
    df: pd.DataFrame,
    zero_col: Optional[str] = None,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """
    Find morphologies (ring_size, arm_size, n_oscillators) 
    where NO entry has fraction_not_converged ≈ 0.
    
    Returns a clean DataFrame of these impossible morphologies.
    """
    # Detect fraction column
    if zero_col is None:
        if "mean_fraction_not_converged" in df.columns:
            zero_col = "mean_fraction_not_converged"
        elif "fraction_not_converged" in df.columns:
            zero_col = "fraction_not_converged"
        else:
            raise ValueError("Expected 'mean_fraction_not_converged' or 'fraction_not_converged'.")

    # Ensure needed columns
    needed = ["ring_size", "arm_size", "n_oscillators", zero_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_tmp = df.copy()
    df_tmp[zero_col] = pd.to_numeric(df_tmp[zero_col], errors="coerce")

    # Group by morphology: identify those where no entry has full convergence
    morpho_cols = ["ring_size", "arm_size", "n_oscillators"]
    grouped = df_tmp.groupby(morpho_cols, as_index=False)[zero_col].agg(
        lambda s: ~np.isclose(s.to_numpy(), 0.0, atol=atol).any()
    )
    grouped.rename(columns={zero_col: "never_converged"}, inplace=True)

    # Keep only impossible morphologies
    impossible = grouped[grouped["never_converged"]].drop(columns=["never_converged"]).reset_index(drop=True)
    
    return impossible


def plot_coupling_density_vs_norms(
    df: pd.DataFrame,
    zero_col: str | None = None,
    atol_zero: float = 1e-12,
    methods: Optional[List[str]] = None,
):
    """
    Scatter plots (both axes log-scaled):
    - X: coupling density = n_couplings / n_oscillators (log)
    - Y1: induced_norm (log)
    - Y2: spectral_gap (log)
    - Color: green if fraction_not_converged == 0, red otherwise
    - Alpha: increases with weight_coupling
    - Filter by methods if provided.
    """
    # Auto-detect fraction column
    if zero_col is None:
        if "mean_fraction_not_converged" in df.columns:
            zero_col = "mean_fraction_not_converged"
        elif "fraction_not_converged" in df.columns:
            zero_col = "fraction_not_converged"
        else:
            raise ValueError("Expected 'mean_fraction_not_converged' or 'fraction_not_converged' in df.")

    # Validate required columns
    needed = ["n_couplings", "n_oscillators", "induced_norm", "spectral_gap", "weight_coupling", zero_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Normalize legacy method name and optional filter
    if "methodology" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"methodology": "method"})
    d = df.copy()
    if "method" not in d.columns:
        d["method"] = "unknown"
    if methods is not None:
        d = d[d["method"].isin(methods)]
        if len(d) == 0:
            raise ValueError("No rows left after filtering by methods.")

    # Numeric cleanup
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=needed)
    d = d[d["n_oscillators"] > 0]

    # x-axis: coupling density
    x = d["n_couplings"] / d["n_oscillators"]

    # Color by full convergence
    is_full = np.isclose(d[zero_col].to_numpy(), 0.0, atol=atol_zero)
    base_colors = np.where(is_full, "tab:green", "tab:red")

    # Alpha by weight
    weights_ref = np.array([0.05, 0.5, 5, 50, 500, 5000], dtype=float)
    alphas_ref = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 1.0], dtype=float)
    w = d["weight_coupling"].to_numpy(dtype=float)
    idx_nearest = np.abs(w[:, None] - weights_ref[None, :]).argmin(axis=1)
    alphas = alphas_ref[idx_nearest]
    colors_rgba = [mcolors.to_rgba(c, a) for c, a in zip(base_colors, alphas)]

    # Per-method marker shapes
    methods_unique = sorted(d["method"].unique())
    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<', 'H', '*']
    marker_map = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(methods_unique)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig.suptitle("Coupling density vs norms (log-log; color by convergence; alpha by weight; shape by method)", fontsize=13)

    # Masks for valid log values
    x_vals = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    ind_vals = d["induced_norm"].to_numpy()
    spec_vals = d["spectral_gap"].to_numpy()

    # Plot per method
    for m in methods_unique:
        idx_m = np.where(d["method"] == m)[0]
        colors_rgba_m = [mcolors.to_rgba(base_colors[i], alphas[i]) for i in idx_m]

        # Induced subplot (log-log)
        mask_ind_m = (x_vals[idx_m] > 0) & (ind_vals[idx_m] > 0)
        if mask_ind_m.any():
            axes[0].scatter(
                x_vals[idx_m][mask_ind_m],
                ind_vals[idx_m][mask_ind_m],
                c=[c for c, ok in zip(colors_rgba_m, mask_ind_m) if ok],
                s=40,
                edgecolors="none",
                marker=marker_map[m],
                label=m
            )

        # Spectral subplot (log-log)
        mask_spec_m = (x_vals[idx_m] > 0) & (spec_vals[idx_m] > 0)
        if mask_spec_m.any():
            axes[1].scatter(
                x_vals[idx_m][mask_spec_m],
                spec_vals[idx_m][mask_spec_m],
                c=[c for c, ok in zip(colors_rgba_m, mask_spec_m) if ok],
                s=40,
                edgecolors="none",
                marker=marker_map[m],
                label=m
            )

    # Axes formatting
    for ax in axes:
        ax.set_xlabel("Coupling density (n_couplings / n_oscillators)")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Induced norm")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Spectral gap")
    axes[1].set_yscale("log")

    # Legends
    color_handles = [
        Line2D([0], [0], marker="o", color="w", label="Fully converged (fraction_not_converged=0)",
               markerfacecolor="tab:green", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Not fully converged",
               markerfacecolor="tab:red", markersize=8),
    ]
    method_handles = [
        Line2D([0], [0], marker=marker_map[m], color="gray", label=m,
               markerfacecolor="gray", markeredgecolor="gray", linestyle="") for m in methods_unique
    ]
    leg1 = axes[0].legend(handles=color_handles, loc="best", title="Convergence")
    axes[0].add_artist(leg1)
    axes[0].legend(handles=method_handles, loc="lower right", title="Method")

    leg1b = axes[1].legend(handles=color_handles, loc="best", title="Convergence")
    axes[1].add_artist(leg1b)
    axes[1].legend(handles=method_handles, loc="lower right", title="Method")

    plt.tight_layout()
    plt.show()

def plot_couplings_vs_weights(
    df: pd.DataFrame,
    zero_col: str | None = None,
    atol_zero: float = 1e-12,
    methods: Optional[List[str]] = None,
    normalize_x: bool = False,
    normalize_y: bool = False,
    alpha_unsuccessful: float = 0.7,  # NEW: transparency for red (unsuccessful) dots
):
    """
    Scatter plot (both axes log-scaled):
    - X: n_couplings (log) or n_couplings/n_oscillators if normalize_x=True
    - Y: weight_coupling (log) or weight_coupling/n_oscillators if normalize_y=True
    - Color: green if fraction_not_converged == 0, red otherwise
    - Size: scales with n_oscillators
    - Shape: per method
    - Filter by methods if provided.
    - alpha_unsuccessful: transparency for red dots (0.0 = invisible, 1.0 = opaque)
    """
    # Auto-detect fraction column
    if zero_col is None:
        if "mean_fraction_not_converged" in df.columns:
            zero_col = "mean_fraction_not_converged"
        elif "fraction_not_converged" in df.columns:
            zero_col = "fraction_not_converged"
        else:
            raise ValueError("Expected 'mean_fraction_not_converged' or 'fraction_not_converged' in df.")

    # Validate required columns
    needed = ["n_couplings", "weight_coupling", "n_oscillators", zero_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Normalize legacy method name and optional filter
    if "methodology" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"methodology": "method"})
    d = df.copy()
    if "method" not in d.columns:
        d["method"] = "unknown"
    if methods is not None:
        d = d[d["method"].isin(methods)]
        if len(d) == 0:
            raise ValueError("No rows left after filtering by methods.")

    # Numeric cleanup
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=needed)
    d = d[(d["n_couplings"] > 0) & (d["weight_coupling"] > 0) & (d["n_oscillators"] > 0)]

    if len(d) == 0:
        raise ValueError("No valid data after filtering (need positive n_couplings, weight_coupling, n_oscillators).")

    # X values (optionally normalized)
    if normalize_x:
        x_vals = (d["n_couplings"] / d["n_oscillators"]).to_numpy()
        x_label = "n_couplings / n_oscillators (log)"
    else:
        x_vals = d["n_couplings"].to_numpy()
        x_label = "Number of Couplings (log)"

    # Y values (optionally normalized)
    if normalize_y:
        y_vals = (d["weight_coupling"] / d["n_oscillators"]).to_numpy()
        y_label = "weight_coupling / n_oscillators (log)"
    else:
        y_vals = d["weight_coupling"].to_numpy()
        y_label = "Weight Coupling (log)"

    # Color by full convergence
    is_full = np.isclose(d[zero_col].to_numpy(), 0.0, atol=atol_zero)
    base_colors = np.where(is_full, "tab:green", "tab:red")

    # Alpha based on success/failure
    alphas = np.where(is_full, 0.7, alpha_unsuccessful)

    # Size by n_oscillators (normalize to 20–200 range)
    n_osc = d["n_oscillators"].to_numpy()
    n_osc_min, n_osc_max = n_osc.min(), n_osc.max()
    if n_osc_max > n_osc_min:
        sizes = 20 + 180 * (n_osc - n_osc_min) / (n_osc_max - n_osc_min)
    else:
        sizes = np.full(len(n_osc), 50)

    # Per-method marker shapes
    methods_unique = sorted(d["method"].unique())
    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<', 'H', '*']
    marker_map = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(methods_unique)}

    fig, ax = plt.subplots(figsize=(10, 7))
    title = "n_couplings vs weight_coupling (log-log"
    if normalize_x or normalize_y:
        title += "; normalized x" if normalize_x else ""
        title += " & " if normalize_x and normalize_y else ""
        title += "normalized y" if normalize_y else ""
    title += "; color=convergence; size=n_oscillators; shape=method)"
    fig.suptitle(title, fontsize=13)

    # Plot per method
    for m in methods_unique:
        idx_m = np.where(d["method"] == m)[0]
        ax.scatter(
            x_vals[idx_m],
            y_vals[idx_m],
            c=[base_colors[i] for i in idx_m],
            s=sizes[idx_m],
            alpha=alphas[idx_m],
            edgecolors="black",
            linewidths=0.5,
            marker=marker_map[m],
            label=m
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # Legends: convergence color + method shapes
    color_handles = [
        Line2D([0], [0], marker="o", color="w", label="Fully converged (fraction_not_converged=0)",
               markerfacecolor="tab:green", markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", label="Not fully converged",
               markerfacecolor="tab:red", markersize=10, markeredgecolor="black", alpha=alpha_unsuccessful),
    ]
    method_handles = [
        Line2D([0], [0], marker=marker_map[m], color="gray", label=m,
               markerfacecolor="gray", markeredgecolor="black", linestyle="", markersize=8)
        for m in methods_unique
    ]

    leg1 = ax.legend(handles=color_handles, loc="upper left", title="Convergence")
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, loc="upper right", title="Method")

    plt.tight_layout()
    plt.show()



def plot_norms_vs_density_and_weight(
    df: pd.DataFrame,
    methods: Optional[List[str]] = None,
    zero_col: Optional[str] = None,
    atol_zero: float = 1e-12,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    4 subplots (all log-log):
      Columns (x-axes):
        - Left: weight_coupling / n_oscillators
        - Right: n_couplings / n_oscillators
      Rows (y-axes):
        - Top: spectral_gap
        - Bottom: induced_norm
    - Filter by methods if provided
    - Color: green if fraction_not_converged == 0 (within atol), red otherwise
    - Shape: different marker per method
    """
    # Detect fraction column
    if zero_col is None:
        if "mean_fraction_not_converged" in df.columns:
            zero_col = "mean_fraction_not_converged"
        elif "fraction_not_converged" in df.columns:
            zero_col = "fraction_not_converged"
        else:
            raise ValueError("Expected 'mean_fraction_not_converged' or 'fraction_not_converged' in dataframe.")

    # Normalize legacy method name and filter
    if "methodology" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"methodology": "method"})
    d = df.copy()
    if "method" not in d.columns:
        d["method"] = "unknown"
    if methods is not None:
        d = d[d["method"].isin(methods)]
        if len(d) == 0:
            raise ValueError("No rows left after filtering by methods.")

    # Required columns
    needed = ["weight_coupling", "n_couplings", "n_oscillators", "spectral_gap", "induced_norm", zero_col]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Numeric cleanup
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=needed)
    d = d[(d["n_oscillators"] > 0)]

    if len(d) == 0:
        raise ValueError("No valid data after cleaning.")

    # Build axes data
    x_weight = (d["weight_coupling"] / d["n_oscillators"]).to_numpy()
    x_density = (d["n_couplings"] / d["n_oscillators"]).to_numpy()
    y_spec = d["spectral_gap"].to_numpy()
    y_ind = d["induced_norm"].to_numpy()

    # Colors by convergence
    is_full = np.isclose(d[zero_col].to_numpy(), 0.0, atol=atol_zero)
    base_colors = np.where(is_full, "tab:green", "tab:red")

    # Marker shapes by method
    methods_unique = sorted(d["method"].unique())
    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<', 'H', '*']
    marker_map = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(methods_unique)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
    fig.suptitle("Spectral gap / Induced norm vs (weight/n_nodes) and (couplings/n_nodes)", fontsize=13)

    # Helper to scatter with log-safe masks
    def scatter(ax, xvals, yvals, label_y: str):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(label_y)
        ax.grid(True, which="both", alpha=0.3)

        # Plot per method
        for m in methods_unique:
            idx_m = np.where(d["method"] == m)[0]
            mask = (xvals[idx_m] > 0) & (yvals[idx_m] > 0)
            if not np.any(mask):
                continue
            ax.scatter(
                xvals[idx_m][mask],
                yvals[idx_m][mask],
                c=[base_colors[i] for i, ok in zip(idx_m, mask) if ok],
                s=40,
                edgecolors="none",
                marker=marker_map[m],
                label=m,
                alpha=0.9,
            )

    # Top row: spectral gap vs x-axes
    scatter(axes[0, 0], x_weight, y_spec, "Spectral gap")
    scatter(axes[0, 1], x_density, y_spec, "Spectral gap")
    axes[0, 0].set_xlabel("weight_coupling / n_oscillators")
    axes[0, 1].set_xlabel("n_couplings / n_oscillators")

    # Bottom row: induced norm vs x-axes
    scatter(axes[1, 0], x_weight, y_ind, "Induced norm")
    scatter(axes[1, 1], x_density, y_ind, "Induced norm")
    axes[1, 0].set_xlabel("weight_coupling / n_oscillators")
    axes[1, 1].set_xlabel("n_couplings / n_oscillators")

    # Legends: convergence color + method shapes (put on bottom-right subplot)
    color_handles = [
        Line2D([0], [0], marker="o", color="w", label="Fully converged",
               markerfacecolor="tab:green", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Not fully converged",
               markerfacecolor="tab:red", markersize=8),
    ]
    method_handles = [
        Line2D([0], [0], marker=marker_map[m], color="gray", label=m,
               markerfacecolor="gray", markeredgecolor="gray", linestyle="")
        for m in methods_unique
    ]
    leg1 = axes[1, 1].legend(handles=color_handles, loc="upper left", title="Convergence")
    axes[1, 1].add_artist(leg1)
    axes[1, 1].legend(handles=method_handles, loc="lower right", title="Method")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def plot_spearman_and_log_pearson(
    df: pd.DataFrame,
    zero_ok_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 6),
    cmap: str = "vlag",
    annotate: bool = True,
    methods: Optional[List[str]] = None,  # NEW
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a correlation matrix for:
      - n_oscillators
      - n_couplings
      - weight_coupling
      - n_couplings/n_oscillators
      - weight_coupling/n_oscillators
      - spectral_gap
      - induced_norm (if present)

    Produces two heatmaps:
      - Left: Spearman correlations (original data)
      - Right: Pearson correlations on log10-transformed data
              (rows with any non-positive value across selected columns are dropped)
    """
    # Normalize legacy method column and filter (if requested)
    if "methodology" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"methodology": "method"})
    d = df.copy()
    if "method" not in d.columns:
        d["method"] = "unknown"
    if methods is not None:
        d = d[d["method"].isin(methods)]
        if len(d) == 0:
            raise ValueError("No rows left after filtering by methods.")

    # Ensure numeric base columns if present (spectral_norm removed)
    for c in ["n_oscillators", "n_couplings", "weight_coupling", "spectral_gap", "induced_norm"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Derived columns
    if all(c in d.columns for c in ["n_couplings", "n_oscillators"]):
        d["n_couplings/n_oscillators"] = d["n_couplings"] / d["n_oscillators"]
    if all(c in d.columns for c in ["weight_coupling", "n_oscillators"]):
        d["weight_coupling/n_oscillators"] = d["weight_coupling"] / d["n_oscillators"]

    # Requested columns (keep available ones)
    requested = [
        "n_oscillators",
        "n_couplings",
        "weight_coupling",
        "n_couplings/n_oscillators",
        "weight_coupling/n_oscillators",
        "spectral_gap",
        "induced_norm",
    ]
    cols = [c for c in requested if c in d.columns]
    if len(cols) < 2:
        raise ValueError(f"Not enough columns available for correlation. Found: {cols}")

    num = d[cols].apply(pd.to_numeric, errors="coerce")

    # Spearman (original)
    spearman_corr = num.corr(method="spearman")

    # Pearson on log10-transformed values (drop non-positive rows)
    positive_mask = np.ones(len(num), dtype=bool)
    for c in cols:
        positive_mask &= (num[c] > 0)
    num_pos = num[positive_mask].copy()
    if len(num_pos) == 0:
        raise ValueError("No rows have strictly positive values across all selected columns for log transform.")
    num_log = np.log10(num_pos)
    pearson_log_corr = num_log.corr(method="pearson")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Correlation matrices: Spearman (original) vs Pearson (log10-transformed)", fontsize=13)
    sns.heatmap(spearman_corr, ax=axes[0], cmap=cmap, vmin=-1, vmax=1, center=0, annot=annotate, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    axes[0].set_title("Spearman (original)")
    sns.heatmap(pearson_log_corr, ax=axes[1], cmap=cmap, vmin=-1, vmax=1, center=0, annot=annotate, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    axes[1].set_title("Pearson (log10)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return spearman_corr, pearson_log_corr


def plot_sgap_vs_ind_norm(
    df: pd.DataFrame,
    zero_col: Optional[str] = None,
    atol_zero: float = 1e-12,
    methods: Optional[List[str]] = None,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    scatter:
      - X: 1 / log10(spectral_gap)
      - Y: log10(induced_norm)
      - Color: green if fraction_not_converged == 0, red otherwise, orange if only median is converged, but not mean
      - Filter by methods if provided.

    Notes:
      - Requires spectral_gap > 0, induced_norm > 0, and log10(spectral_gap) != 0.
      - Rows failing these constraints are filtered out.
    """
    # Require both mean and median fraction columns (we'll plot three views)
    if zero_col is None:
        mean_col = "mean_fraction_not_converged"
        median_col = "median_fraction_not_converged"
    else:
        # if user passed a zero_col, still try to derive mean/median names
        mean_col = zero_col if "mean" in zero_col else "mean_fraction_not_converged"
        median_col = zero_col if "median" in zero_col else "median_fraction_not_converged"
    for col in (mean_col, median_col):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in dataframe.")

    # Normalize legacy method name and filter
    if "methodology" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"methodology": "method"})
    d = df.copy()
    if "method" not in d.columns:
        d["method"] = "unknown"
    if methods is not None:
        d = d[d["method"].isin(methods)]
        if len(d) == 0:
            raise ValueError("No rows left after filtering by methods.")

    # Validate required columns
    needed = ["spectral_gap", "induced_norm", mean_col, median_col]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=needed)
    d = d[(d["spectral_gap"] > 0) & (d["induced_norm"] > 0)]

    # Compute axes
    log_spec = np.log10(d["spectral_gap"].to_numpy())
    log_ind = np.log10(d["induced_norm"].to_numpy())


    # Filter rows where log_spec == 0 (would lead to division by zero)
    mask_valid = np.isfinite(log_spec) & np.isfinite(log_ind) & (log_spec != 0.0)
    d = d.iloc[np.where(mask_valid)[0]].copy()
    log_spec = log_spec[mask_valid]
    log_ind = log_ind[mask_valid]


    if len(d) == 0:
        raise ValueError("No valid rows after filtering for log transforms (check positive values).")

    # Prepare mean/median zero masks
    mean_zero = np.isclose(d[mean_col].to_numpy(), 0.0, atol=atol_zero)
    median_zero = np.isclose(d[median_col].to_numpy(), 0.0, atol=atol_zero)

    # Choose x-axis values depending on invert flag
    x_vals = log_spec
    xlabel = "log10(spectral_gap)"

    # Create 1x3 subplots: Mean-based, Median-based, Combined
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    titles = ["Mean criterion", "Median criterion", "Areas of $\lambda_2$ and $||L||_2$ with successful convergence"]


    # Subplot 1: mean criterion
    colors1 = np.where(mean_zero, "tab:green", "tab:red")
    axes[0].scatter(x_vals, log_ind, c=colors1, s=40, alpha=0.85, edgecolors="none", zorder=3)
    axes[0].set_title(titles[0])
    axes[0].set_xlabel(xlabel)
    axes[0].grid(True, which="both", alpha=0.3, zorder=0)
    # Legend for first subplot: mean-based
    handles1 = [
        Line2D([0], [0], marker="o", color="w", label="Converged (mean)", markerfacecolor="tab:green", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Not converged", markerfacecolor="tab:red", markersize=9),
    ]
    axes[0].legend(handles=handles1, loc="best")


    # Subplot 2: median criterion
    colors2 = np.where(median_zero, "tab:green", "tab:red")
    axes[1].scatter(x_vals, log_ind, c=colors2, s=40, alpha=0.85, edgecolors="none", zorder=3)
    axes[1].set_title(titles[1])
    axes[1].set_xlabel(xlabel)
    axes[1].grid(True, which="both", alpha=0.3, zorder=0)
    # Legend for second subplot: median-based
    handles2 = [
        Line2D([0], [0], marker="o", color="w", label="Converged (median)", markerfacecolor="tab:green", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Not converged", markerfacecolor="tab:red", markersize=9),
    ]
    axes[1].legend(handles=handles2, loc="best")


    # Subplot 3: combined criterion with 3-color mapping
    # green: both zero; red: both nonzero; orange: median==0 & mean!=0 (or asymmetric case)
    both_zero = mean_zero & median_zero
    both_nonzero = (~mean_zero) & (~median_zero)
    median_only = median_zero & (~mean_zero)
    mean_only = mean_zero & (~median_zero)
    colors3 = np.array(["tab:red"] * len(x_vals), dtype=object)
    colors3[both_zero] = "tab:green"
    colors3[median_only] = "orange"
    colors3[mean_only] = "orange"

    axes[2].scatter(x_vals, log_ind, c=colors3, s=40, alpha=0.85, edgecolors="none", zorder=3)
    axes[2].set_title(titles[2])
    axes[2].set_xlabel(xlabel)
    axes[2].grid(True, which="both", alpha=0.3, zorder=0)


    # Y label on first axis
    axes[0].set_ylabel("log10(induced_norm)")

    # Build legend for combined plot
    handles3 = [
        Line2D([0], [0], marker="o", color="w", label="Converged (mean and median)", markerfacecolor="tab:green", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Converged (median only)", markerfacecolor="orange", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Not converged", markerfacecolor="tab:red", markersize=9),
    ]

    axes[2].legend(handles=handles3, loc="best")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes



def get_sgap_ind_norm_convergence_ranges(
    df: pd.DataFrame,
    atol: float = 1e-12,
) -> Dict[str, Optional[Dict[str, Tuple[float, float]]]]:
    """
    Find candidate 2D rectangular ranges (axis-aligned) that contain only successful points.

        This function evaluates two success definitions and returns a dictionary with keys:
            - 'mean': using `mean_fraction_not_converged` (or `mean_fraction_non_converged`)
            - 'median': using `median_fraction_not_converged`

    For each definition, candidate rectangles are formed by enumerating all pairs of successful
    points (their spectral_gap and induced_norm), taking those axis-aligned rectangles that
    contain no unsuccessful points. The selected rectangle maximizes the induced_norm span
    (primary) and spectral_gap span (secondary tiebreaker).

    Returns a dict mapping keys 'mean' and 'median' to either:
        {'sgap': (spec_min, spec_max), 'ind_norm': (ind_min, ind_max)}
    or None if no candidate was found.
    """
    # acceptable column name variants
    mean_candidates = ["mean_fraction_non_converged", "mean_fraction_not_converged"]
    median_candidates = ["median_fraction_not_converged"]

    mean_col = next((c for c in mean_candidates if c in df.columns), None)
    median_col = next((c for c in median_candidates if c in df.columns), None)
    if mean_col is None and median_col is None:
        raise ValueError("Expected at least one of 'mean_fraction_non_converged'/'mean_fraction_not_converged' or 'median_fraction_not_converged' in dataframe.")

    for c in ("spectral_gap", "induced_norm"):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # Normalize numeric columns and drop NA rows
    d = df.copy()
    d["spectral_gap"] = pd.to_numeric(d["spectral_gap"], errors="coerce")
    d["induced_norm"] = pd.to_numeric(d["induced_norm"], errors="coerce")
    if mean_col is not None:
        d[mean_col] = pd.to_numeric(d[mean_col], errors="coerce")
    if median_col is not None:
        d[median_col] = pd.to_numeric(d[median_col], errors="coerce")
    drop_cols = ["spectral_gap", "induced_norm"] + ([mean_col] if mean_col is not None else []) + ([median_col] if median_col is not None else [])
    d = d.dropna(subset=drop_cols)
    if len(d) == 0:
        return {"mean": None, "median": None}

    # helper to find best rectangle for a boolean success mask (on d)
    def _best_rect_for_mask(success_mask: np.ndarray) -> Optional[Dict[str, Tuple[float, float]]]:
        # successes: unique (spectral_gap, induced_norm)
        successes = d.loc[success_mask, ["spectral_gap", "induced_norm"]].drop_duplicates()
        if len(successes) == 0:
            return None
        if len(successes) == 1:
            srow = successes.iloc[0]
            spec_v, ind_v = float(srow["spectral_gap"]), float(srow["induced_norm"])
            inside_idx = d[(d["spectral_gap"] == spec_v) & (d["induced_norm"] == ind_v)].index
            if inside_idx.empty:
                return None
            # map success_mask (aligned with d.index)
            inside_success = success_mask[np.isin(d.index.values, inside_idx.values)]
            if np.all(inside_success):
                return {"sgap": (spec_v, spec_v), "ind_norm": (ind_v, ind_v)}
            return None

        points = successes.to_numpy()
        N = points.shape[0]
        candidates: List[Tuple[float, float, float, float]] = []
        for i in range(N):
            for j in range(i + 1, N):
                s1 = points[i]
                s2 = points[j]
                spec_min, spec_max = float(min(s1[0], s2[0])), float(max(s1[0], s2[0]))
                ind_min, ind_max = float(min(s1[1], s2[1])), float(max(s1[1], s2[1]))

                inside = d[(d["spectral_gap"] >= spec_min) & (d["spectral_gap"] <= spec_max) &
                           (d["induced_norm"] >= ind_min) & (d["induced_norm"] <= ind_max)]
                if inside.empty:
                    candidates.append((spec_min, spec_max, ind_min, ind_max))
                    continue

                # check all inside succeed according to the mask (use original boolean indexing)
                # Build indices mask into d
                inside_idx = inside.index
                # map success_mask (which is aligned with d.index)
                inside_success = success_mask[np.isin(d.index.values, inside_idx.values)]
                if np.all(inside_success):
                    candidates.append((spec_min, spec_max, ind_min, ind_max))

        if not candidates:
            return None

        # choose best candidate maximizing induced_norm span, tiebreak spectral_gap span
        best = max(candidates, key=lambda r: (r[3] - r[2], r[1] - r[0]))
        return {"sgap": (best[0], best[1]), "ind_norm": (best[2], best[3])}

    results: Dict[str, Optional[Dict[str, Tuple[float, float]]]] = {"mean": None, "median": None}

    # mean-based
    if mean_col is not None:
        mean_mask = np.isclose(d[mean_col].to_numpy(), 0.0, atol=atol)
        results["mean"] = _best_rect_for_mask(mean_mask)

    # median-based
    if median_col is not None:
        median_mask = np.isclose(d[median_col].to_numpy(), 0.0, atol=atol)
        results["median"] = _best_rect_for_mask(median_mask)

    return results


def plot_sgap_vs_ind_norm_with_convergence_square(
    df: pd.DataFrame,
    atol: float = 1e-12,
    show: bool = True,
    path: Optional[str] = None,
    methods: Optional[List[str]] = None,
    ranges: Optional[Dict[str, Optional[Dict[str, Tuple[float, float]]]]] = None,
    mean_enabled: bool = True,
    median_enabled: bool = True,
    combined_enabled: bool = True,
    include_weight_arrow: bool = False,
    fontsize_title: int = 12,
    fontsize_legend: int = 10,
    fontsize_labels: int = 11,
    fontsize_textbox: int = 10,
    enable_grid: bool = True,
) -> Tuple[plt.Figure, Dict[str, Optional[Dict[str, Tuple[float, float]]]]]:
    """
    Scatter plot of log10(spectral_gap) vs log10(induced_norm) with convergence rectangles.
    
    Subplots created based on enabled flags:
      - mean_enabled: Left subplot showing mean-based convergence range
      - median_enabled: Middle subplot showing median-based convergence range
      - combined_enabled: Right subplot showing both ranges
    
    Args:
        fontsize_title: Font size for subplot titles. Defaults to 12.
        fontsize_legend: Font size for legend text. Defaults to 10.
        fontsize_labels: Font size for axis labels (xlabel, ylabel). Defaults to 11.
        fontsize_textbox: Font size for the summary text box. Defaults to 10.
    
    Returns (fig, ranges) where `ranges` is the dict returned by
    `get_sgap_ind_norm_convergence_ranges` (keys: 'mean','median').
    """
    # Build base scatter data (without showing)
    fig_base, axes_base = plot_sgap_vs_ind_norm(df, zero_col=None, atol_zero=atol, methods=methods, show=False)
    plt.close(fig_base)  # Close the temporary figure
    
    # Obtain or compute convergence ranges
    if ranges is None:
        ranges = get_sgap_ind_norm_convergence_ranges(df, atol=atol)
    
    # Count enabled subplots
    n_subplots = sum([mean_enabled, median_enabled, combined_enabled])
    if n_subplots == 0:
        raise ValueError("At least one of mean_enabled, median_enabled, or combined_enabled must be True.")
    
    # Create new figure with correct number of subplots
    fig, axes = plt.subplots(1, n_subplots, figsize=(7 * n_subplots, 6.3), sharey=True)
    if n_subplots == 1:
        axes = [axes]  # Make it a list for consistent indexing
    
    # Get data from base plot
    if "methodology" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"methodology": "method"})
    d = df.copy()
    if "method" not in d.columns:
        d["method"] = "unknown"
    if methods is not None:
        d = d[d["method"].isin(methods)]
    
    for c in ["spectral_gap", "induced_norm", "mean_fraction_not_converged", "median_fraction_not_converged"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["spectral_gap", "induced_norm", "mean_fraction_not_converged", "median_fraction_not_converged"])
    d = d[(d["spectral_gap"] > 0) & (d["induced_norm"] > 0)]
    
    log_spec = np.log10(d["spectral_gap"].to_numpy())
    log_ind = np.log10(d["induced_norm"].to_numpy())
    mask_valid = np.isfinite(log_spec) & np.isfinite(log_ind) & (log_spec != 0.0)
    log_spec = log_spec[mask_valid]
    log_ind = log_ind[mask_valid]
    d = d.iloc[np.where(mask_valid)[0]].copy()
    
    mean_zero = np.isclose(d["mean_fraction_not_converged"].to_numpy(), 0.0, atol=atol)
    median_zero = np.isclose(d["median_fraction_not_converged"].to_numpy(), 0.0, atol=atol)
    
    # Helper to draw rectangle
    def _draw_rect_on_ax(ax_obj, range_dict, linestyle="-", label=None):
        if not range_dict:
            return None
        spec_range = range_dict.get("sgap")
        ind_range = range_dict.get("ind_norm")
        if spec_range is None or ind_range is None:
            return None
        spec_min, spec_max = spec_range
        ind_min, ind_max = ind_range
        log_spec_min, log_spec_max = np.log10([spec_min, spec_max])
        log_ind_min, log_ind_max = np.log10([ind_min, ind_max])
        width = log_spec_max - log_spec_min
        height = log_ind_max - log_ind_min
        rect = Rectangle(
            (log_spec_min, log_ind_min),
            width,
            height,
            edgecolor="black",
            facecolor="none",
            linewidth=2,
            linestyle=linestyle,
            zorder=4,
            label=label,
        )
        ax_obj.add_patch(rect)
        return rect
    
    mean_range = ranges.get("mean") if isinstance(ranges, dict) else None
    median_range = ranges.get("median") if isinstance(ranges, dict) else None
    
    ax_idx = 0
    
    # Mean subplot
    if mean_enabled:
        colors = np.where(mean_zero, "tab:green", "tab:red")
        axes[ax_idx].scatter(log_spec, log_ind, c=colors, s=40, alpha=0.85, edgecolors="none", zorder=3)
        axes[ax_idx].set_title("Mean criterion", fontsize=fontsize_title)
        axes[ax_idx].set_xlabel("$log_{10}(\lambda_2)$", fontsize=fontsize_labels)
        if enable_grid:
            axes[ax_idx].grid(True, which="both", alpha=0.3, zorder=0)
        
        mean_rect = _draw_rect_on_ax(axes[ax_idx], mean_range, linestyle="-", label="100% conv range")
        handles = [
            Line2D([0], [0], marker="o", color="w", label="Converged (100%)", markerfacecolor="tab:green", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Not converged", markerfacecolor="tab:red", markersize=9),
        ]
        if mean_rect is not None:
            handles.append(mean_rect)
        axes[ax_idx].legend(handles=handles, loc="best", fontsize=fontsize_legend)
        ax_idx += 1
    
    # Median subplot
    if median_enabled:
        colors = np.where(median_zero, "tab:green", "tab:red")
        axes[ax_idx].scatter(log_spec, log_ind, c=colors, s=40, alpha=0.85, edgecolors="none", zorder=3)
        axes[ax_idx].set_title("Median criterion", fontsize=fontsize_title)
        axes[ax_idx].set_xlabel("$log_{10}(\lambda_2)$", fontsize=fontsize_labels)
        if enable_grid:
            axes[ax_idx].grid(True, which="both", alpha=0.3, zorder=0)
        
        median_rect = _draw_rect_on_ax(axes[ax_idx], median_range, linestyle="--", label="50% conv range")
        handles = [
            Line2D([0], [0], marker="o", color="w", label="Converged (50%)", markerfacecolor="tab:green", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Not converged", markerfacecolor="tab:red", markersize=9),
        ]
        if median_rect is not None:
            handles.append(median_rect)
        axes[ax_idx].legend(handles=handles, loc="best", fontsize=fontsize_legend)
        ax_idx += 1
    
    # Combined subplot
    if combined_enabled:
        both_zero = mean_zero & median_zero
        both_nonzero = (~mean_zero) & (~median_zero)
        median_only = median_zero & (~mean_zero)
        colors = np.array(["tab:red"] * len(log_spec), dtype=object)
        colors[both_zero] = "tab:green"
        colors[median_only] = "orange"
        
        axes[ax_idx].scatter(log_spec, log_ind, c=colors, s=40, alpha=0.85, edgecolors="none", zorder=3)
        axes[ax_idx].set_title("Areas of $\lambda_2$ and $||L||_2$ with successful convergence", fontsize=fontsize_title)
        axes[ax_idx].set_xlabel("$log_{10}(\lambda_2)$", fontsize=fontsize_labels)
        if enable_grid:
            axes[ax_idx].grid(True, which="both", alpha=0.3, zorder=0)
        
        mean_rect = _draw_rect_on_ax(axes[ax_idx], mean_range, linestyle="-", label="100% conv range")
        median_rect = _draw_rect_on_ax(axes[ax_idx], median_range, linestyle="--", label="50% conv range")
        handles = [
            Line2D([0], [0], marker="o", color="w", label="Converged (100%)", markerfacecolor="tab:green", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Converged ($\geq$ 50%)", markerfacecolor="orange", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Not converged (< 50%)", markerfacecolor="tab:red", markersize=9),
        ]
        if mean_rect is not None:
            handles.append(mean_rect)
        if median_rect is not None:
            handles.append(median_rect)
        axes[ax_idx].legend(handles=handles, loc="best", fontsize=fontsize_legend)
    
    # Y label on first axis
    axes[0].set_ylabel("$log_{10}(||L||_2)$", fontsize=fontsize_labels)
    
    # Add weight arrow if requested
    if include_weight_arrow:
        # Determine which axis to add the arrow to (prefer combined, then last available)
        arrow_ax = axes[-1]
        from matplotlib.patches import FancyArrowPatch
        arrow = FancyArrowPatch(
            (2.2, 0), (4.2, 2),
            arrowstyle='->', mutation_scale=20, linewidth=2, color='black',
            zorder=5
        )
        arrow_ax.add_patch(arrow)
        # Add text underneath the arrow
        arrow_ax.text(4, 0.2, "Increasing\nweight K", fontsize=fontsize_labels, ha='center',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Print ranges summary
    def _safe_get(rdict, key, subkey):
        try:
            v = rdict.get(key)
            if v is None:
                return None
            return v.get(subkey)
        except Exception:
            return None
    
    spec_median = _safe_get(ranges, "median", "sgap")
    spec_mean = _safe_get(ranges, "mean", "sgap")
    ind_median = _safe_get(ranges, "median", "ind_norm")
    ind_mean = _safe_get(ranges, "mean", "ind_norm")
    
    def _fmt_range_val(r):
        if r is None:
            return "None"
        try:
            a, b = r
            return f"({int(round(a))}, {int(round(b))})"
        except Exception:
            return str(r)
    
    msg1 = f"Ranges for $\lambda_2$ and $||L||_2$ where convergence occurs:"
    msg2 = f"50% convergence: {_fmt_range_val(ind_median)}"
    msg3 = f"100% convergence: {_fmt_range_val(ind_mean)}"
    
    print(msg1)
    print(msg2)
    print(msg3)
    
    fig.subplots_adjust(bottom=0.22)
    textbox = msg1 + "\n" + msg2 + "\n" + msg3
    fig.text(0.5, 0.02, textbox, ha="center", va="bottom", fontsize=fontsize_textbox,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    
    if path:
        fig.savefig(path)
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig, ranges


def plot_density_vs_sgap_over_ind_norm(
    df: pd.DataFrame,
    methods: Optional[List[str]] = None,
    show: bool = True,
    path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    X: coupling density = n_couplings / n_oscillators
    Y: spectral_gap / induced_norm

    Colors:
      - green: converged (mean_fraction_not_converged == 0 by default)
      - red: not converged

    Optional:
      - zero_col: override default mean column name (must contain 'mean' or pass explicit name)
      - methods: filter by methods
    """
    # choose success column

    # ensure required columns
    for c in ("n_couplings", "n_oscillators", "spectral_gap", "induced_norm"):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    d = df.copy()
    if "methodology" in d.columns and "method" not in d.columns:
        d = d.rename(columns={"methodology": "method"})
    if "method" not in d.columns:
        d["method"] = "unknown"
    if methods is not None:
        d = d[d["method"].isin(methods)]
        if len(d) == 0:
            raise ValueError("No rows left after filtering by methods.")

    # numeric cleanup
    for c in ("n_couplings", "n_oscillators", "spectral_gap", "induced_norm"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["n_couplings", "n_oscillators", "spectral_gap", "induced_norm"])

    # valid positive checks
    valid = (d["n_oscillators"] > 0) & (d["spectral_gap"] > 0) & (d["induced_norm"] > 0)
    d = d.loc[valid].copy()
    if d.empty:
        raise ValueError("No valid rows after filtering for positive values.")

    # compute axes
    # maximum possible couplings per node: (n_oscillators - 1) / 2
    max_couplings = d["n_oscillators"] * (d["n_oscillators"] - 1) / 2.0
    # ensure positive max_couplings
    valid_max = max_couplings > 0
    d = d.loc[valid_max].copy()
    if d.empty:
        raise ValueError("No valid rows after filtering for positive max_couplings.")
    # fraction of max possible couplings (clamp to [0,1])
    density_frac = (d["n_couplings"] / max_couplings.loc[d.index]).to_numpy()
    density_frac = np.clip(density_frac, 0.0, 1.0)
    ratio = (d["spectral_gap"] / d["induced_norm"]).to_numpy()


    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(density_frac, ratio, s=40, alpha=0.9, edgecolors="none", zorder=3)
    ax.set_xlabel("Fraction of max possible couplings: n_couplings / (n_osc * (n_osc-1)/2)")
    ax.set_ylabel("spectral_gap / induced_norm")
    ax.grid(True, which="both", alpha=0.3)

    # clamp axis to [0,1]
    ax.set_xlim(0.0, 1.0)
    
    # Quadratic regression
    coeffs = np.polyfit(density_frac, ratio, 2)
    quad_x = np.linspace(0.0, 1.0, 200)
    quad_y = np.polyval(coeffs, quad_x)
    ax.plot(quad_x, quad_y, label='Quadratic fit', color='green', linestyle='--', zorder=2)
    ax.legend()

    plt.tight_layout()
    if path:
        fig.savefig(path)
    if show:
        plt.show()
    return fig, ax



def summarize_successful_morphologies_by_method(
    df: pd.DataFrame,
    zero_col: Optional[str] = None,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """
    Return a summary table per method with:
      - method
      - n_successful_morphologies: number of unique morphologies that had at least one successful weight
        (morphology defined as (ring_size, arm_size) when available, otherwise (ring_size, n_oscillators))
      - min_n_oscillators: smallest n_oscillators among successful rows for that method (or NaN)
      - max_n_oscillators: largest n_oscillators among successful rows for that method (or NaN)
            - pct_successful_morphologies: percentage of successful morphologies relative to total unique
                morphologies (based on the same morphology columns) in the input dataframe.
    """
    # choose zero column
    mean_candidates = ["mean_fraction_non_converged", "mean_fraction_not_converged"]
    if zero_col is None:
        zero_col = next((c for c in mean_candidates if c in df.columns), None)
    if zero_col is None or zero_col not in df.columns:
        raise ValueError(f"Expected one of {mean_candidates} in dataframe (or pass zero_col).")

    # normalize method column
    d = df.copy()
    if "methodology" in d.columns and "method" not in d.columns:
        d = d.rename(columns={"methodology": "method"})
    if "method" not in d.columns:
        d["method"] = "unknown"

    # choose morphology definition
    if "ring_size" in d.columns and "arm_size" in d.columns:
        morph_cols = ["ring_size", "arm_size"]
    elif "ring_size" in d.columns and "n_oscillators" in d.columns:
        morph_cols = ["ring_size", "n_oscillators"]
    else:
        raise ValueError("Expected columns for morphology: prefer ('ring_size','arm_size') or ('ring_size','n_oscillators').")

    # ensure numeric where needed
    num_cols = set(morph_cols) | {"n_oscillators", zero_col}
    for c in num_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=[zero_col] + [c for c in morph_cols if c in d.columns])
    if "n_oscillators" not in d.columns:
        d["n_oscillators"] = np.nan

    # total unique morphologies in the original dataframe (based on morph_cols)
    total_morphs = df[list(morph_cols)].dropna().drop_duplicates().shape[0]

    # success mask
    d["__success__"] = np.isclose(d[zero_col].to_numpy(), 0.0, atol=atol)

    rows = []
    for method, grp in d.groupby("method", sort=True):
        succ = grp[grp["__success__"]]
        if succ.empty:
            rows.append({
                "method": method,
                "n_successful_morphologies": 0,
                "min_n_oscillators": np.nan,
                "max_n_oscillators": np.nan,
            })
            continue

        # count unique morphologies (based on morph_cols)
        unique_morph = succ[morph_cols].drop_duplicates()
        n_morph = int(len(unique_morph))

        # min/max n_oscillators among successful rows
        if succ["n_oscillators"].dropna().empty:
            min_n = np.nan
            max_n = np.nan
        else:
            min_n = int(succ["n_oscillators"].min())
            max_n = int(succ["n_oscillators"].max())

        rows.append({
            "method": method,
            "n_successful_morphologies": n_morph,
            "pct_successful_morphologies": np.round((n_morph / total_morphs * 100.0), 2) if total_morphs > 0 else np.nan,
            "min_n_oscillators": min_n,
            "max_n_oscillators": max_n,
        })

    out_df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    return out_df


# ==============================



if __name__ == "__main__":
    rng = jax.random.PRNGKey(1)
    mu = 10.0
    nvar = 10
    cv_range = (0.4, 0.4)
    ring_range = (15, 15)
    arm_range = (10, 10)
    sample_size = 100
    samples = gamma_with_cv(rng, mu, cv=0.2, sample_size=sample_size)
    print(f"mean samples: {jnp.mean(samples)}, std samples: {jnp.std(samples)}, cv samples: {jnp.std(samples)/jnp.mean(samples)}")

    ring_setup_list, parameter_variation_list = generate_ring_setup_variations(
        rng=rng,
        cpg_type="equal_arms",
        nvar=nvar,
        range_of_variation=arm_range,
        mu_gamma=mu
    )

    for i in range(nvar):
        ring_setup = ring_setup_list[i]
        variational_param = parameter_variation_list[i]
        print(f"Ring setup variation {i}: parameter: {variational_param:.3f} - ring setup: {ring_setup}")
    print("Done.")


    # ==============================
    # tmp directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, "../../tmp/")
    tmp_dir = os.path.abspath(tmp_dir)  # Normalize the path
    print(f"tmp_dir: {tmp_dir}")

    if os.path.exists(os.path.join(tmp_dir, "test_analysis_utils.csv")):
        os.remove(os.path.join(tmp_dir, "test_analysis_utils.csv"))

    header = ["methodology", "morphology_type", "morphology_parameter", "weight_coupling", "run_id", "steps_to_convergence", "fraction_not_converged"]
    csv_file_path = os.path.join(tmp_dir, "test_analysis_utils.csv")
    create_csv(csv_file_path, header)
    print(f"Created CSV file at: {csv_file_path}")

    entry = {
        "methodology": "base",
        "morphology_type": "varying_arms",
        "morphology_parameter": 0.2,
        "weight_coupling": 0.5,
        "run_id": 1,
        "steps_to_convergence": 150.0,
        "fraction_not_converged": 0.0
    }

    add_csv_entry(csv_file_path, entry)


    # Pareto plot testing
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path for testing
        csv_path = "../../experiments/b01_r04/numerical_metrics/convergence_results_aggregated.csv"
    
    create_interactive_convergence_plot(csv_path, port=8050)






    

