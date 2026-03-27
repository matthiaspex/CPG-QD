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
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import seaborn as sns

from scipy.stats import chi2



def aggregate_convergence_csv(csv_path: str) -> None:
    """
    Aggregate the new-format convergence CSV by experimental setup:
    (morphology_size, ratio_couplings_oscillators, weight_coupling).

    For each setup, compute mean, std (ddof=1), and median for:
      - step_conv_p50, step_conv_p75, step_conv_p90, step_conv_p100
      - fraction_not_converged

    Also compute pairwise uncertainty score between each step_conv_pXX and fraction_not_converged:
      uncertainty = sqrt( (std_step/mean_step)^2 + (std_frac/mean_frac)^2 )
      with NaN/Inf handled via np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)

    Writes: "<dir>/convergence_results_aggregated.csv"
    """
    df = pd.read_csv(csv_path)

    groupby_cols = ["morphology_size", "ratio_couplings_oscillators", "weight_coupling"]
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

    # Compute actual ratio using carried constants; guard against divide-by-zero
    if "n_couplings" in aggregated.columns and "n_oscillators" in aggregated.columns:
        num = aggregated["n_couplings"].astype(float).to_numpy()
        den = aggregated["n_oscillators"].astype(float).to_numpy()
        ratio = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0)
        aggregated["actual_ratio_couplings_oscillators"] = ratio
        aggregated["actual_ratio_couplings_oscillators"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Place actual ratio right after the reported ratio column
    if (
        "actual_ratio_couplings_oscillators" in aggregated.columns
        and "ratio_couplings_oscillators" in aggregated.columns
    ):
        cols = list(aggregated.columns)
        cols.remove("actual_ratio_couplings_oscillators")
        insert_at = cols.index("ratio_couplings_oscillators") + 1
        cols = cols[:insert_at] + ["actual_ratio_couplings_oscillators"] + cols[insert_at:]
        aggregated = aggregated[cols]

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


def plot_convergence_scatter(
    df: pd.DataFrame,
    use_log_scale: bool = False,
    atol_zero: float = 1e-6,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    save_path: Optional[str] = None,
    only_converged: bool = False,
    ratio_selection: Optional[List[Any]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of ratio_couplings_oscillators vs weight_coupling,
    colored by convergence status and with transparency scaled by morphology_size.
    
    Args:
        df: DataFrame with aggregated convergence results
        use_log_scale: If True, use log scales for axes (no pre-transformation).
                       If False (default), transform values before plotting.
        atol_zero: Tolerance for considering mean_fraction_not_converged as zero
        figsize: Figure size tuple
        show: If True (default), display the plot
        save_path: If provided, save the figure to this path
        only_converged: If True, display only fully converged (green) points
        ratio_selection: Optional list of ratio_couplings_oscillators values to include
    
    Returns:
        fig, ax: matplotlib Figure and Axes objects
    """
    # Validate required columns
    required = [
        "ratio_couplings_oscillators",
        "actual_ratio_couplings_oscillators",
        "weight_coupling",
        "mean_fraction_not_converged",
        "morphology_size",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Clean data
    d = df.copy()

    # Optional ratio filter with tolerant numeric matching and string support
    if ratio_selection is not None:
        ratios_series = d["ratio_couplings_oscillators"]

        effective_series = ratios_series.copy()
        if "actual_ratio_couplings_oscillators" in d.columns:
            mask_max = ratios_series.astype(str) == "max"
            effective_series.loc[mask_max] = d.loc[mask_max, "actual_ratio_couplings_oscillators"]

        mask = pd.Series(False, index=d.index)

        numeric_targets: List[float] = []
        string_targets = set()
        for r in ratio_selection:
            try:
                numeric_targets.append(float(r))
            except (ValueError, TypeError):
                string_targets.add(str(r))

        if numeric_targets:
            ratio_numeric = pd.to_numeric(effective_series, errors="coerce")
            num_mask = np.zeros(len(d), dtype=bool)
            for t in numeric_targets:
                num_mask |= np.isclose(ratio_numeric, t, rtol=1e-8, atol=1e-10)
            mask |= num_mask

        if string_targets:
            mask |= ratios_series.astype(str).isin(string_targets)

        d = d[mask]
        if len(d) == 0:
            available = sorted(ratios_series.dropna().unique())
            raise ValueError(
                f"No data after applying ratio_selection filter. Available ratios: {available}"
            )

    d = d.dropna(subset=required)

    # Optionally keep only fully converged points
    is_converged_mask = np.isclose(d["mean_fraction_not_converged"].to_numpy(), 0.0, atol=atol_zero)
    if only_converged:
        d = d[is_converged_mask]
        if len(d) == 0:
            raise ValueError("No fully converged points to plot after filtering")
        # After filtering, everything is converged
        is_converged_mask = np.ones(len(d), dtype=bool)

    # Prepare x and y values
    if use_log_scale:
        # Use raw values, matplotlib will handle the log scaling
        x_vals = d["actual_ratio_couplings_oscillators"].to_numpy()
        y_vals = d["weight_coupling"].to_numpy()
        x_label = "ratio_couplings_oscillators"
        y_label = "weight_coupling"
    else:
        # Pre-transform values
        x_vals = np.log2(d["actual_ratio_couplings_oscillators"].to_numpy())
        y_vals = np.log10(d["weight_coupling"].to_numpy())
        x_label = "log₂(ratio_couplings_oscillators)"
        y_label = "log₁₀(weight_coupling)"
    
    # Color by convergence: green if mean_fraction_not_converged == 0, red otherwise
    colors = np.where(is_converged_mask, "tab:green", "tab:red")
    

    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(
        x_vals,
        y_vals,
        c=colors,
        s=50,
        edgecolors="black",
        linewidths=0.5,
    )
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    if use_log_scale:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title("Convergence Results (log scales)", fontsize=13)
    else:
        ax.set_title("Convergence Results (pre-transformed)", fontsize=13)
    
    ax.grid(True, alpha=0.3, which="both")
    
    # Legend for colors
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Fully converged",
               markerfacecolor="tab:green", markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", label="Not fully converged",
               markerfacecolor="tab:red", markersize=10, markeredgecolor="black"),
    ]
    ax.legend(handles=handles, loc="best", title="Convergence Status")
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax



def plot_scatter_steps_conv_pxx_vs_log_induced_norm(
    df: pd.DataFrame,
    x_col: str = "mean_step_conv_p90",
    frac_not_conv_col: str = "mean_fraction_not_converged",
    induced_col: str = "induced_norm",
    xlim: tuple | None = (0, 200),
    ylim_induced: tuple | None = None,
    atol_zero: float = 1e-12,  # tolerance for "fully converged"
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Scatter plot:
    - x = p90 steps to convergence, y = induced norm (log scale)
    Points are colored:
      - Green: fraction_not_converged == 0 (fully converged)
      - Red: otherwise
    
    Args:
        df: DataFrame with aggregated convergence results
        x_col: Column name for x-axis (convergence step metric)
        frac_not_conv_col: Column name for fraction not converged
        induced_col: Column name for induced norm
        xlim: Optional x-axis limits
        ylim_induced: Optional y-axis limits for induced norm subplot
        atol_zero: Tolerance for considering fraction_not_converged as zero
        show: If True (default), display the plot
        save_path: If provided, save the figure to this path
    """
    needed = [x_col, frac_not_conv_col, induced_col]
    df_plot = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

    # For log scale: keep only positive induced norms
    df_plot = df_plot[df_plot[induced_col] > 0]

    # Color by full convergence (fraction_not_converged == 0)
    is_full = np.isclose(df_plot[frac_not_conv_col].to_numpy(), 0.0, atol=atol_zero)
    colors = np.where(is_full, "tab:green", "tab:red")

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("p90 steps vs induced norm (log y)", fontsize=14)

    # Plot: induced norm (log y)
    ax.scatter(df_plot[x_col], df_plot[induced_col], s=40, alpha=0.7, c=colors)
    ax.set_xlabel("Mean step_conv_p90")
    ax.set_ylabel("Induced norm (log)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # Optional ranges
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim_induced is not None:
        ax.set_ylim(*ylim_induced)

    # Legend showing color meaning
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Fully converged (fraction_not_converged=0)",
               markerfacecolor="tab:green", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Not fully converged",
               markerfacecolor="tab:red", markersize=8),
    ]
    ax.legend(handles=handles, loc="best")

    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


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
    Interactive Plotly/Dash visualization for condensed convergence results.
    
    Features:
    - Tab 1: Regular plot with all metrics
      - Slider for morphology_size
      - Slider for ratio_couplings_oscillators
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
    
    step_metrics = ['step_conv_p50', 'step_conv_p75', 'step_conv_p90', 'step_conv_p100']
    position_options = ['mean', 'median']
    
    # Get unique values for sliders and checklists
    morph_values = sorted(df['morphology_size'].dropna().unique())
    # Sort ratios numerically when possible, place non-numeric (e.g., 'max') after numeric
    _raw_ratio_values = df['ratio_couplings_oscillators'].dropna().unique()
    def _ratio_sort_key(v):
        try:
            return (0, float(v))
        except (ValueError, TypeError):
            return (1, str(v))
    ratio_values = sorted(_raw_ratio_values, key=_ratio_sort_key)
    weight_values = sorted(df['weight_coupling'].dropna().unique())
    
    # Create color map for morphology sizes
    morph_colors = {m: c for m, c in zip(morph_values, 
                                          px.colors.qualitative.Plotly[:len(morph_values)])}
    
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        dbc.Row([dbc.Col(html.H1("CPG Convergence Analysis (Condensed)",
                                 className="text-center mb-4"))]),
        dbc.Row([
            dbc.Col([
                html.H4("Controls", className="mb-3"),
                
                # Morphology Size checklist (multi-select)
                html.Label("Morphology Size:", className="fw-bold mt-3"),
                html.Div(
                    dbc.ButtonGroup([
                        dbc.Button("Enable All", id='morphology-enable-all', color='primary', size='sm', className='me-2'),
                        dbc.Button("Disable All", id='morphology-disable-all', color='secondary', size='sm')
                    ]),
                    className='mb-2'
                ),
                dcc.Checklist(
                    id='morphology-size-checklist',
                    options=[{'label': f"{int(m)}", 'value': m} for m in morph_values],
                    value=morph_values[:1] if len(morph_values) > 0 else [],
                    labelStyle={'display': 'block', 'margin': '5px'}
                ),
                
                html.Hr(),
                
                # Ratio Couplings/Oscillators checklist (multi-select)
                html.Label("Ratio Couplings/Oscillators:", className="fw-bold mt-3"),
                html.Div(
                    dbc.ButtonGroup([
                        dbc.Button("Enable All", id='ratio-enable-all', color='primary', size='sm', className='me-2'),
                        dbc.Button("Disable All", id='ratio-disable-all', color='secondary', size='sm')
                    ]),
                    className='mb-2'
                ),
                dcc.Checklist(
                    id='ratio-checklist',
                    options=[{'label': (f"{float(r):.2f}" if isinstance(r, (int, float, np.number)) else str(r)), 'value': r} for r in ratio_values],
                    value=ratio_values[:1] if len(ratio_values) > 0 else [],
                    labelStyle={'display': 'block', 'margin': '5px'}
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
                
            ], width=3, className="bg-light p-3", style={"maxHeight": "88vh", "overflowY": "auto"}),
            
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
    
    # Toggle buttons for Morphology Size checklist
    @app.callback(
        Output('morphology-size-checklist', 'value'),
        [Input('morphology-enable-all', 'n_clicks'), Input('morphology-disable-all', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_morphology(enable_clicks, disable_clicks):
        trigger = ctx.triggered_id
        if trigger == 'morphology-enable-all':
            return morph_values
        if trigger == 'morphology-disable-all':
            return []
        return no_update

    # Toggle buttons for Ratio checklist
    @app.callback(
        Output('ratio-checklist', 'value'),
        [Input('ratio-enable-all', 'n_clicks'), Input('ratio-disable-all', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_ratios(enable_clicks, disable_clicks):
        trigger = ctx.triggered_id
        if trigger == 'ratio-enable-all':
            return ratio_values
        if trigger == 'ratio-disable-all':
            return []
        return no_update

    # Main plot callback - All Data Tab
    @app.callback(
        [Output('convergence-plot', 'figure'),
         Output('point-count', 'children')],
        [Input('morphology-size-checklist', 'value'),
         Input('ratio-checklist', 'value'),
         Input('weight-coupling-range-slider', 'value'),
         Input('xaxis-metric-radio', 'value'),
         Input('position-metric-radio', 'value')],
        prevent_initial_call=False
    )
    def update_plot(selected_morphs, selected_ratios, weight_range, xaxis_metric, position_metric):
        
        # Handle empty selections
        if not selected_morphs:
            return go.Figure().add_annotation(text="No morphology sizes selected"), "0 points"
        if not selected_ratios:
            return go.Figure().add_annotation(text="No ratios selected"), "0 points"
        
        # Get weight coupling range
        weight_min_idx = weight_range[0]
        weight_max_idx = weight_range[1]
        selected_weight_min = weight_values[weight_min_idx] if weight_min_idx < len(weight_values) else weight_values[0]
        selected_weight_max = weight_values[weight_max_idx] if weight_max_idx < len(weight_values) else weight_values[-1]
        
        # Filter by selected morphology sizes, selected ratios, and weight_coupling range
        filtered = df[
            (df['morphology_size'].isin(selected_morphs)) &
            (df['ratio_couplings_oscillators'].isin(selected_ratios)) &
            (df['weight_coupling'] >= selected_weight_min) &
            (df['weight_coupling'] <= selected_weight_max)
        ].copy()
        
        if len(filtered) == 0:
            return go.Figure().add_annotation(text="No data for selected parameters"), "0 points"
        
        # Get n_oscillators and n_couplings from filtered data
        n_oscillators = filtered['n_oscillators'].iloc[0] if len(filtered) > 0 else 'N/A'
        n_couplings = filtered['n_couplings'].iloc[0] if len(filtered) > 0 else 'N/A'
        
        # Compute alpha based on ratio_couplings_oscillators, normalized to [0.2, 1.0]
        # Convert to numeric, handling string values like 'max'
        ratio_vals = pd.to_numeric(filtered['ratio_couplings_oscillators'], errors='coerce').to_numpy()
        ratio_vals_valid = ratio_vals[~np.isnan(ratio_vals)]
        # Default alpha for all rows
        filtered['alpha'] = 0.6
        if len(ratio_vals_valid) > 1:
            ratio_min, ratio_max = ratio_vals_valid.min(), ratio_vals_valid.max()
            if ratio_max > ratio_min:
                mask = ~np.isnan(ratio_vals)
                scaled = 0.2 + 0.8 * (ratio_vals[mask] - ratio_min) / (ratio_max - ratio_min)
                scaled = np.clip(scaled, 0.2, 1.0)
                filtered.loc[mask, 'alpha'] = scaled
        
        # Build figure
        fig = go.Figure()
        weights = sorted(filtered['weight_coupling'].unique())
        weight_to_size = {w: 10 + 30 * (i / max(len(weights) - 1, 1))
                          for i, w in enumerate(weights)}
        
        # X and Y column names
        x_col = f"{position_metric}_{xaxis_metric}"
        y_col = f"{position_metric}_fraction_not_converged"
        
        if x_col not in filtered.columns or y_col not in filtered.columns:
            return go.Figure().add_annotation(text="Required columns not found"), "0 points"
        
        total_points = 0
        
        # Plot each combination of morphology_size and ratio separately
        for morph_size in selected_morphs:
            for ratio in selected_ratios:
                # Filter for this specific morphology and ratio
                subset = filtered[
                    (filtered['morphology_size'] == morph_size) &
                    (filtered['ratio_couplings_oscillators'] == ratio)
                ].copy()
                
                if len(subset) == 0:
                    continue
                
                # Sort by weight coupling for line connectivity
                subset = subset.sort_values('weight_coupling')
                total_points += len(subset)
                
                # Build hover text
                hover_texts = []
                for _, row in subset.iterrows():
                    spec_gap = row.get('spectral_gap', 'N/A')
                    ind_norm = row.get('induced_norm', 'N/A')
                    if isinstance(spec_gap, float):
                        spec_gap = f"{spec_gap:.4f}"
                    if isinstance(ind_norm, float):
                        ind_norm = f"{ind_norm:.4f}"
                    
                    ratio_val = row.get('ratio_couplings_oscillators', 'N/A')
                    if isinstance(ratio_val, float):
                        ratio_str = f"{ratio_val:.2f}"
                    else:
                        ratio_str = str(ratio_val)
                    
                    hover_text = (
                        f"<b>Morphology Size:</b> {int(row['morphology_size'])}<br>"
                        f"<b>Ratio:</b> {ratio_str}<br>"
                        f"<b>Weight:</b> {row['weight_coupling']}<br>"
                        f"<b>X:</b> {row[x_col]:.2f}<br>"
                        f"<b>Y:</b> {row[y_col]:.3f}<br>"
                        f"<b>Spectral Gap:</b> {spec_gap}<br>"
                        f"<b>Induced Norm:</b> {ind_norm}"
                    )
                    hover_texts.append(hover_text)
                
                # Build a robust ratio label for legend name
                ratio_label = f"{float(ratio):.2f}" if isinstance(ratio, (int, float, np.number)) else str(ratio)
                # Resolve a color for this morphology, tolerating type differences
                color_candidate = morph_colors.get(morph_size)
                if color_candidate is None:
                    color_candidate = morph_colors.get(str(morph_size))
                if color_candidate is None:
                    try:
                        color_candidate = morph_colors.get(float(morph_size))
                    except Exception:
                        color_candidate = None
                if color_candidate is None:
                    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
                    color_candidate = palette[hash(str(morph_size)) % len(palette)]
                fig.add_trace(go.Scatter(
                    x=subset[x_col],
                    y=subset[y_col],
                    mode='markers+lines',
                    name=f'Morph {int(morph_size)}, Ratio {ratio_label}',
                    marker=dict(
                        size=[weight_to_size[w] for w in subset['weight_coupling']],
                        color=color_candidate,
                        opacity=subset['alpha'].tolist(),
                        line=dict(width=1, color='white')
                    ),
                    line=dict(color=color_candidate, width=2),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=f'morph_{morph_size}',
                    showlegend=False
                ))
        
        # Add legend-only opaque markers per morphology size (avoid varying transparency in legend)
        unique_morphs = sorted(filtered['morphology_size'].unique())
        for morph_size in unique_morphs:
            color_candidate = morph_colors.get(morph_size)
            if color_candidate is None:
                color_candidate = morph_colors.get(str(morph_size))
            if color_candidate is None:
                try:
                    color_candidate = morph_colors.get(float(morph_size))
                except Exception:
                    color_candidate = None
            if color_candidate is None:
                palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
                color_candidate = palette[hash(str(morph_size)) % len(palette)]
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                name=f'Morph {int(morph_size)}',
                marker=dict(size=12, color=color_candidate, opacity=1.0, line=dict(width=1, color='white')),
                legendgroup=f'morph_{morph_size}',
                showlegend=True,
                visible='legendonly'
            ))

        # Layout
        # Prepare a safe ratios label list without forcing numeric conversion
        ratio_labels = [
            (f"{float(r):.2f}" if isinstance(r, (int, float, np.number)) else str(r))
            for r in selected_ratios
        ]
        fig.update_layout(
            title=f"{xaxis_metric} vs Fraction Not Converged<br>morphology_sizes={sorted([int(m) for m in selected_morphs])}, ratios={ratio_labels}",
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
        [Input('morphology-size-checklist', 'value'),
         Input('ratio-checklist', 'value'),
         Input('weight-coupling-range-slider', 'value'),
         Input('xaxis-metric-radio', 'value'),
         Input('position-metric-radio', 'value')],
        prevent_initial_call=False
    )
    def update_pareto_plot(selected_morphs, selected_ratios, weight_range, xaxis_metric, position_metric):
        
        # Handle empty selections
        if not selected_morphs:
            return go.Figure().add_annotation(text="No morphology sizes selected"), "0 points"
        if not selected_ratios:
            return go.Figure().add_annotation(text="No ratios selected"), "0 points"
        
        # Get weight coupling range
        weight_min_idx = weight_range[0]
        weight_max_idx = weight_range[1]
        selected_weight_min = weight_values[weight_min_idx] if weight_min_idx < len(weight_values) else weight_values[0]
        selected_weight_max = weight_values[weight_max_idx] if weight_max_idx < len(weight_values) else weight_values[-1]
        
        # Filter by selected morphology sizes, selected ratios, and weight_coupling range
        filtered = df[
            (df['morphology_size'].isin(selected_morphs)) &
            (df['ratio_couplings_oscillators'].isin(selected_ratios)) &
            (df['weight_coupling'] >= selected_weight_min) &
            (df['weight_coupling'] <= selected_weight_max)
        ].copy()
        
        if len(filtered) == 0:
            return go.Figure().add_annotation(text="No data for selected parameters"), "0 points"
        
        # Get n_oscillators and n_couplings from filtered data
        n_oscillators = filtered['n_oscillators'].iloc[0] if len(filtered) > 0 else 'N/A'
        n_couplings = filtered['n_couplings'].iloc[0] if len(filtered) > 0 else 'N/A'
        
        # Compute Pareto front
        x_col = f"{position_metric}_{xaxis_metric}"
        y_col = f"{position_metric}_fraction_not_converged"
        
        if x_col not in filtered.columns or y_col not in filtered.columns:
            return go.Figure().add_annotation(text="Required columns not found"), "0 points"
        
        points = filtered[[x_col, y_col]].values
        pareto_indices = pareto_mask_fast(points)
        
        filtered['is_pareto'] = pareto_indices
        
        # Compute alpha based on ratio_couplings_oscillators, normalized to [0.2, 1.0]
        # Convert to numeric, handling string values like 'max'
        ratio_vals = pd.to_numeric(filtered['ratio_couplings_oscillators'], errors='coerce').to_numpy()
        ratio_vals_valid = ratio_vals[~np.isnan(ratio_vals)]
        # Default alpha for all rows
        filtered['alpha'] = 0.6
        if len(ratio_vals_valid) > 1:
            ratio_min, ratio_max = ratio_vals_valid.min(), ratio_vals_valid.max()
            if ratio_max > ratio_min:
                mask = ~np.isnan(ratio_vals)
                scaled = 0.2 + 0.8 * (ratio_vals[mask] - ratio_min) / (ratio_max - ratio_min)
                scaled = np.clip(scaled, 0.2, 1.0)
                filtered.loc[mask, 'alpha'] = scaled
        
        # Build figure
        fig = go.Figure()
        weights = sorted(filtered['weight_coupling'].unique())
        weight_to_size = {w: 10 + 30 * (i / max(len(weights) - 1, 1))
                          for i, w in enumerate(weights)}
        
        total_points = 0
        pareto_points = 0
        
        # Plot each combination of morphology_size and ratio separately
        for morph_size in selected_morphs:
            for ratio in selected_ratios:
                # Filter for this specific morphology and ratio
                subset = filtered[
                    (filtered['morphology_size'] == morph_size) &
                    (filtered['ratio_couplings_oscillators'] == ratio)
                ].copy()
                
                if len(subset) == 0:
                    continue
                
                # Sort by weight coupling for line connectivity
                subset = subset.sort_values('weight_coupling')
                total_points += len(subset)
                pareto_points += int(subset['is_pareto'].sum())
                
                # Build hover text for all points
                hover_texts = []
                for _, row in subset.iterrows():
                    spec_gap = row.get('spectral_gap', 'N/A')
                    ind_norm = row.get('induced_norm', 'N/A')
                    if isinstance(spec_gap, float):
                        spec_gap = f"{spec_gap:.4f}"
                    if isinstance(ind_norm, float):
                        ind_norm = f"{ind_norm:.4f}"
                    
                    ratio_val = row.get('ratio_couplings_oscillators', 'N/A')
                    if isinstance(ratio_val, float):
                        ratio_str = f"{ratio_val:.2f}"
                    else:
                        ratio_str = str(ratio_val)
                    
                    pareto_status = "Yes ⭐" if row['is_pareto'] else "No"
                    hover_text = (
                        f"<b>Morphology Size:</b> {int(row['morphology_size'])}<br>"
                        f"<b>Ratio:</b> {ratio_str}<br>"
                        f"<b>Weight:</b> {row['weight_coupling']}<br>"
                        f"<b>X:</b> {row[x_col]:.2f}<br>"
                        f"<b>Y:</b> {row[y_col]:.3f}<br>"
                        f"<b>Spectral Gap:</b> {spec_gap}<br>"
                        f"<b>Induced Norm:</b> {ind_norm}<br>"
                        f"<b>Pareto:</b> {pareto_status}"
                    )
                    hover_texts.append(hover_text)
                
                # Plot all points as circles
                ratio_label = f"{float(ratio):.2f}" if isinstance(ratio, (int, float, np.number)) else str(ratio)
                # Resolve a color for this morphology, tolerating type differences
                color_candidate = morph_colors.get(morph_size)
                if color_candidate is None:
                    color_candidate = morph_colors.get(str(morph_size))
                if color_candidate is None:
                    try:
                        color_candidate = morph_colors.get(float(morph_size))
                    except Exception:
                        color_candidate = None
                if color_candidate is None:
                    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
                    color_candidate = palette[hash(str(morph_size)) % len(palette)]
                fig.add_trace(go.Scatter(
                    x=subset[x_col],
                    y=subset[y_col],
                    mode='markers+lines',
                    name=f'Morph {int(morph_size)}, Ratio {ratio_label}',
                    marker=dict(
                        size=[weight_to_size[w] for w in subset['weight_coupling']],
                        color=color_candidate,
                        opacity=subset['alpha'].tolist(),
                        line=dict(width=1, color='white'),
                        symbol='circle'
                    ),
                    line=dict(color=color_candidate, width=2),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=f'morph_{morph_size}',
                    showlegend=False
                ))
        
        # Add pareto stars overlay on top
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
                
                ratio_val = row.get('ratio_couplings_oscillators', 'N/A')
                if isinstance(ratio_val, float):
                    ratio_str = f"{ratio_val:.2f}"
                else:
                    ratio_str = str(ratio_val)
                
                hover_text = (
                    f"<b>Morphology Size:</b> {int(row['morphology_size'])}<br>"
                    f"<b>Ratio:</b> {ratio_str}<br>"
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

        # Add legend-only opaque markers per morphology size (avoid varying transparency in legend)
        unique_morphs = sorted(filtered['morphology_size'].unique())
        for morph_size in unique_morphs:
            color_candidate = morph_colors.get(morph_size)
            if color_candidate is None:
                color_candidate = morph_colors.get(str(morph_size))
            if color_candidate is None:
                try:
                    color_candidate = morph_colors.get(float(morph_size))
                except Exception:
                    color_candidate = None
            if color_candidate is None:
                palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
                color_candidate = palette[hash(str(morph_size)) % len(palette)]
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                name=f'Morph {int(morph_size)}',
                marker=dict(size=12, color=color_candidate, opacity=1.0, line=dict(width=1, color='white')),
                legendgroup=f'morph_{morph_size}',
                showlegend=True,
                visible='legendonly'
            ))
        
        # Layout
        ratio_labels = [
            (f"{float(r):.2f}" if isinstance(r, (int, float, np.number)) else str(r))
            for r in selected_ratios
        ]
        fig.update_layout(
            title=f"{xaxis_metric} vs Fraction Not Converged (Pareto Front)<br>morphology_sizes={sorted([int(m) for m in selected_morphs])}, ratios={ratio_labels}",
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





def plot_scatter_log_spectral_gap_vs_log_induced_norm(
    df: pd.DataFrame,
    atol_zero: float = 1e-6,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    save_path: Optional[str] = None,
    only_converged: bool = False,
    ratio_selection: Optional[List[Any]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of log(spectral_gap) vs log(induced_norm),
    colored by convergence status.
    
    Args:
        df: DataFrame with aggregated convergence results
        atol_zero: Tolerance for considering mean_fraction_not_converged as zero
        figsize: Figure size tuple
        show: If True (default), display the plot
        save_path: If provided, save the figure to this path
        only_converged: If True, display only fully converged (green) points
        ratio_selection: Optional list of ratio_couplings_oscillators values to include
    
    Returns:
        fig, ax: matplotlib Figure and Axes objects
    """
    # Validate required columns
    required = [
        "spectral_gap",
        "induced_norm",
        "mean_fraction_not_converged",
        "ratio_couplings_oscillators",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Clean data
    d = df.copy()

    # Optional ratio filter with tolerant numeric matching and string support
    if ratio_selection is not None:
        ratios_series = d["ratio_couplings_oscillators"]

        effective_series = ratios_series.copy()
        if "actual_ratio_couplings_oscillators" in d.columns:
            mask_max = ratios_series.astype(str) == "max"
            effective_series.loc[mask_max] = d.loc[mask_max, "actual_ratio_couplings_oscillators"]

        mask = pd.Series(False, index=d.index)

        numeric_targets: List[float] = []
        string_targets = set()
        for r in ratio_selection:
            try:
                numeric_targets.append(float(r))
            except (ValueError, TypeError):
                string_targets.add(str(r))

        if numeric_targets:
            ratio_numeric = pd.to_numeric(effective_series, errors="coerce")
            num_mask = np.zeros(len(d), dtype=bool)
            for t in numeric_targets:
                num_mask |= np.isclose(ratio_numeric, t, rtol=1e-8, atol=1e-10)
            mask |= num_mask

        if string_targets:
            mask |= ratios_series.astype(str).isin(string_targets)

        d = d[mask]
        if len(d) == 0:
            available = sorted(ratios_series.dropna().unique())
            raise ValueError(
                f"No data after applying ratio_selection filter. Available ratios: {available}"
            )

    d = d.dropna(subset=required)
    
    # Keep only positive spectral gap and induced norm for log scale
    d = d[(d["spectral_gap"] > 0) & (d["induced_norm"] > 0)]
    
    if len(d) == 0:
        raise ValueError("No valid data with positive spectral_gap and induced_norm")

    # Optionally keep only fully converged points
    is_converged_mask = np.isclose(d["mean_fraction_not_converged"].to_numpy(), 0.0, atol=atol_zero)
    if only_converged:
        d = d[is_converged_mask]
        if len(d) == 0:
            raise ValueError("No fully converged points to plot after filtering")
        # After filtering, everything is converged
        is_converged_mask = np.ones(len(d), dtype=bool)

    # Prepare x and y values (log scale)
    x_vals = np.log10(d["spectral_gap"].to_numpy())
    y_vals = np.log10(d["induced_norm"].to_numpy())
    
    # Color by convergence: green if mean_fraction_not_converged == 0, red otherwise
    colors = np.where(is_converged_mask, "tab:green", "tab:red")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(
        x_vals,
        y_vals,
        c=colors,
        s=50,
        edgecolors="black",
        linewidths=0.5,
    )
    
    ax.set_xlabel("log₁₀(spectral_gap)", fontsize=12)
    ax.set_ylabel("log₁₀(induced_norm)", fontsize=12)
    ax.set_title("Spectral Gap vs Induced Norm (log scales)", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    
    # Legend for colors
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Fully converged",
               markerfacecolor="tab:green", markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", label="Not fully converged",
               markerfacecolor="tab:red", markersize=10, markeredgecolor="black"),
    ]
    ax.legend(handles=handles, loc="best", title="Convergence Status")
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax