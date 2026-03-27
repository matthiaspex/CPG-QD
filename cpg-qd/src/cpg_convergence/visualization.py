# Visualisation module
import os
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy

import numpy as np
from jax import numpy as jnp

import mediapy as media
import imageio

import matplotlib
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt, cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi


def plot_time_array():
    raise NotImplementedError("This function is not yet implemented.")

def plot_grid_time_arrays(data, ring_setup, title: str, ylabel, show=True, path_save=None):
    """
    Plot grid with number of ring nodes in columns and number of arm nodes in rows.
    Each subplot shows the time array for the corresponding configuration."""
    raise NotImplementedError("This function is not yet implemented.")


def create_histogram(data, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency', color='blue', edgecolor='black', alpha=0.7, subtitles=None):
    """
    Creates a histogram plot and returns the figure object.
    Can also take list of data and list of subtitles

    Parameters:
    - data: array-like or list, the data to plot
    - bins: int, number of bins in the histogram
    - color: str, color of the bars
    - edgecolor: str, color of the bar edges
    - alpha: float, transparency of the bars
    - title: str, title of the plot
    - xlabel: str, label for the x-axis
    - ylabel: str, label for the y-axis

    Returns:
    - fig: matplotlib.figure.Figure, the figure object of the histogram
    """
    if isinstance(data, list):
        num_histograms = len(data)
        fig, axes = plt.subplots(num_histograms, 1, figsize=(8, 6 * num_histograms))

        # Ensure axes is iterable
        if num_histograms == 1:
            axes = [axes]

        for i, data in enumerate(data):
            ax = axes[i]
            ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
            ax.set_xlabel(xlabel, fontsize = 16)
            ax.set_ylabel(ylabel, fontsize = 16)
            if subtitles and (len(subtitles)==num_histograms):
                ax.set_title(subtitles[i], fontsize=20)
                ax.grid(True, linestyle='--', alpha=0.5)

        # Set the global title
        fig.suptitle(title, fontsize=24)  # Adjust y for spacing
        plt.tight_layout()
        plt.subplots_adjust(top=1-(0.25/num_histograms))  # Add space for the global title


    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


def show_video(
        images: List[np.ndarray | None],
        fps: int = 60
        ) -> str | None:
    # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
    filtered_images = [image for image in images if image is not None]
    num_nones = len(images) - len(filtered_images)
    if num_nones > 0:
        logging.warning(f"env.render produced {num_nones} None's. Resulting video might be a bit choppy (consquence of https://github.com/google-deepmind/mujoco/issues/1379).")
    return media.show_video(images=filtered_images, fps=fps)


def save_video_from_raw_frames(
        frames,
        fps: int,
        file_path: str
):
    imgio_kargs = {
        'fps': fps, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
        'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
        }
    writer = imageio.get_writer(file_path, **imgio_kargs)
    for frame in frames:
        writer.append_data(frame)
    writer.close()



# Visualisations for MAPElites
def get_voronoi_finite_polygons_2d(
    centroids: np.ndarray, radius: Optional[float] = None
) -> Tuple[List, np.ndarray]:
    """Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions."""
    voronoi_diagram = Voronoi(centroids)
    if voronoi_diagram.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = voronoi_diagram.vertices.tolist()

    center = voronoi_diagram.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(voronoi_diagram.points, axis=0).max()

    # Construct a map containing all ridges for a given point
    all_ridges: Dict[jnp.ndarray, jnp.ndarray] = {}
    for (p1, p2), (v1, v2) in zip(
        voronoi_diagram.ridge_points, voronoi_diagram.ridge_vertices
    ):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(voronoi_diagram.point_region):
        vertices = voronoi_diagram.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = voronoi_diagram.points[p2] - voronoi_diagram.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = voronoi_diagram.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = voronoi_diagram.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_2d_map_elites_repertoire(
    repertoire: MapElitesRepertoire,
    descriptor_types: List[str],
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Axes]:
    """Plot a visual representation of a 2d map elites repertoire.

    Args:
        repertoire: the MapElitesRepertoire containing fitnesses, centroids, descriptors, etc.
        minval: minimum values for the descriptors
        maxval: maximum values for the descriptors
        ax: a matplotlib axe for the figure to plot. Defaults to None.
        vmin: minimum value for the fitness. Defaults to None. If not given,
            the value will be set to the minimum fitness in the repertoire.
        vmax: maximum value for the fitness. Defaults to None. If not given,
            the value will be set to the maximum fitness in the repertoire.

    Raises:
        NotImplementedError: does not work for descriptors dimension different
        from 2.

    Returns:
        A figure and axes object, corresponding to the visualisation of the
        repertoire.
    """
    centroids = repertoire.centroids
    descriptors = repertoire.descriptors
    if repertoire.fitnesses.ndim == 2:
        repertoire_fitnesses = repertoire.fitnesses.squeeze(axis=1)
    elif repertoire.fitnesses.ndim >= 3:
        raise ValueError("fitnesses must be 1D or 2D")

    grid_empty = repertoire_fitnesses.ravel() == -jnp.inf
    num_descriptors = centroids.shape[1]
    if num_descriptors != 2:
        raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

    my_cmap = cm.viridis

    fitnesses = repertoire_fitnesses
    if vmin is None:
        vmin = float(jnp.min(fitnesses[~grid_empty]))
    if vmax is None:
        vmax = float(jnp.max(fitnesses[~grid_empty]))

    # set the parameters
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "figure.figsize": [10, 10],
    }

    matplotlib.rcParams.update(params)

    # create the plot object
    fig = None
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # fill the plot with the colors
    for idx, fitness in enumerate(fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]

            ax.fill(*zip(*polygon), alpha=0.8, color=my_cmap(norm(fitness)))

    # if descriptors are specified, add points location
    if descriptors is not None:
        descriptors = descriptors[~grid_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c=fitnesses[~grid_empty],
            cmap=my_cmap,
            s=10,
            zorder=0,
        )

    # aesthetic
    ax.set_xlabel(descriptor_types[0])
    ax.set_ylabel(descriptor_types[1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    ax.set_title("Archive of Elites", fontsize= int(1.5*font_size))
    ax.set_aspect("equal")

    return fig, ax




def set_figure_fontsizes(
    fig: Figure,
    ax: Axes,
    title_fontsize: int = 20,
    label_fontsize: int = 16,
    tick_fontsize: int = 12,
    colorbar_fontsize: int = 12,
) -> Tuple[Figure, Axes]:
    """
    Modify font sizes in a matplotlib figure and axes object.
    
    Args:
        fig: matplotlib Figure object
        ax: matplotlib Axes object
        title_fontsize: font size for the title
        label_fontsize: font size for axis labels
        tick_fontsize: font size for tick labels
        colorbar_fontsize: font size for colorbar tick labels
    
    Returns:
        Tuple of modified Figure and Axes objects
    """
    # Set title fontsize
    ax.set_title(ax.get_title(), fontsize=title_fontsize)
    
    # Set axis label fontsizes
    ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)
    
    # Set tick label fontsizes
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Find and update colorbar fontsize
    # The colorbar is typically in a separate axes created by make_axes_locatable
    for axes in fig.get_axes():
        if axes != ax:
            # This is likely the colorbar axes
            axes.tick_params(labelsize=colorbar_fontsize)
    
    return fig, ax