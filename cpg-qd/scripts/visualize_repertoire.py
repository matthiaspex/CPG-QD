"""Visualize a repertoire from a given archive"""
import os
import sys
import pickle
import jax.numpy as jnp

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from cpg_convergence.visualization import plot_2d_map_elites_repertoire, set_figure_fontsizes

resutls_dir = os.path.join(os.path.dirname(__file__), "..", "results") 
exp_path = "b02_r33_35"
exp_dir = os.path.join(resutls_dir, exp_path)

with open(os.path.join(exp_dir, "slow_repertoire.pkl"), "rb") as f:
    repertoire_slow = pickle.load(f)

with open(os.path.join(exp_dir, "fast_stable_repertoire.pkl"), "rb") as f:
    repertoire_fast_stable = pickle.load(f)

print(dir(repertoire_slow))
print(f"coverage slow: {jnp.sum(repertoire_slow.fitnesses > 0)/repertoire_slow.fitnesses.shape[0]}")
print(f"coverage fast/stable: {jnp.sum(repertoire_fast_stable.fitnesses > 0)/repertoire_fast_stable.fitnesses.shape[0]}")
print(f"qd_score slow: {jnp.sum(jnp.where(repertoire_slow.fitnesses > 0, repertoire_slow.fitnesses, 0))}")
print(f"qd_score fast/stable: {jnp.sum(jnp.where(repertoire_fast_stable.fitnesses > 0, repertoire_fast_stable.fitnesses, 0))}")
# sys.exit()

print(type(repertoire_slow.fitnesses))
shared_vmax = max(jnp.max(repertoire_slow.fitnesses), jnp.max(repertoire_fast_stable.fitnesses))

fig_slow, ax_slow = plot_2d_map_elites_repertoire(
repertoire=repertoire_slow,
descriptor_types=["bilateral_contralateral_score", "assistive_score"],
minval=[-1, 1],
maxval=[1, 5],
vmin=0, # set colorscale minimum value at 0.
vmax=shared_vmax, # set colorscale maximum value the same for both figures.
)

fig_med, ax_med = plot_2d_map_elites_repertoire(
repertoire=repertoire_fast_stable,
descriptor_types=["bilateral_contralateral_score", "assistive_score"],
minval=[-1, 1],
maxval=[1, 5],
vmin=0, # set colorscale minimum value at 0.
vmax=shared_vmax, # set colorscale maximum value the same for both figures.
)


def post_process_fig(fig,
                     ax,
                     title_fontsize=24,
                     label_fontsize=20,
                     tick_fontsize=16,
                     colorbar_fontsize=16,
                     remove_text=False,
                     new_title=None,
                     new_xlabel=None,
                     new_ylabel=None,
                     reduce_whitespace=True,
                     remove_axis_ticks=False
                     ):
    """Set font sizes for the figure."""
    if remove_text:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        cbar = fig.axes[1]  # Assuming the colorbar is the second axis
        cbar.set_ylabel("")
        cbar.tick_params(labelsize=0)  # Remove colorbar tick labels
    if remove_axis_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if new_title is not None:
        ax.set_title(new_title, fontsize=title_fontsize)
    if new_xlabel is not None:
        ax.set_xlabel(new_xlabel, fontsize=label_fontsize)
    if new_ylabel is not None:
        ax.set_ylabel(new_ylabel, fontsize=label_fontsize)
    if reduce_whitespace:
        fig.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)
        
    set_figure_fontsizes(fig, ax, title_fontsize, label_fontsize, tick_fontsize, colorbar_fontsize)
    

post_process_fig(fig_slow, ax_slow,
                 title_fontsize=34,
                 label_fontsize=30,
                 tick_fontsize=24,
                 colorbar_fontsize=24,
                 remove_text=False,
                 new_title="Slow CPG",
                 reduce_whitespace=True
                 )

fig_slow.savefig(os.path.join(exp_dir, "slow_repertoire_visualization.png"), dpi=300,
                 bbox_inches='tight', pad_inches=0.1)


post_process_fig(fig_med, ax_med,
                 title_fontsize=34,
                 label_fontsize=30,
                 tick_fontsize=24,
                 colorbar_fontsize=24,
                 remove_text=False,
                 new_title="Fast/stable CPG",
                 reduce_whitespace=True
                 )

fig_med.savefig(os.path.join(exp_dir, "fast_stable_repertoire_visualization.png"), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)


def put_figures_together(figures, save_path, dpi=300, direction="horizontal"):
    """Combine multiple figures and save as a single image."""
    from PIL import Image
    import io

    # Convert each figure to a PIL Image
    images = []
    for fig in figures:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.3)
        buf.seek(0)
        img = Image.open(buf)
        images.append(img)

    direction = direction.lower()
    if direction not in {"horizontal", "vertical"}:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    if direction == "horizontal":
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        combined_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        current_x = 0
        for img in images:
            combined_img.paste(img, (current_x, 0))
            current_x += img.width
    else:  # vertical
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        combined_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))

        current_y = 0
        for img in images:
            combined_img.paste(img, (0, current_y))
            current_y += img.height

    # Save the combined image
    combined_img.save(save_path)

put_figures_together(
    figures=[fig_slow, fig_med],
    save_path=os.path.join(exp_dir, "combined_repertoire_visualization_horizontal.png"),
    dpi=300,
    direction="horizontal"
)

put_figures_together(
    figures=[fig_slow, fig_med],
    save_path=os.path.join(exp_dir, "combined_repertoire_visualization_vertical.png"),
    dpi=300,
    direction="vertical"
)




# Bigger fonts
post_process_fig(fig_slow, ax_slow,
                 title_fontsize=45,
                 label_fontsize=40,
                 tick_fontsize=35,
                 colorbar_fontsize=35,
                 remove_text=False,
                 new_title="Slow CPG",
                 reduce_whitespace=True,
                 new_xlabel="Descr 1",
                 new_ylabel="Descr 2",
                 remove_axis_ticks=True
                 )



post_process_fig(fig_med, ax_med,
                 title_fontsize=45,
                 label_fontsize=40,
                 tick_fontsize=35,
                 colorbar_fontsize=35,
                 remove_text=False,
                 new_title="Fast/stable CPG",
                 reduce_whitespace=True,
                 new_xlabel="Descr 1",
                 new_ylabel="Descr 2",
                 remove_axis_ticks=True
                 )

put_figures_together(
    figures=[fig_slow, fig_med],
    save_path=os.path.join(exp_dir, "combined_repertoire_visualization_vertical_big_fonts.png"),
    dpi=300,
    direction="vertical"
)