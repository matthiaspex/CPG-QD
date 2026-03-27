from typing import Optional
from matplotlib import pyplot as plt
import wandb
import sys
import numpy as np

from cpg_convergence.utils import get_run_name_from_config
from cpg_convergence.defaults import FPS

class WandbLogger:
    def __init__(
            self,
            project: str,
            group: str = None,
            run_name: str = None,
            tags: list = None,
            notes: str = None,
            config: dict = None,
            enable: bool = True
        ):
        """
        Initialize the WandbLogger.
        Args:
            project (str): Name of the WandB project.
            group (str, optional): Group name for the run.
            run_name (str, optional): Name of the run.
            tags (list, optional): List of tags for the run.
            notes (str, optional): Notes for the run.
            config (dict, optional): Configuration dictionary for the, logged for future reference.
            enable (bool): Whether to enable WandB logging. Defaults to True.
        """
        if run_name is None:
            run_name = get_run_name_from_config(config) if (config is not None) else None
        self.enable = enable
        self.step = 0

        if self.enable:
            self.run = wandb.init(project=project, group=group, name=run_name, tags=tags, notes=notes, config=config)

    def reset_step(self):
        """Reset the step counter to 0."""
        self.step = 0

    def advance_step(self):
        """Advance the step counter by 1."""
        self.step += 1

    def log(self, metrics: dict, step: int = None):
        """By specifying the step, wandb does not implicitly advance to next step with every log call."""
        if step is None:
            step = self.step
        if self.enable:
            wandb.log(metrics, step=step)

    def log_video(
            self,
            video_name: str,
            video_path: str,
            caption: Optional[str] = None,
            fps: int = FPS,
            format: str = "mp4"
        ):
        """
        Log a video to WandB.
        Args:
            video_name (str): Name of the video.
            video_path (str): Path to the video file.
            caption (str): Caption for the video.
            fps (int): Frames per second for the video.
            format (str): Format of the video file. Defaults to "mp4".
        """
        if self.enable:
            wandb.log({video_name: wandb.Video(video_path, caption=caption, fps=fps, format=format)})


    def log_image(self, image_name: str, image_path: str):
        """
        Log an image to WandB.
        Args:
            image_name (str): Name of the image.
            image_path (str): Path to the image file.
        """
        if self.enable:
            wandb.log({image_name: wandb.Image(image_path)})

    def log_image_per_step(self, image_name: str, image_path: str, step: int = None):
        if step is None:
            step = self.step
        if self.enable:
            wandb.log({image_name: wandb.Image(image_path)}, step=step)

    def log_histogram(self, histogram_name: str, data: np.ndarray, step: int = None):
        if step is None:
            step = self.step

        if self.enable:
            wandb.log({histogram_name: wandb.Histogram(data)}, step=step)

    def log_heatmap(self, heatmap_name: str, data: np.ndarray, step: int = None):
        """
        Log 2D data as a heatmap to WandB.
        """
        if step is None:
            step = self.step
        
        if self.enable:
            plt.figure(figsize=(4, 4))
            plt.imshow(data, cmap="seismic", aspect="auto")
            plt.colorbar()
            plt.title(heatmap_name)

            wandb.log({heatmap_name: wandb.Image(plt)}, step=step)

            plt.close()  # Close the plot to avoid displaying it in Jupyter notebooks


    def finish(self):
        if self.enable:
            wandb.finish()
