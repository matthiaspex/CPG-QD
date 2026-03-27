
import logging
import os
import sys
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
from typing import List
import mediapy as media

import mujoco
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.mjcf.component import MJCFRootComponent

from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.specification import BrittleStarMorphologySpecification

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
jnp.set_printoptions(precision=3, suppress=True, linewidth=100)


tmp_dir=os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp"))


def visualize_mjcf(
        mjcf: MJCFRootComponent,
        show: bool = False,
        path: str | None = None,
        height: int = 480,
        width: int = 640,
        ) -> str | None:
    """Render MJCF; optionally display and/or save to disk."""
    model = mujoco.MjModel.from_xml_string(mjcf.get_mjcf_str())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    image = renderer.render()

    if show:
        try:
            media.show_image(image)
        except Exception as exc:  # non-interactive/GUI-less fallback
            logging.warning("Could not display image: %s", exc)

    if path is not None:
        media.write_image(path, image)
        logging.info("MJCF render saved to %s", path)



def post_render(
        render_output: List[np.ndarray],
        environment_configuration: MuJoCoEnvironmentConfiguration
        ) -> np.ndarray:
    if render_output is None:
        return

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [np.concatenate(env_frames, axis=1) for env_frames in frames_per_env]

    # Vertically stack frames of different environments
    render_output = np.concatenate(render_output, axis=0)

    return render_output[:, :, ::-1]  # RGB to BGR


def show_video(
        images: List[np.ndarray | None]
        ) -> str | None:
    # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
    filtered_images = [image for image in images if image is not None]
    num_nones = len(images) - len(filtered_images)
    if num_nones > 0:
        logging.warning(
                f"env.render produced {num_nones} None's. Resulting video might be a bit choppy (consquence of https://github.com/google-deepmind/mujoco/issues/1379)."
                )
    return media.show_video(images=filtered_images)


def create_morphology(
        morphology_specification: BrittleStarMorphologySpecification
        ) -> MJCFBrittleStarMorphology:
    morphology = MJCFBrittleStarMorphology(
            specification=morphology_specification
            )
    return morphology

arm_setup = [2] * 7

morphology_specification = default_brittle_star_morphology_specification(
        num_arms=len(arm_setup), 
        num_segments_per_arm=arm_setup, 
        # Whether or not to use position-based control (i.e. the actuation or control inputs are target joint positions).
        use_p_control=True,
        )
morphology = create_morphology(morphology_specification=morphology_specification)
visualize_mjcf(mjcf=morphology, path=os.path.join(tmp_dir, "bs_morph.png"))




from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena


def create_arena(
        arena_configuration: AquariumArenaConfiguration
        ) -> MJCFAquariumArena:
    arena = MJCFAquariumArena(
            configuration=arena_configuration
            )
    return arena



arena_configuration = AquariumArenaConfiguration(
        size=(10, 5), sand_ground_color=True, attach_target=False, wall_height=1.5, wall_thickness=0.1
        )
arena = create_arena(arena_configuration=arena_configuration)
visualize_mjcf(mjcf=arena)


morphology = create_morphology(morphology_specification=morphology_specification)
print(type(morphology))
arena = create_arena(arena_configuration=arena_configuration)
print(type(arena))
arena.attach(morphology, free_joint=True)
print(type(arena))
visualize_mjcf(mjcf=arena)

print(arena.mjcf_model.find_all("actuator"))

model = mujoco.MjModel.from_xml_string(arena.get_mjcf_str())
data = mujoco.MjData(model)

print(type(model))
print(type(data))



print(model.nsensor)
print(model.sensor("BrittleStarMorphology/central_disk_framepos_sensor"))

for i in range(model.nsensor):
    print(model.sensor(i).name)


print(model.ngeom)
print(model.nu)