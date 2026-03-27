import yaml
import numpy as np
import pickle
import os
from scipy.spatial.transform import Rotation as R


def load_config_from_yaml(
        yaml_path: str
        ):
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_config_to_yaml(
        cfg: dict,
        file_path: str,
        ):
    assert isinstance(cfg, dict), "Config must be a dictionary."
    assert file_path.endswith('.yaml'), "File path must end with .yaml"

    with open(file_path, "w") as f:
        dump = yaml.dump(cfg)
        f.write(dump)
        

def get_run_name_from_config(
        cfg: dict
):
    """
    If an f-string like format is provided in the config file, which takes other config information:
    e.g.    run_name_format: "{reward_type} arms {arm_setup} popsize {popsize} {notes}" 
    This string can be read out using this function.
    Only information up untill 3 indents in the config file can be read out this way.
    """
    config_flat = {}
    for k, v in cfg.items():
        if not isinstance(v, dict):
            config_flat[k] = v
        else:
            for l, w in v.items():
                if not isinstance(w, dict):
                    config_flat[l] = w
                else:
                    for m, u in w.items():
                        if not isinstance(u, dict):
                            config_flat[m] = u

    run_name = cfg["wandb"]["run_name"].format(**config_flat)
    return run_name


def save_to_pickle(
    data: dict,
    file_path: str,
): 
    assert file_path.endswith('.pkl'), "File path must end with .pkl"

    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_dict_from_pickle(
    file_path: str
) -> dict:
    assert file_path.endswith('.pkl'), "File path must end with .pkl"

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data



def clip_and_rescale(
    values: np.ndarray,
    clip_min: float,
    clip_max: float,
    rescale_min: float,
    rescale_max: float,
) -> np.ndarray:
    """
    1. Clip the input values to the range [clip_min, clip_max]
    2. Rescale the clipped values to the range [rescale_min, rescale_max]
    Usage: In evolutionary optimization, the sampling in the hyperspace should be isotropic.
           Rescaling to the actual parameter ranges is done after sampling.
    """
    clipped = np.clip(values, clip_min, clip_max)
    # Normalizing to 0-1
    normalized = (clipped - clip_min) / (clip_max - clip_min)

    # Scaling to new range
    scaled = normalized * (rescale_max - rescale_min) + rescale_min

    return scaled



def quaternion_to_axis_angle(q):
    """
    convert quaternion [w, x, y, z] to axis-angle representation [x, y, z]
    """
    # SciPy expects quaternions in [x, y, z, w] order
    r = R.from_quat([q[1], q[2], q[3], q[0]])

    # Convert to Euler angles (in radians, default 'xyz' order)
    euler_angles = r.as_euler('xyz', degrees=False)

    return euler_angles








    