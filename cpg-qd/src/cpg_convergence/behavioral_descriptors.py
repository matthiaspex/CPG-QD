"""
Class that can extract and process metrics from the simulation data.
"""

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import scipy.signal as signal
import jax
from jax import numpy as jnp
import jax.tree_util as tree
import sys
import time

from cpg_convergence.utils import quaternion_to_axis_angle
from cpg_convergence.defaults import CONTACT_THRESHOLD
        


class BehavioralDescriptorsExtractor:
    """
    Note: everything should be compatible with an nbatch dimension at axis 0.

    Class allowing to extract and analyze brittle star behaviour from state, sensordata.

    Available descriptors (properties):
    General:
    - ground_contact_fraction
    - disk_elevation
    - sine_total_displacement
    - cosine_total_displacement
    Specifically for 5-armed brittle stars:
    - bilateral_contralateral_score
    - bilateral_score
    - contralateral_score
    - bilateral_score_grf
    - contralateral_score_grf
    - assistive_score


    Args:
        state array [nbatch, nsteps, nstate]: The state object containing simulation data.
        sensordata (dict[arrays]) [nbatch, nsteps, nsensordata]: The sensor data dictionary containing various sensor readings.

    Side effects at initialization:
    - Enhances sensordata with 3D disk rotation if not already present. (from 4D quaternion)
    - Get estimated gait frequency via fourier analysis of the XY velocity norm.
                Either the actual gait frequency (1 stroke/cycle)
                or half the gait frequency (2 strokes/cycle).
    - Identifies cycle start indices
    """
    def __init__(
            self,
            state,
            sensordata,
            steps_to_omit_transient: int = 0,
            arm_setup: Optional[List[int]] = None,
            ):
        if steps_to_omit_transient > 0:
            state = state[:, steps_to_omit_transient:, :]
            sensordata = jax.tree_util.tree_map(lambda x: x[:, steps_to_omit_transient:, ...], sensordata)

        sensordata = enhance_sensordata_with_3D_disk_rotation(sensordata)

        self.state = state
        self.sensordata = sensordata

        self.arm_setup = arm_setup



    @property
    def XY_velocity_norm(self):
        """dimensions: [nbatch, nsteps]"""
        return np.linalg.norm(self.sensordata['central_disk']['linvel'][:, :, :2], axis=-1)
    
    @property
    def arm_roles(self) -> dict:
        """
        Only for 5-armed brittle stars.
        A dict with structure:
        {'left': {'front': np.ndarray(nbatch),
                   'hind': np.ndarray(nbatch)},
         'right': {'front': np.ndarray(nbatch),
                   'hind': np.ndarray(nbatch)},
        }
        """
        if not hasattr(self, '_arm_roles'):
            assert len(self.arm_setup) == 5, "Arm roles are only defined for 5-armed brittle stars."
            self._arm_roles = self._get_arm_roles(self.central_limb)
        return self._arm_roles
    
    @property
    def leading_or_trailing(self) -> np.ndarray[str]:
        """Only for 5-armed brittle stars."""
        if not hasattr(self, '_leading_or_trailing'):
            assert len(self.arm_setup) == 5, "Leading or trailing limb is only defined for 5-armed brittle stars."
            self._leading_or_trailing = np.where(self.central_limb < 5, "leading", "trailing")
        return self._leading_or_trailing
    
    @property
    def central_limb(self):
        """
        Only for 5-armed brittle stars.
        Array with nbatch central limbs
        If leading limb: index 0 -> 4, trailing limb: index 5 -> 9
        """
        if not hasattr(self, '_central_limb'):
            assert len(self.arm_setup) == 5, "Central limb is only defined for 5-armed brittle stars."
            self._central_limb = self._get_central_limb(self.sensordata)
        return self._central_limb
    
    @property
    def bilateral_contralateral_score(self) -> np.ndarray:
        """
        Only for 5-armed brittle stars.
        Returns descriptor with dimension [nbatch,]
        """
        if not hasattr(self, '_bilateral_contralateral_score'):
            assert len(self.arm_setup) == 5, "Bilateral contralateral score is only defined for 5-armed brittle stars."
            _compute_bilateral_contralateral_score_batched = jax.vmap(compute_bilateral_contralateral_score)
            self._bilateral_contralateral_score = _compute_bilateral_contralateral_score_batched(self.state, self.sensordata, self.arm_roles)

        return self._bilateral_contralateral_score
    
    @property
    def bilateral_score(self) -> np.ndarray:
        """
        Only for 5-armed brittle stars.
        Returns descriptor with dimension [nbatch,]
        """
        if not hasattr(self, '_bilateral_score'):
            assert len(self.arm_setup) == 5, "Bilateral score is only defined for 5-armed brittle stars."
            _compute_bilateral_score_contralateral_score_separately_batched = jax.vmap(compute_bilateral_score_contralateral_score_separately)
            self._bilateral_score, self._contralateral_score = _compute_bilateral_score_contralateral_score_separately_batched(self.state, self.sensordata, self.arm_roles)

        return self._bilateral_score

    @property
    def bilateral_score_grf(self) -> np.ndarray:
        """
        Only for 5-armed brittle stars.
        Returns descriptor with dimension [nbatch,]
        """
        if not hasattr(self, '_bilateral_score_grf'):
            assert len(self.arm_setup) == 5, "Bilateral score grf is only defined for 5-armed brittle stars."
            _vectorized_compute_bilateral_score_contralateral_score_from_grf = jax.vmap(compute_bilateral_score_contralateral_score_from_grf)
            self._bilateral_score_grf, _ = _vectorized_compute_bilateral_score_contralateral_score_from_grf(self.state, self.sensordata, self.arm_roles)

        return self._bilateral_score_grf    

    @property
    def contralateral_score(self) -> np.ndarray:
        """
        Only for 5-armed brittle stars.
        Returns descriptor with dimension [nbatch,]
        """
        if not hasattr(self, '_contralateral_score'):
            assert len(self.arm_setup) == 5, "Contralateral score is only defined for 5-armed brittle stars."
            _compute_bilateral_score_contralateral_score_separately_batched = jax.vmap(compute_bilateral_score_contralateral_score_separately)
            self._bilateral_score, self._contralateral_score = _compute_bilateral_score_contralateral_score_separately_batched(self.state, self.sensordata, self.arm_roles)

        return self._contralateral_score
    
    @property
    def contralateral_score_grf(self) -> np.ndarray:
        """
        Only for 5-armed brittle stars.
        Returns descriptor with dimension [nbatch,]
        """
        if not hasattr(self, '_contralateral_score_grf'):
            assert len(self.arm_setup) == 5, "Contralateral score grf is only defined for 5-armed brittle stars."
            _vectorized_compute_bilateral_score_contralateral_score_from_grf = jax.vmap(compute_bilateral_score_contralateral_score_from_grf)
            _, self._contralateral_score_grf = _vectorized_compute_bilateral_score_contralateral_score_from_grf(self.state, self.sensordata, self.arm_roles)

        return self._contralateral_score_grf
    

    @property
    def assistive_score(self):
        """
        Only for 5-armed brittle stars.
        Returns descriptor with dimension [nbatch,], with a number of the number of assisitve limbs
        """
        if not hasattr(self, '_assistive_score'):
            assert len(self.arm_setup) == 5, "Assistive score is only defined for 5-armed brittle stars."
            vectorized_compute_assistive_score = jax.vmap(compute_assistive_score)
            self._assistive_score, self._normalized_propulsive_force_per_arm = vectorized_compute_assistive_score(self.state, self.sensordata)
        return self._assistive_score
    
    

    @property
    def ground_contact_fraction(self):
        """
        Returns descriptor with dimension [nbatch,]
        """
        if not hasattr(self, '_ground_contact_fraction'):
            contact_bools = (self.sensordata["segments"]["contact"] > CONTACT_THRESHOLD).astype(int)
            self._ground_contact_fraction = np.mean(np.mean(contact_bools, axis=-1), axis=-1)  # mean across segments and timesteps

        return self._ground_contact_fraction
    
    @property
    def disk_elevation(self):
        """
        Returns 95%-percentile across the episode of disk_elevation over time.
        Returns descriptor with dimension [nbatch,]"""
        if not hasattr(self, '_disk_elevation'):
            self._disk_elevation = np.percentile(self.sensordata["central_disk"]["position"][:, :, 2], 95, axis=-1)

        return self._disk_elevation
    
    @property
    def sine_total_displacement(self):
        """Returns descriptor with dimension [nbatch,]"""
        if not hasattr(self, '_sine_total_displacement'):
            positions = self.sensordata["central_disk"]["position"]  # shape: [nbatch, nsteps, 3]
            delta_x = positions[:, -1, 0] - positions[:, 0, 0]
            delta_y = positions[:, -1, 1] - positions[:, 0, 1]
            total_euclidian_displacement = np.sqrt(delta_x ** 2 + delta_y ** 2)
            self._sine_total_displacement = delta_y / total_euclidian_displacement
        return self._sine_total_displacement

    @property
    def cosine_total_displacement(self):
        """Returns descriptor with dimension [nbatch,]"""
        if not hasattr(self, '_cosine_total_displacement'):
            positions = self.sensordata["central_disk"]["position"]  # shape: [nbatch, nsteps, 3]
            delta_x = positions[:, -1, 0] - positions[:, 0, 0]
            delta_y = positions[:, -1, 1] - positions[:, 0, 1]
            total_euclidian_displacement = np.sqrt(delta_x ** 2 + delta_y ** 2)
            self._cosine_total_displacement = delta_x / total_euclidian_displacement
        return self._cosine_total_displacement
    
    @staticmethod # easier for mapping in the future
    def _get_central_limb(sensordata):
        """
        Determine the central limb (leading or trailing) based on XY velocity and disk z-rotation.

        Args:
            sensordata [nbatch, nsteps, nsensordata]: The sensor data dictionary
        Returns:
            central_limb (np.ndarray): Array of shape [nbatch] with the index of the central limb.
                - index 0 -> 4 for leading limb
                - index 5 -> 9 for trailing limb
        """
        xy_dir = np.array([sensordata["central_disk"]["position"][:, -1, 0] - sensordata["central_disk"]["position"][:, 0, 0],
                           sensordata["central_disk"]["position"][:, -1, 1] - sensordata["central_disk"]["position"][:, 0, 1]])
        unit_xy_dir = (xy_dir / np.linalg.norm(xy_dir, axis=0)).T

        z_rotation = np.mean(sensordata["central_disk"]["rotation"][:, :, 2], axis=1)  # Assuming z-rotation is the third component

        get_arm_allignment_with_vector_vectorized = jax.vmap(get_arm_allignment_with_vector, in_axes=(0, 0, None))
        alignment_factors = get_arm_allignment_with_vector_vectorized(unit_xy_dir, z_rotation, 5)

        central_limb = np.argmax(alignment_factors, axis=1)  # Get the index of the arm with the highest alignment factor
        signs = np.sign(alignment_factors[np.arange(len(central_limb)), central_limb])
        correction = np.where(signs > 0, 0, 5)

        return central_limb + correction
    

    @staticmethod
    def _get_arm_roles(central_limb) -> Dict[str, np.ndarray]:
        """A dict with structure:
        {'left': {'front': np.ndarray(nbatch),
                   'hind': np.ndarray(nbatch)},
         'right': {'front': np.ndarray(nbatch),
                   'hind': np.ndarray(nbatch)},
        }
        """
        get_index_per_arm_role_vectorized = jax.vmap(get_index_per_arm_role, in_axes=(0, None))
        arm_roles = get_index_per_arm_role_vectorized(central_limb, 5)
        return arm_roles
    

def enhance_sensordata_with_3D_disk_rotation(sensordata, verbose=False):
    if not "rotation" in sensordata["central_disk"].keys():
        rotation = np.zeros((sensordata["central_disk"]["quaternion"].shape[0],
                            sensordata["central_disk"]["quaternion"].shape[1],
                            3))  # Initialize rotation array
        for i in range(sensordata["central_disk"]["quaternion"].shape[0]):
            for j in range(sensordata["central_disk"]["quaternion"].shape[1]):
                rotation[i, j] = quaternion_to_axis_angle(sensordata["central_disk"]["quaternion"][i, j])
        sensordata["central_disk"]["rotation"] = rotation
    else:
        if verbose:
            print("Rotation already exists in sensordata['central_disk']. Skipping enhancement.")
    return sensordata


def get_arm_allignment_with_vector(unit_vector, disk_rotation, narms=5):
    """Takes a single vector and computes its alignment with the arm directions.
    can be jax.vmapped for nbatch dimension"""
    arm_dir = jnp.zeros((narms, 2))
    angle = 2*jnp.pi/narms
    disk_rotation

    for i in range(narms):
        arm_dir = arm_dir.at[i,:].set(jnp.array([jnp.cos(i*angle+disk_rotation), jnp.sin(i*angle+disk_rotation)]))

    alignment_factors = jnp.dot(unit_vector, arm_dir.T)
    return jnp.array(alignment_factors)

def get_index_per_arm_role(central_limb, narms=5):
    assert narms == 5, "Only 5 arms are supported."

    def leading_case(central_limb):
        central_limb_ind = central_limb % narms
        return {
            "left": {
                "front": (central_limb_ind + 1) % narms,
                "hind": (central_limb_ind + 2) % narms
            },
            "right": {
                "front": (central_limb_ind - 1) % narms,
                "hind": (central_limb_ind - 2) % narms
            }
        }
    def trailing_case(central_limb):
        central_limb_ind = central_limb % narms
        return {
            "left": {
                "front": (central_limb_ind - 2) % narms,
                "hind": (central_limb_ind - 1) % narms
            },
            "right": {
                "front": (central_limb_ind + 2) % narms,
                "hind": (central_limb_ind + 1) % narms
            }
        }

    return jax.lax.cond(
        central_limb // 5 == 0,
        leading_case,
        trailing_case,
        central_limb
    )


def cosine_similarity(A,B):
    """
    Computes dotproduct of normalized vectors. Measure of how close those vectors are in a hyperplane.
    Returns value between -1 and 1.
    """
    cosim = jnp.dot(A, B) / (jnp.linalg.norm(A) * jnp.linalg.norm(B))
    return cosim


def compute_angvels_per_arm(state, sensordata, arm_roles):
    """
    This is WITHOUT batch dimension.
    Computes angular velocities per arm in the XY plane, with respect to movement direction.

    Args: -> no batch dimension, vmapped afterwards
        state [ndarray] (nsteps, nstate)
        sensordata [ndarray] (nsteps, nsensordata): The sensor data for the current cycle
        arm_roles [dict[strings, ints]]: keys: 'left', 'right'; 'front', 'hind'. Every leave is array with dim 

    Returns:
        dict{str, ndarray}:
            - keys: 'left_front', 'right_front', 'left_hind', 'right_hind'
            - values: [ndarray] angular velocities per arm in the XY plane (shape (nsteps-1,))
    """
    assert len(state.shape) == 2, "nbatch dimension must not be present; if nbatch dimension is required, use vmap."
    xy_dir = jnp.array([sensordata["central_disk"]["position"][-1, 0] - sensordata["central_disk"]["position"][0, 0],
                        sensordata["central_disk"]["position"][-1, 1] - sensordata["central_disk"]["position"][0, 1]])
    unit_xy_dir = (xy_dir / jnp.linalg.norm(xy_dir, axis=0))

    segment_positions = sensordata["segments"]["position"] # shape (nsteps, nsegments, 3) (no nbatch dimension)
    narms = 5
    nsegments = segment_positions.shape[1] // narms
    nsteps = segment_positions.shape[0]

    # calculate vector for every arm from shoulder to halfway point.
    arm_xy_vectors = jnp.zeros((nsteps, narms, 2))
    for arm in range(narms):
        base_segment_index = arm * nsegments
        halfway_arm_segment_index = base_segment_index + nsegments // 2

        dx = segment_positions[:, halfway_arm_segment_index, 0] - segment_positions[:, base_segment_index, 0]
        dy = segment_positions[:, halfway_arm_segment_index, 1] - segment_positions[:, base_segment_index, 1]
        arm_xy_vectors = arm_xy_vectors.at[:, arm, 0].set(dx)
        arm_xy_vectors = arm_xy_vectors.at[:, arm, 1].set(dy)
        unit_arm_xy_vectors = arm_xy_vectors / jnp.linalg.norm(arm_xy_vectors, axis=-1, keepdims=True)

    # compute angle between arm vector and XY_direction vector for every arm and every timestep
    # angle = arccos( dot(a,b) / (||a||*||b||) )
    # since a and b are unit vectors, ||a||*||b|| = 1
    # so angle = arccos( dot(a,b) )
    arm_angles = jnp.arccos(jnp.sum(unit_arm_xy_vectors * unit_xy_dir, axis=-1))

    arm_angular_velocities = jnp.diff(arm_angles, axis=0) # differentiate along time dimension

    angvels_left_front = arm_angular_velocities[jnp.arange(arm_angular_velocities.shape[0]), arm_roles["left"]["front"]]
    angvels_right_front = arm_angular_velocities[jnp.arange(arm_angular_velocities.shape[0]), arm_roles["right"]["front"]]
    angvels_left_hind = arm_angular_velocities[jnp.arange(arm_angular_velocities.shape[0]), arm_roles["left"]["hind"]]
    angvels_right_hind = arm_angular_velocities[jnp.arange(arm_angular_velocities.shape[0]), arm_roles["right"]["hind"]]
    return {
        "left_front": angvels_left_front,
        "right_front": angvels_right_front,
        "left_hind": angvels_left_hind,
        "right_hind": angvels_right_hind
    }


def compute_bilateral_contralateral_score(state, sensordata, arm_roles):
    """
    This is WITHOUT batch dimension.
    Perfect bilateral: score +1: all limbs in sync
    Perfect contralateral: score -1: contralateral limbs in sync, ipsilateral limbs out of sync
    Using Metric inspired by Wakita et al. (2020), but modified to resemble cosine similarity metric
        to compare angular ip velocities of limbs

    Args: -> no batch dimension, vmapped afterwards
        state [ndarray] (nsteps, nstate)
        sensordata [ndarray] (nsteps, nsensordata): The sensor data for the current cycle
        arm_roles [dict[strings, ints]]: keys: 'left', 'right'; 'front', 'hind'. Every leave is array with dim 

    Returns:
        score [ndarray] (nbatch,): The bilateral/contralateral score for each batch
    """
    angvels = compute_angvels_per_arm(state, sensordata, arm_roles)
    angvels_left_front = angvels["left_front"]
    angvels_right_front = angvels["right_front"]
    angvels_left_hind = angvels["left_hind"]
    angvels_right_hind = angvels["right_hind"]

    # Note: angles are relative compared to movement direction
    ipsilateral_front_score = cosine_similarity(angvels_left_front, angvels_right_front)
    ipsilateral_hind_score = cosine_similarity(angvels_left_hind, angvels_right_hind)
    ipsilateral_right_score = cosine_similarity(angvels_right_front, angvels_right_hind)
    ipsilateral_left_score = cosine_similarity(angvels_left_front, angvels_left_hind)
    bilateral_score_contralateral_score = 0.25 * ( ipsilateral_front_score + ipsilateral_hind_score + ipsilateral_right_score + ipsilateral_left_score)

    return bilateral_score_contralateral_score


def compute_bilateral_score_contralateral_score_separately(state, sensordata, arm_roles):
    """
    This is WITHOUT batch dimension.
    Bilateral score:
    - only looks at 2 front limbs
    - max score = +1: both limbs in phase
    - min score = 0: both limbs out in counterphase
    Contralateral score:
    - looks at pairs: left front - right hind, right front - left hind
    - max score = +1: both pairs in phase
    - min score = -1: both pairs in counterphase

    Using Metric inspired by Wakita et al. (2020), but modified to resemble cosine similarity metric
        to compare angular ip velocities of limbs

    Args: -> no batch dimension, vmapped afterwards
        state [ndarray] (nsteps, nstate)
        sensordata [ndarray] (nsteps, nsensordata): The sensor data for the current cycle
        arm_roles [dict[strings, ints]]: keys: 'left', 'right'; 'front', 'hind'. Every leave is array with dim 

    Returns:
        bilateral_score [float]: The bilateral score
        contralateral_score [float]: The contralateral score
    """
    angvels = compute_angvels_per_arm(state, sensordata, arm_roles)
    angvels_left_front = angvels["left_front"]
    angvels_right_front = angvels["right_front"]
    angvels_left_hind = angvels["left_hind"]
    angvels_right_hind = angvels["right_hind"]

    bilateral_score = cosine_similarity(angvels_left_front, angvels_right_front)

    contralateral_pair1_score = cosine_similarity(angvels_left_front, angvels_right_hind)
    contralateral_pair2_score = cosine_similarity(angvels_right_front, angvels_left_hind)
    contralateral_score = 0.5 * (contralateral_pair1_score + contralateral_pair2_score)
    
    return bilateral_score, contralateral_score


def compute_grf_per_arm(state, sensordata) -> jnp.ndarray:
    """No Batch Dimension. shape of sensordata['segments']['ground_reaction_force']: (nsteps, nsegments, 3 or 6)
    Returns: xy_grf_per_arm: shape (nsteps, narms, 2)
    """
    # sum GRFs across segments of an arm
    xy_grf = sensordata["segments"]["ground_reaction_force"][:, :, :2]
    narms = 5
    nsegments = xy_grf.shape[1] // narms
    nsteps = xy_grf.shape[0]
    xy_grf_per_arm = jnp.zeros((nsteps, narms, 2))
    for arm in range(narms):
        arm_segment_indices = jnp.arange(arm * nsegments, (arm + 1) * nsegments)
        xy_grf_per_arm = xy_grf_per_arm.at[:, arm, :].set(jnp.sum(xy_grf[:, arm_segment_indices, :], axis=1))
    return xy_grf_per_arm  # shape (nsteps, narms, 2)


def normalized_dot(v1, v2):
    v1 = jnp.asarray(v1, dtype=jnp.float32)
    v2 = jnp.asarray(v2, dtype=jnp.float32)
    dot = jnp.dot(v1, v2)
    n1 = jnp.linalg.norm(v1)
    n2 = jnp.linalg.norm(v2)
    max_norm = jnp.maximum(n1, n2)
    return jnp.where(max_norm == 0, 1.0, dot / (max_norm**2))
vectorized_normalized_dot = jax.vmap(normalized_dot, in_axes=(0,0))


def compute_bilateral_score_contralateral_score_from_grf(state, sensordata, arm_roles):
    """
    This is WITHOUT batch dimension.
    Computes whether arms exert ground reaction force in the same direction or not:
    - +1: GRF in same direction and of similar magnitude
    - -1: GRF in opposite direction and of similar magnitude -> unlikely, since this would hinder the fitness
    - 0: GRF orthogonal or at least one arm not in contact with the ground. -> orthogonal GRF unlikely due to brittle star morphology

    Bilateral pair: left front - right front
    Contralateral pair: average of left front - right hind, right front - left hind

    Args: -> no batch dimension, vmapped afterwards
        state [ndarray] (nsteps, nstate)
        sensordata [ndarray] (nsteps, nsensordata): The sensor data for the current cycle
        arm_roles [dict[strings, ints]]: keys: 'left', 'right'; 'front', 'hind'. Every leave is array with dim 

    Returns:
        score [ndarray] (nbatch,): The bilateral/contralateral score for each batch
    """
    # compute unit XY direction vector of the central disk
    xy_dir = jnp.array([sensordata["central_disk"]["position"][-1, 0] - sensordata["central_disk"]["position"][0, 0],
                       sensordata["central_disk"]["position"][-1, 1] - sensordata["central_disk"]["position"][0, 1]])
    # use scalar norm (not axis=0) and protect against zero
    norm_xy = jnp.linalg.norm(xy_dir)
    unit_xy_dir = xy_dir / (norm_xy + 1e-12)   # shape (2,)

    grf_per_arm = compute_grf_per_arm(state, sensordata)  # shape (nsteps, narms, 2)

    # projection scalar per timestep: dot(grf(t), unit_xy_dir) -> shape (nsteps,)
    lf_idx = arm_roles["left"]["front"]
    rf_idx = arm_roles["right"]["front"]
    lh_idx = arm_roles["left"]["hind"]
    rh_idx = arm_roles["right"]["hind"]

    proj_lf = jnp.sum(grf_per_arm[:, lf_idx, :] * unit_xy_dir, axis=-1)  # (nsteps,)
    proj_rf = jnp.sum(grf_per_arm[:, rf_idx, :] * unit_xy_dir, axis=-1)
    proj_lh = jnp.sum(grf_per_arm[:, lh_idx, :] * unit_xy_dir, axis=-1)
    proj_rh = jnp.sum(grf_per_arm[:, rh_idx, :] * unit_xy_dir, axis=-1)

    # projected vectors (nsteps, 2) if you need the vector form
    grf_lf_projected_movement_dir = proj_lf[..., None] * unit_xy_dir # shape(nsteps, 2)
    grf_rf_projected_movement_dir = proj_rf[..., None] * unit_xy_dir
    grf_lh_projected_movement_dir = proj_lh[..., None] * unit_xy_dir
    grf_rh_projected_movement_dir = proj_rh[..., None] * unit_xy_dir

    # compute normalized dot products
    norm_dot_front = vectorized_normalized_dot(
        grf_lf_projected_movement_dir,    # shape(nsteps, 2)
        grf_rf_projected_movement_dir     # shape(nsteps, 2)
    )
    bilateral_score_grf = jnp.mean(norm_dot_front) # average value over time

    norm_dot_contralateral_lf_rh = vectorized_normalized_dot( # applay for every timestep
        grf_lf_projected_movement_dir,    # shape(nsteps, 2)
        grf_rh_projected_movement_dir     # shape(nsteps, 2)
    )
    norm_dot_contralateral_rf_lh = vectorized_normalized_dot(
        grf_rf_projected_movement_dir,    # shape(nsteps, 2)
        grf_lh_projected_movement_dir     # shape(nsteps, 2)
    )
    contralateral_score_grf = 0.5 * (jnp.mean(norm_dot_contralateral_lf_rh) + jnp.mean(norm_dot_contralateral_rf_lh))
    
    return bilateral_score_grf, contralateral_score_grf


def compute_assistive_score(state, sensordata) -> int:
    """
    Basically a score of how many arms assist in propulsion.

    This is WITHOUT batch dimension. Still needs to be vmapped afterwards. (use of jax compulsory)
    Based on ground reaction forces (GRF), computes how much every arm contributes to the forward movement throughout the entire episode.
    Every arms contribution is normalized over the most propulsive arm.
    The score is added together to get a score between 1 and 5 (narms)

    This is similar to the approach of Kano et al. (2017) A brittle star-like robot capable of immediately adapting to unexpected physical damage.
    Args:
        state [ndarray] (nsteps, nstate)
        sensordata [dict] (nsteps, nsensordata): The sensor data for the current cycle
    Returns:
        assistive_count [int]: The number of assistive arms
    """
    # compute unit XY direction vector of the central disk
    xy_dir = jnp.array([sensordata["central_disk"]["position"][-1, 0] - sensordata["central_disk"]["position"][0, 0],
                       sensordata["central_disk"]["position"][-1, 1] - sensordata["central_disk"]["position"][0, 1]])
    unit_xy_dir = (xy_dir / jnp.linalg.norm(xy_dir, axis=0)).T
    
    # compute cosine similarity between XY_velocity_vector and every segment for every timestep.
    # project the GRF vectors onto the XY direction vector and only keep positive values
    xy_grf_per_arm = compute_grf_per_arm(state, sensordata)  # shape (nsteps, narms, 2)
    propulsive_force = jnp.sum(xy_grf_per_arm * unit_xy_dir, axis=-1) # projection of GRF onto XY direction
    propulsive_force = jnp.where(propulsive_force > 0, propulsive_force, 0)  # only consider propulsive forces in the direction of movement

    # sum across time
    total_propulsive_force_per_arm = jnp.sum(propulsive_force, axis=0)  # shape (narms,)

    # normalize by the most propulsive arm
    max_propulsive_force = jnp.max(total_propulsive_force_per_arm)
    normalized_propulsive_force_per_arm = total_propulsive_force_per_arm / max_propulsive_force


    assistive_score = jnp.sum(normalized_propulsive_force_per_arm)
    return assistive_score, normalized_propulsive_force_per_arm



if __name__ == "__main__":
    # Example usage
    import numpy as np
    from pprint import pprint
    from cpg_convergence.utils import load_dict_from_pickle

    
    # Load state and sensordata
    # state = load_dict_from_pickle('path_to_state.pkl')
    sensordata = load_dict_from_pickle("example_data/sensordata_example.pkl")
    mock_state = np.zeros(sensordata["ip_joints"]["position"].shape) 

    # test on nbatch != 1
    nbatch = 2
    sensordata_tiled = tree.tree_map(lambda x: np.tile(x, [nbatch if i == 0 else 1 for i in range(x.ndim)]), sensordata)
    mock_state_tiled = np.tile(mock_state, [nbatch if i == 0 else 1 for i in range(mock_state.ndim)])
    
    pprint(jax.tree_util.tree_map(lambda x: x.shape, sensordata_tiled))

    # Initialize MetricsExtractor
    metrics = BehavioralDescriptorsExtractor(
        mock_state_tiled,
        sensordata_tiled,
        steps_to_omit_transient=15,
        arm_setup=[5, 5, 5, 5, 5]  # 5-armed brittle star
    )

    xy_velocity_norm = metrics.XY_velocity_norm
    print("XY Velocity Norm shape:", xy_velocity_norm.shape)

    print(f"central_limb: {metrics.central_limb}")
    print(f"arm_roles: {metrics.arm_roles}")
    print(f"arm roles shape: {jax.tree_util.tree_map(lambda x: x.shape, metrics.arm_roles)}")

    print(f"leading or trailing: {metrics.leading_or_trailing}")

    print(f"bilateral_contralateral_score: {metrics.bilateral_contralateral_score}")
    print(f"bilateral_score: {metrics.bilateral_score}")
    print(f"contralateral_score: {metrics.contralateral_score}")
    print(f"bilateral_score_grf: {metrics.bilateral_score_grf}")
    print(f"contralateral_score_grf: {metrics.contralateral_score_grf}")

    print(f"assistive_score: {metrics.assistive_score}")


    print(f"ground_contact_fraction: {metrics.ground_contact_fraction}")

    # plot_timearray(sensordata_tiled["central_disk"]["position"][0, :, 2], title="Central Disk z posiition", ylabel="z position")
    print(f"disk_elevation: {metrics.disk_elevation}")

    print(f"sine_total_displacement: {metrics.sine_total_displacement}")
    print(f"cosine_total_displacement: {metrics.cosine_total_displacement}")

    print("Script finished.")



