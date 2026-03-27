from typing import Callable, Literal, Tuple, Optional, Union, List
import functools
import sys
import os
import subprocess
import copy
import numpy as np
import shutil

import jax
import jax.numpy as jnp
from flax import struct
import chex
from evosax import ParameterReshaper

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import networkx as nx

from cpg_convergence.utils import clip_and_rescale
from cpg_convergence.defaults import CPG_DEFAULT_AMPLITUDE_GAIN, CPG_DEFAULT_OFFSET_GAIN, CPG_DEFAULT_WEIGHT_SCALE,\
    CPG_DEFAULT_DT, CPG_DEFAULT_SOLVER, OMEGA, PARAM_CLIP_MIN, PARAM_CLIP_MAX,\
    X_SCALE_MIN, X_SCALE_MAX, R_SCALE_MIN, R_SCALE_MAX, OMEGA_SCALE_MIN, OMEGA_SCALE_MAX, \
    CONTROL_TIMESTEP, DISABLE_RANDOMNESS, CPG_RESET_PHASE_RANGES


def euler_solver(
        current_time: float,
        y: float,
        derivative_fn: Callable[[float, float], float],
        delta_time: float
        ) -> float:
    slope = derivative_fn(current_time, y)
    next_y = y + delta_time * slope
    return next_y


def rk4_solver(
        current_time: float,
        y: float,
        derivative_fn: Callable[[float, float], float],
        delta_time: float
        ) -> float:
    # This is the original euler
    slope1 = derivative_fn(current_time, y)
    # These are additional slope calculations that improve our approximation of the true slope
    slope2 = derivative_fn(current_time + delta_time / 2, y + slope1 * delta_time / 2)
    slope3 = derivative_fn(current_time + delta_time / 2, y + slope2 * delta_time / 2)
    slope4 = derivative_fn(current_time + delta_time, y + slope3 * delta_time)
    average_slope = (slope1 + 2 * slope2 + 2 * slope3 + slope4) / 6
    next_y = y + average_slope * delta_time
    return next_y


@struct.dataclass # automatically .__init__, .__repr__, .replace
class CPGState:
    time: float
    phases: jnp.ndarray
    dot_amplitudes: jnp.ndarray  # first order derivative of the amplitude
    amplitudes: jnp.ndarray
    dot_offsets: jnp.ndarray  # first order derivative of the offset
    offsets: jnp.ndarray
    outputs: jnp.ndarray

    # We'll make these modulatory parameters part of the state as they will change as well
    R: jnp.ndarray
    X: jnp.ndarray
    omegas: jnp.ndarray # radians per second
    rhos: jnp.ndarray

    # Sometimes, CPGs have specific adjacency matrix modifications, so we can store them here if needed
    weights: jnp.ndarray


class CPG:
    def __init__(
            self,
            adjacency_matrix: jnp.ndarray,
            weight_scale: float = CPG_DEFAULT_WEIGHT_SCALE,
            amplitude_gain: float = CPG_DEFAULT_AMPLITUDE_GAIN,
            offset_gain: float = CPG_DEFAULT_OFFSET_GAIN,
            dt: float = CPG_DEFAULT_DT,
            solver: Literal["euler", "rk4"] = CPG_DEFAULT_SOLVER,
            simplified_phase_de: bool = False
            ) -> None:
        self.weight_scale = weight_scale
        self.weights = weight_scale * adjacency_matrix
        self.weights_original = copy.deepcopy(self.weights)
        self._amplitude_gain = amplitude_gain
        self._offset_gain = offset_gain
        self._dt = dt
        self._simplified_phase_de = simplified_phase_de
        assert solver in ["euler", "rk4"], f"'solver' must be one of ['euler', 'rk4']"

        if solver == "euler":
            self._solver = euler_solver
        else:
            self._solver = rk4_solver

    def update_weights_with_rhos(self, rhos: jnp.ndarray) -> None:
        """
        Update the weights with new phase biases (rhos).
        Args:
            rhos (jnp.ndarray): New phase biases for the oscillators.
        """
        assert rhos.shape == self.weights.shape, f"rhos must have shape weights: {self.weights.shape}"
        adjacency_matrix = jnp.where(rhos != 0, 1.0, 0.0)
        self.weights = self.weight_scale * adjacency_matrix


    @property
    def num_oscillators(
            self
            ) -> int:
        return self.weights.shape[0]
    
    @property
    def num_couplings(
            self
            ) -> int:
        if len(self.weights.shape) != 2:
            return jnp.sum(jnp.tril(self.weights[0]) != 0)
        else:
            return jnp.sum(jnp.tril(self.weights) != 0)
    

    @staticmethod
    def phase_de(
            weights: jnp.ndarray,
            amplitudes: jnp.ndarray,
            phases: jnp.ndarray,
            phase_biases: jnp.ndarray,
            omegas: jnp.ndarray,
            simplified_phase_de: bool = False
            ) -> jnp.ndarray:
        # original as used by Sproewitz and Ijspeert
        if simplified_phase_de == False: 
            @jax.vmap  # vectorizes this function for us over an additional batch dimension (in this case over all oscillators)
            def sine_term(
                phase_i: float,
                phase_biases_i: float
                ) -> jnp.ndarray:
                return jnp.sin(phases - phase_i - phase_biases_i)
            couplings = jnp.sum(weights * amplitudes * sine_term(phase_i=phases, phase_biases_i=phase_biases), axis=1)
            return omegas + couplings
        
        # alternative for possibly improved phase bias convergence
        elif simplified_phase_de == True:
            @jax.vmap  # vectorizes this function for us over an additional batch dimension (in this case over all oscillators)
            def sine_term(
                    phase_i: float,
                    phase_biases_i: float
                    ) -> jnp.ndarray:

                # return jnp.sin((phases - phase_i - phase_biases_i)/2) # so that a phase difference of pi is not reduced to 0. Problem, now 2pi is 0 again.
                return jnp.sin(phases - phase_i - phase_biases_i)

            # couplings = jnp.sum(weights * amplitudes * sine_term(phase_i=phases, phase_biases_i=phase_biases), axis=1)
            couplings = jnp.sum(weights * sine_term(phase_i=phases, phase_biases_i=phase_biases), axis=1)
            return omegas + couplings
        

    @staticmethod
    def second_order_de(
            gain: jnp.ndarray,
            modulator: jnp.ndarray,
            values: jnp.ndarray,
            dot_values: jnp.ndarray
            ) -> jnp.ndarray:

        return gain * ((gain / 4) * (modulator - values) - dot_values)

    @staticmethod
    def first_order_de(
            dot_values: jnp.ndarray
            ) -> jnp.ndarray:
        return dot_values

    @staticmethod
    def output(
            offsets: jnp.ndarray,
            amplitudes: jnp.ndarray,
            phases: jnp.ndarray
            ) -> jnp.ndarray:
        return offsets + amplitudes * jnp.cos(phases)

    def reset(
            self,
            rng: chex.PRNGKey=None,
            cpg_reset_phase_ranges: Tuple[float, float]=CPG_RESET_PHASE_RANGES,
            disable_randomness: bool=DISABLE_RANDOMNESS,
            ) -> CPGState:
        """
        Generate a novel initialized CPGState object.
        Args:
            rng: chex.PRNGKey, optional -> not required if disable randomness == True
            phase_init_range: Tuple[float, float], optional -> range for uniform initialization of the phases of the oscillators
            If disable_randomness is True, the rng key is ignored and a fixed seed=0 is used.

        """
        if disable_randomness == False:
            assert rng is not None, "If disable_randomness is False, a valid rng key must be provided"
            phase_rng = rng
    
        if disable_randomness == True:
            phase_rng = jax.random.PRNGKey(0)

        phases=jax.random.uniform(
            key=phase_rng, shape=(self.num_oscillators,), dtype=jnp.float32,
            minval=cpg_reset_phase_ranges[0], maxval=cpg_reset_phase_ranges[1]
            )    

        # noinspection PyArgumentList
        state = CPGState(
                phases=phases,
                amplitudes=jnp.zeros(self.num_oscillators),
                offsets=jnp.zeros(self.num_oscillators),
                dot_amplitudes=jnp.zeros(self.num_oscillators),
                dot_offsets=jnp.zeros(self.num_oscillators),
                outputs=jnp.zeros(self.num_oscillators),
                time=0.0,
                R=jnp.zeros(self.num_oscillators),
                X=jnp.zeros(self.num_oscillators),
                omegas=jnp.zeros(self.num_oscillators),
                rhos=jnp.zeros_like(self.weights),
                weights=self.weights
                )
        return state
    
    @staticmethod
    def modulate(
            state: CPGState,
            R: jnp.ndarray = None,
            X: jnp.ndarray = None,
            omegas: jnp.ndarray = None,
            rhos: jnp.ndarray = None
            ) -> CPGState:
        """
        Modulate the CPG state with new parameters. They can change at any moment,
        because they only affect the steady state solution of the CPG.
        Args:
            state (CPGState): The current state of the CPG.
            R (jnp.ndarray (num_oscillators,), optional): Modulation for the amplitude. Defaults to None.
            X (jnp.ndarray (num_oscillators,), optional): Modulation for the offset. Defaults to None.
            omegas (jnp.ndarray ([1 or num_oscillators],), optional): New frequencies for the oscillators. Defaults to None.
            rhos (jnp.ndarray (num_oscillators, num_oscillators), optional): New phase biases for the oscillators. Defaults to None.
        Returns:
            CPGState: The updated CPG state with the new modulation parameters.
        """
        state = state.replace(R=R)
        state = state.replace(X=X)
        state = state.replace(omegas=omegas)
        state = state.replace(rhos=rhos)
        return state
    
    @property
    def modulation_params_example_empty(
            self,
            ) -> dict:
        """
        Returns a modulation parameter dictionary with the correct shape, filled with zeros.
        This can be used for the evosax.ParameterReshaper
        The modulation dict contains the following keys:
        - R: jnp.ndarray (1D) of shape (num_oscillators,) - Amplitudes for the oscillators
        - X: jnp.ndarray (1D) of shape (num_oscillators,) - Offsets for the oscillators
        - omegas: jnp.ndarray (1D) of shape (num_oscillators,) - Frequencies (in radians per second)
        - rhos: jnp.ndarray (2D) of shape (num_oscillators, num_oscillators) - Phase biases for the oscillators
        """
        return {
            "R": jnp.zeros(self.num_oscillators),
            "X": jnp.zeros(self.num_oscillators),
            "omegas": jnp.zeros(self.num_oscillators),  # frequencies in radians per second
            "rhos": jnp.zeros_like(self.weights)
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            state: CPGState
            ) -> CPGState:
        # Update phase
        new_phases = self._solver(
                current_time=state.time,
                y=state.phases,
                derivative_fn=lambda
                    t,
                    y: self.phase_de(
                        omegas=state.omegas,
                        amplitudes=state.amplitudes,
                        phases=y,
                        phase_biases=state.rhos,
                        weights=state.weights,
                        simplified_phase_de=self._simplified_phase_de
                        ),
                delta_time=self._dt
                )
        new_dot_amplitudes = self._solver(
                current_time=state.time,
                y=state.dot_amplitudes,
                derivative_fn=lambda
                    t,
                    y: self.second_order_de(
                        gain=self._amplitude_gain, modulator=state.R, values=state.amplitudes, dot_values=y
                        ),
                delta_time=self._dt
                )
        new_amplitudes = self._solver(
                current_time=state.time,
                y=state.amplitudes,
                derivative_fn=lambda
                    t,
                    y: self.first_order_de(dot_values=state.dot_amplitudes),
                delta_time=self._dt
                )
        new_dot_offsets = self._solver(
                current_time=state.time,
                y=state.dot_offsets,
                derivative_fn=lambda
                    t,
                    y: self.second_order_de(
                        gain=self._offset_gain, modulator=state.X, values=state.offsets, dot_values=y
                        ),
                delta_time=self._dt
                )
        new_offsets = self._solver(
                current_time=0,
                y=state.offsets,
                derivative_fn=lambda
                    t,
                    y: self.first_order_de(dot_values=state.dot_offsets),
                delta_time=self._dt
                )

        new_outputs = self.output(offsets=new_offsets, amplitudes=new_amplitudes, phases=new_phases)
        # noinspection PyUnresolvedReferences
        return state.replace(
                phases=new_phases,
                dot_amplitudes=new_dot_amplitudes,
                amplitudes=new_amplitudes,
                dot_offsets=new_dot_offsets,
                offsets=new_offsets,
                outputs=new_outputs,
                time=state.time + self._dt
                )
    
    @property
    def laplacian_matrix(self):
        """Compute the Laplacian of the weights matrix. -> Always based on original weights, before modulating strengths and making it antisymmetric"""
        weights_laplacian = jnp.abs(self.weights)
        degree_matrix = jnp.diag(jnp.sum(weights_laplacian, axis=1))
        laplacian_matrix = degree_matrix - weights_laplacian
        return laplacian_matrix

    @property
    def eigenvalues_of_laplacian(self):
        """Compute the eigenvalues of the Laplacian of the weights matrix."""
        eigenvalues = jnp.linalg.eigvals(self.laplacian_matrix)
        return eigenvalues
    
    @property
    def spectral_gap(self):
        """Compute the spectral gap of the weights matrix."""
        eigenvalues = jnp.sort(jnp.real(self.eigenvalues_of_laplacian))
        return eigenvalues[1] - eigenvalues[0] # second smallest - smallest eigenvalue, eigenvalue[0] is 0 for connected graphs

    @property
    def smallest_eigenvalue(self):
        """Compute the smallest eigenvalue of the Laplacian of the weights matrix."""
        eigenvalues = jnp.real(self.eigenvalues_of_laplacian)
        return jnp.min(eigenvalues)
        
    @property
    def max_eigenvalue(self):
        """Compute the maximum eigenvalue of the Laplacian of the weights matrix."""
        eigenvalues = jnp.real(self.eigenvalues_of_laplacian)
        return jnp.max(eigenvalues)
    
    @property
    def induced_norm(self):
        """Compute the induced norm (2-norm) of the Laplacian matrix.
        This is the largest spectral value (singular value) of the Laplacian matrix.
        This is the same as Spectral norm (but not the same as spectral radius)"""
        singular_values = jnp.linalg.svd(self.laplacian_matrix, compute_uv=False)
        return jnp.max(singular_values)
    
    @property
    def is_laplacian_symmetric(self) -> bool:
        """Check if the Laplacian matrix is symmetric."""
        laplacian = self.laplacian_matrix
        return jnp.allclose(laplacian, laplacian.T)
    
    @property
    def are_all_eigenvalues_real(self) -> bool:
        """Check if all eigenvalues of the Laplacian matrix are real."""
        eigenvalues = self.eigenvalues_of_laplacian
        return jnp.all(jnp.isreal(eigenvalues))


@jax.vmap
def replace_weights_across_batch(state: CPGState, weights: jnp.ndarray) -> CPGState:
    """
    Core function: replaces weight (2D) in 1 CPGState object
    After vmapping, this function can replace weights for a batch of CPGState objects
    """
    return state.replace(weights=weights)
    
    

class CPG_Ring_Arms(CPG):
    """
    CPG class specifically for morphologies with a central ring and arm-like appendices.
    Inherits from the CPG class, so it supports solving differential equations.
    7 Methods are supported:
    - "base": base CPG with connections between ganglion oscillators in the ring and in the arms
    - "cobweb": cobweb CPG with additional connections between oscillators of neighbouring arms
    - "fully_connected": fully connected CPG with all-to-all connections between all oscillators
    - "leader_follower": leader-follower CPG with fully connected ring and unidirectional connections in the arms
    - "popularity": popularity CPG, where weights are scaled by how connected nodes are.
    - "modified_de": base CPG with simplified phase differential equation
    - "ratio_couplings_oscillators": ratio of n_couplings to n_oscillators
    """
    def __init__(
            self,
            ring_setup: list | jnp.ndarray,
            rng: chex.PRNGKey,
            method: Literal["base", "cobweb", "fully_connected", "leader_follower", "popularity", "modified_de", "ratio_couplings_oscillators"] = "base",
            ratio_couplings_oscillators: Optional[int] | Literal["max"] = None, # only used if method=="ratio_couplings_oscillators", min 1.0
            weight_scale: float = CPG_DEFAULT_WEIGHT_SCALE,
            amplitude_gain: float = CPG_DEFAULT_AMPLITUDE_GAIN,
            offset_gain: float = CPG_DEFAULT_OFFSET_GAIN,
            dt: float = CPG_DEFAULT_DT,
            solver: Literal["euler", "rk4"] = CPG_DEFAULT_SOLVER,
            omega: float | None = OMEGA,
            ) -> None:
        """
        Initialize the BS_CPG with the ring_setup.
        main difference to CPG class is that it requires only ring_setup insetad of adjacency_matrix.
        Args:
            ring_setup (list|jnp.ndarray): list specifying number of segments per arm,
                e.g. [5, 5, 5, 5, 5] for a 5-segment arm with 5 node ring.
            method (str): method to create the adjacency matrix, one of ["base", "cobweb", "fully_connected", "leader_follower", "popularity", "modified_de", "ratio_couplings_oscillators"]
            weight_scale (float): weight scale for the CPG connections.
            amplitude_gain (float): gain for amplitude modulation.
            offset_gain (float): gain for offset modulation.
            dt (float): time step for the CPG solver.
            solver (str): solver to use for the CPG, one of ["euler", "rk4"].
            modified_de (bool): whether to use the simplified phase differential equation.
            omega (float): base frequency for the CPG oscillators in radians per second, can be None
        """
        self.ring_setup = jnp.array(ring_setup)
        self.method = method
        self.noscillators = len(ring_setup) + sum(ring_setup)
        self.omega = omega
        self.ratio_couplings_oscillators = ratio_couplings_oscillators

        # ratio couplings to oscillators edge cases:
        if method == "ratio_couplings_oscillators":
            assert ratio_couplings_oscillators is not None, "If method is 'ratio_couplings_oscillators', ratio_couplings_oscillators must be provided."
            if ratio_couplings_oscillators == "max":
                self.method = "fully_connected"
            else:
                assert ratio_couplings_oscillators >= 1, f"ratio_couplings_oscillators must be between 1"
                assert isinstance(ratio_couplings_oscillators, int), "ratio_couplings_oscillators must be an integer value."

                if ratio_couplings_oscillators == 1:
                    self.method = "base"
                elif ratio_couplings_oscillators >= (self.noscillators -1)/2:
                    self.method = "fully_connected"
                    print(f"since ratio_couplings_oscillators >= (n_oscillators-1)/2 (in this case {(self.noscillators -1)/2}), using fully_connected method instead.")
                else:
                    pass

        rng, rng_random_modulation_params, rng_cpg_state = jax.random.split(rng, 3)
        self.set_random_modulation_params(rng_random_modulation_params)

        adjacency_matrix_init = jnp.where(self.modulation_params["rhos"][0] != 0, 1.0, 0.0)

        if method == "modified_de":
            modified_de = True
        else:
            modified_de = False



        # Construct the superior CPG once the other functionalities are correctly accounted for.
        super().__init__(
                adjacency_matrix=adjacency_matrix_init,
                weight_scale=weight_scale,
                amplitude_gain=amplitude_gain,
                offset_gain=offset_gain,
                dt=dt,
                solver=solver,
                simplified_phase_de=modified_de
                )
        
    def _get_independent_modulation_params_dict_empty(self) -> dict:
        """
        Returns a modulation parameter dictionary with the correct shape, filled with zeros.
        This can be used for the evosax.ParameterReshaper
        The modulation dict contains the following keys:
        - R: jnp.ndarray (1D) of shape (num_oscillators,) - Amplitudes for the oscillators
        - X: jnp.ndarray (1D) of shape (num_oscillators,) - Offsets for the oscillators
        - omegas: jnp.ndarray (1D) of shape (num_oscillators,) - Frequencies (in radians per second)
        - rhos: the independent phase biases are the ones required to add every node to the tree structure,
                so shape is (num_oscillators-1,) as the root node has no phase bias
        """
        if self.omega is None:
            return {
                "R": jnp.zeros(self.noscillators),
                "X": jnp.zeros(self.noscillators),
                "omegas": jnp.zeros(1),  # optimize the frequency used by all oscillators
                "rhos": jnp.zeros(self.noscillators - 1)
            }
            
        else:
            return {
                "R": jnp.zeros(self.noscillators),
                "X": jnp.zeros(self.noscillators),
                "rhos": jnp.zeros(self.noscillators - 1)
            }
        
    @property
    def control(self):
        "shape (nbatch, nsteps, noscillators). Concatenate future timesteps along axis=1."
        if not hasattr(self, '_control'):
            self.reset_control()
        return self._control
    
    @property
    def phases(self):
        "shape (nbatch, nsteps, noscillators). Concatenate future timesteps along axis=1."
        if not hasattr(self, '_phases'):
            self.reset_phases()
        return self._phases
    
        
    @property
    def clean_adjacency_matrix(self) -> chex.Array:
        """Clean adjacency just has ring with arms connections, no additional connections."""
        if not hasattr(self, '_clean_adjacency_matrix'):
            self._clean_adjacency_matrix = self._get_clean_adjacency_matrix()
        return self._clean_adjacency_matrix    

        
    def _get_clean_adjacency_matrix(self) -> chex.Array:
        """
        Create a ring with arms adjacency matrix based on the ring_setup.
        Args:
            ring_setup (list): list specifying number of segments per arm,
            e.g. [5, 5, 5, 5, 5] for a 5-segment arm with 5 node ring.
        Returns:
            Adjacency matrix (chex.Array): adjacency matrix of the CPG, specifying the connections between oscillators.
            Adjacency matrix contains only 1's and 0's.
        """
        ring_setup = self.ring_setup
        ring_indices = jnp.arange(len(ring_setup))
        arm_indices = jnp.arange(len(ring_setup), self.noscillators)

        adjacency_matrix = jnp.zeros((self.noscillators, self.noscillators))
        # connect ring oscillators counterclockwise (clockwise happens at symmetry of adjacency matrix)
        adjacency_matrix = adjacency_matrix.at[ring_indices, jnp.roll(ring_indices, -1)].set(1.)


        # connect arms to ring
        if sum(ring_setup) > 0:
            # connect ring to arms
            first_osc_per_arm_indices = (jnp.cumsum(jnp.array([0] + list(ring_setup[:-1])))) + len(ring_setup)
            nonzero_arm_indices = jnp.nonzero(ring_setup)[0]
            first_osc_per_arm_indices = jnp.unique(first_osc_per_arm_indices)
            adjacency_matrix = adjacency_matrix.at[ring_indices[nonzero_arm_indices], first_osc_per_arm_indices[:len(nonzero_arm_indices)]].set(1.)

            # connect arms internally. Note: last index per arm does not connect to anything further
            arm_indices_to_delete = jnp.cumsum(ring_setup) + (len(ring_setup) - 1) # last index per arm
            arm_indices_without_last_arm_per_arm = jnp.delete(arm_indices, jnp.where(jnp.isin(arm_indices, arm_indices_to_delete))[0])
            adjacency_matrix = adjacency_matrix.at[arm_indices_without_last_arm_per_arm, arm_indices_without_last_arm_per_arm + 1].set(1.)

        # make adjacency matrix symmetric
        adjacency_matrix = jnp.maximum(adjacency_matrix, adjacency_matrix.T)  # ensure symmetry

        return adjacency_matrix
    

    def visualize_clean_adjacency_matrix(
            self,
            **kwargs
            ) -> None:
        """Visualize the clean adjacency matrix using networkx.
        kwargs:
            path (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.
            show_plot (bool, optional): Whether to show the plot. Defaults to False.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (15, 15).
        """
        visualize_connectivity(
            node_positions=self.graph_node_positions,
            ring_setup=self.ring_setup,
            adjacency_matrix=self.clean_adjacency_matrix,
            **kwargs
        )

    def visualize_modulated_adjacency_matrix(
            self,
            nbatch_index: int = 0,
            **kwargs
            ) -> None:
        """
        Visualize modulated adjacency matrix
        Visualize the modulated adjacency matrix using networkx.
        kwargs:
            path (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.
            show_plot (bool, optional): Whether to show the plot. Defaults to False.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (15, 15).
        """
        visualize_connectivity(
            node_positions=self.graph_node_positions,
            ring_setup=self.ring_setup,
            adjacency_matrix=jnp.where(self.modulation_params["rhos"][nbatch_index] != 0, 1.0, 0.0),
            **kwargs
        )

    def plot_clean_adjacency_matrix_heatmap(
            self,
            figsize: Optional[Tuple[int, int]] = None,
            **kwargs
        ) -> None:
        """
        Clean adjacency matrix is purely the ring with arms connections.
        Visualize the clean adjacency matrix as annotated heatmap.
        kwargs:
            path (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.
            show_plot (bool, optional): Whether to show the plot. Defaults to False.
        """
        if figsize is None:
            figsize = (self.noscillators // 2 + 4, self.noscillators // 2 + 4) 
        plot_matrix_as_annotated_heatmap(
            matrix=self.clean_adjacency_matrix,
            title="Clean Adjacency Matrix Heatmap",
            scale = False,
            figsize=figsize,
            **kwargs
        )

    def plot_modulated_adjacency_matrix_heatmap(
            self,
            nbatch_index: int = 0,
            figsize: Optional[Tuple[int, int]] = None,
            **kwargs
        ) -> None:
        """
        Visualize the modulated adjacency matrix as annotated heatmap.
        kwargs:
            path (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.
            show_plot (bool, optional): Whether to show the plot. Defaults to False.
        """
        if figsize is None:
            figsize = (self.noscillators // 2 + 4, self.noscillators // 2 + 4) 

        plot_matrix_as_annotated_heatmap(
            matrix=self.cpg_state.weights[nbatch_index],
            # jnp.where(self.modulation_params["rhos"][nbatch_index] != 0, 1.0, 0.0),
            title="Modulated Rhos Adjacency Matrix Heatmap",
            scale = False,
            figsize=figsize,
            **kwargs
        )

    def plot_modulated_rhos_matrix_heatmap(
            self,
            nbatch_index: int = 0,
            figsize: Optional[Tuple[int, int]] = None,
            **kwargs
        ) -> None:
        """
        Visualize the modulated rhos matrix as annotated heatmap.
        kwargs:
            path (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.
            show_plot (bool, optional): Whether to show the plot. Defaults to False.
        """
        if figsize is None:
            figsize = (self.noscillators // 2 + 4, self.noscillators // 2 + 4) 

        plot_matrix_as_annotated_heatmap(
            matrix=self.modulation_params["rhos"][nbatch_index],
            title="Modulated Phase Biases Heatmap",
            scale = True,
            scale_label="Phase Bias (radians)",
            figsize=figsize,
            **kwargs
        )


    @property
    def graph_node_positions(self):
        if not hasattr(self, '_graph_node_positions'):
            self._graph_node_positions = self.get_graph_node_positions()
        return self._graph_node_positions

    def get_graph_node_positions(self):
        matrix = self.clean_adjacency_matrix
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        pos = nx.kamada_kawai_layout(G) # better to prevent crossing and improve spacing
        return pos


    @property
    def parameter_reshaper(self) -> ParameterReshaper:
        """
        Returns an evosax.ParameterReshaper object for the CPG modulation parameters.
        The reshaper uses the independent modulation parameters dict to reshape the flat parameter vector.
        The ParameterReshaper.total_params can be used for sampling candidate solutions in evolutionary optimization.
        """
        if not hasattr(self, '_parameter_reshaper'):
            example_dict = self._get_independent_modulation_params_dict_empty()
            self._parameter_reshaper = ParameterReshaper(example_dict, verbose=False)
        return self._parameter_reshaper
    

    @property
    def modulation_params(self):
        """returns the full modulation params dict to pass on to the CPG.modulate() method"""
        try:
            return self._modulation_params
        except AttributeError:
            raise AttributeError("modulation_params have not been set yet. Use set_modulation_params() to set them first.")
        

    def print_modulation_params_ranges(self) -> None:
        """
        Prints the ranges of the modulation parameters to check if they are within expected limits.
        """
        print("CPG Modulation parameters ranges:")
        for key, value in self.modulation_params.items():
            if isinstance(value, jnp.ndarray) or isinstance(value, np.ndarray):
                print(f"{key}: {value.min()} to {value.max()}")
            else:
                print(f"{key}: {value}")
        

    def set_random_modulation_params(
        self,
        rng: chex.PRNGKey,
        nbatch: int = 1,
        mu: float = 0.0,
        sigma: float = 0.5,
    ) -> None:
        """
        set attribute self._modulation_params to random modulation params
        dimension of self.modulation_params will be:
        - R: (nbatch, num_oscillators)
        - X: (nbatch, num_oscillators)
        - omegas: (nbatch, num_oscillators)
        - rhos: (nbatch, num_oscillators, num_oscillators)
        """
        rng, rng_params = jax.random.split(rng)

        independent_params_flat = mu + sigma * jax.random.normal(rng_params, shape=(nbatch, self.parameter_reshaper.total_params))
        independent_params = self.parameter_reshaper.reshape(independent_params_flat)
        independent_params = _independent_params_clipping_and_rescaling(independent_params)

        if self.method == "ratio_couplings_oscillators":
            rng, rng_ratio = jax.random.split(rng)
            rng_ratio = jax.random.split(rng_ratio, nbatch)
        else:
            rng_ratio = None

        modulation_params = self._generate_modulation_params(
            independent_params,
            self.ring_setup,
            self.omega,
            self.method,
            self.ratio_couplings_oscillators,
            rng_ratio
        )

        self.nbatch = nbatch
        self.ncouplings = jnp.sum(jnp.tril(modulation_params["rhos"][0] != 0))
        self._modulation_params = modulation_params


    def set_modulation_params_from_independent_params(
            self,
            independent_params: dict | jnp.ndarray,
            returned_clipped_genotypes: bool = False,
            rng: chex.PRNGKey = None,
    ):
        """
        The independent params are the number of parameters required to define to modulate the CPG.
        Independent params are essentially the genotype, i.e. all the degrees of freedom that define the CPG
        They are:
        - R: jnp.ndarray (nbatch, 1D) of shape (num_oscillators,) - Amplitudes for the oscillators
        - X: jnp.ndarray (nbatch, 1D) of shape (num_oscillators,) - Offsets for the oscillators
        - omegas: jnp.ndarray (nbatch, 1D) of shape (1,) - Frequency (in radians per second) for all oscillators
        - rhos: jnp.ndarray (nbatch, 1D) of shape (num_oscillators - 1,) - Phase biases for the oscillators (in tree structure)

        The independent_params can be provided as a dict or as a flat jnp.ndarray of shape (nbatch, total_params).
        CLIPPING and RESCALING happens internally.

        rng required for method=="ratio_couplings_oscillators"
        """
        if isinstance(independent_params, (jnp.ndarray, np.ndarray)) or hasattr(independent_params, '__array__'):
            independent_params = self.parameter_reshaper.reshape(independent_params)
        independent_params = _independent_params_clipping_and_rescaling(independent_params)


        if self.method == "ratio_couplings_oscillators":
            rng, rng_ratio = jax.random.split(rng)
            rng_ratio = jax.random.split(rng_ratio, nbatch)
        else:
            rng_ratio = None

        modulation_params = self._generate_modulation_params(
            independent_params,
            self.ring_setup,
            self.omega,
            self.method,
            self.ratio_couplings_oscillators,
            rng_ratio
        )

        self.nbatch = modulation_params["R"].shape[0]
        self.ncouplings = jnp.sum(jnp.tril(modulation_params["rhos"][0] != 0))
        self._modulation_params = modulation_params

        if returned_clipped_genotypes:
            return self.parameter_reshaper.flatten(independent_params)


    @staticmethod
    @functools.partial(jax.vmap, in_axes=(0, None, None, None, None, 0), out_axes=0)
    def _generate_modulation_params(
            independent_params: dict,
            ring_setup: chex.Array,
            omega: float | None,
            method: Literal["base", "cobweb", "fully_connected", "leader_follower", "popularity", "modified_de", "ratio_couplings_oscillators"],
            ratio_couplings_oscillators: int | None,
            rng: chex.PRNGKey = None,
        ) -> dict:
        """
        Generate modulation params for the CPG_Ring_Arms structure.
        ratio_couplings_oscillators is only used if method=="ratio_couplings_oscillators"
        rng only required when method=="ratio_couplings_oscillators"
        """
        assert method in ["base", "cobweb", "fully_connected", "leader_follower", "popularity", "modified_de", "ratio_couplings_oscillators"], \
            f"method must be one of ['base', 'cobweb', 'fully_connected', 'leader_follower', 'popularity', 'modified_de', 'ratio_couplings_oscillators'], got {method}"
        
        modulation_params = _generate_tree_without_loops(independent_params, ring_setup, omega)
        modulation_params = _close_ring_connection_in_modulation_params(modulation_params, ring_setup)
        
        if method == "base":
            pass # no additional connections to add
        
        elif method == "cobweb":
            modulation_params = _add_cobweb_connections_to_modulation_params(modulation_params, ring_setup)
        
        elif method == "fully_connected":
            modulation_params = _add_full_connections_to_modulation_params(modulation_params, ring_setup)

        elif method == "ratio_couplings_oscillators":
            modulation_params = _add_ratio_couplings_connections_to_modulation_params(modulation_params, ring_setup, ratio_couplings_oscillators, rng)
        
        elif method == "leader_follower":
            modulation_params = _add_fully_connected_ring_to_modulation_params(modulation_params, ring_setup)
            modulation_params = _remove_radially_outward_connections_in_arms_from_modulation_params(modulation_params, ring_setup)
        

        elif method == "popularity":
            pass # no additional connections compared to base. Rhos will be scaled according to popularity
                 # during modulation step.
        
        elif method == "modified_de":
            pass # no additional connections compared to base. Simplified phase differential equation will be used.
        
        else:
            raise ValueError(f"method must be one of ['base', 'cobweb', 'fully_connected', 'leader_follower', 'popularity', 'modified_de'], got {method}")
        
        return modulation_params

    def reset_state(
            self,
            rng: chex.PRNGKey=None,
            cpg_reset_phase_ranges: Tuple[float, float]=CPG_RESET_PHASE_RANGES,
            disable_randomness: bool=DISABLE_RANDOMNESS,
            ) -> CPGState:
        """
        Generate a novel initialized CPGState object.
        Since it is vmapped, returns a batch of CPGState objects.
        Args:
            rng: chex.PRNGKey, optional -> not required if disable randomness == True
            phase_init_range: Tuple[float, float], optional -> range for uniform initialization of the phases of the oscillators
            If disable_randomness is True, the rng key is ignored and a fixed seed=0 is used.

        """
        if DISABLE_RANDOMNESS == False:
            assert rng is not None, "If disable_randomness is False, a valid rng key must be provided"
            rng = jax.random.split(rng, self.modulation_params["R"].shape[0])  # split rng for each batch element

        state = _reset_state_vectorized(
            rng,
            self,
            cpg_reset_phase_ranges,
            disable_randomness
        )
        self.cpg_state = state


    def reset_control(self) -> None:
        """
        Seperate function, since it can be used independent of state resetting.
        """
        assert hasattr(self, 'nbatch'), "nbatch has not been set yet. Use set_modulation_params() or set_random_modulation_params() to set it first."
        self._control = jnp.empty((self.nbatch, 0, self.noscillators))

    def _append_control(
            self,
            new_control: jnp.ndarray
            ) -> None:
        """
        New control must have shape (nbatch, nsteps, noscillators)
        """   
        assert new_control.ndim == 3, "new_control must have 3 dimensions (nbatch, nsteps, noscillators)"
        assert new_control.shape[0] == self.nbatch, f"new_control must have nbatch={self.nbatch} as first dimension"
        assert new_control.shape[2] == self.noscillators, f"new_control must have noscillators={self.noscillators} as last dimension"
        self._control = jnp.concatenate((self._control, new_control), axis=1)


    def reset_phases(self):
        """
        Reset the self.phases of the current CPG state.
        The phases are relevant for assessing the quality of the convergence.
        """
        assert hasattr(self, 'nbatch'), "nbatch has not been set yet. Use set_modulation_params() or set_random_modulation_params() to set it first."
        self._phases = jnp.empty((self.nbatch, 0, self.noscillators))

    def _append_phases(
            self,
            new_phases: jnp.ndarray
            ) -> None:
        """
        New phases must have shape (nbatch, nsteps, noscillators)
        """   
        assert new_phases.ndim == 3, "new_phases must have 3 dimensions (nbatch, nsteps, noscillators)"
        assert new_phases.shape[0] == self.nbatch, f"new_phases must have nbatch={self.nbatch} as first dimension"
        assert new_phases.shape[2] == self.noscillators, f"new_phases must have noscillators={self.noscillators} as last dimension"
        self._phases = jnp.concatenate((self._phases, new_phases), axis=1
    )


    def modulate_state(self) -> None:
        """
        Modulate the CPG state with the current modulation parameters
        stored as an CPG_Ring_attribute.
        Since it is vmapped, expects a batch of CPGState objects.
        Args:
            state (CPGState): The current state of the CPG.
        Returns:
            CPGState: The updated CPG state with the new modulation parameters.
        """
        assert hasattr(self, 'cpg_state'), "cpg_state has not been initialized yet. Use reset_state() to initialize it first."

        state = _modulate_state_vectorized(
            self.cpg_state,
            self.modulation_params,
            self,
        )
        self.cpg_state = state


    def step_state_n_times(
            self,
            nsteps: int,
            popularity_normalization_factor: int = 1,
            ) -> None:
        """
        Step the CPG state n times.
        Since it is vmapped, expects a batch of CPGState objects.
        Args:
            nsteps (int): Number of steps to take.
        popularity_normalization_factor (str): Normalization factor for popularity: if the difference in subnetwork is equal to this number,
                    the popularity factor will be 0.37.
        """
        if self.method == "popularity":
            assert popularity_normalization_factor is not None, "popularity_normalization_factor must be provided for 'popularity' method."
            self._step_state_n_times_popularity(nsteps)
        
        else:
            state = self.cpg_state
            vectorized_cpg_step = jax.vmap(self.step, in_axes=(0,), out_axes=0)

            def step(carry, _):
                """step function for jax.lax.scan
                Returns both control (outputs) and phases for concatenation
                """
                cpg_state = carry
                cpg_state = vectorized_cpg_step(cpg_state)
                return cpg_state, (cpg_state.outputs, cpg_state.phases)

            final_state, (control, phases) = jax.lax.scan(step, state, None, length=nsteps)
            # print(f"Stepped CPG for {nsteps} steps. Control shape: {control.shape}, Phases shape: {phases.shape}")
            self.cpg_state = final_state
            self._append_control(jnp.swapaxes(control, 0, 1))  # (nsteps, nbatch, noscillators) -> (nbatch, nsteps, noscillators)
            self._append_phases(jnp.swapaxes(phases, 0, 1))     # (nsteps, nbatch, noscillators) -> (nbatch, nsteps, noscillators)


    def _step_state_n_times_popularity(self, nsteps: int) -> None:
        """
        Step the CPG state n times.
        Since it is vmapped, expects a batch of CPGState objects.
        Args:
            nsteps (int): Number of steps to take.
        """
        vectorized_cpg_step = jax.vmap(self.step, in_axes=(0,), out_axes=0)
        cpg_state = self.cpg_state

        for _ in range(nsteps):
            # step once
            cpg_state = vectorized_cpg_step(cpg_state)

            # store control and phases
            control_single_step = cpg_state.outputs
            phases_single_step = cpg_state.phases
            self._append_control(control_single_step[:, jnp.newaxis, :])  # (nbatch, 1, noscillators)
            self._append_phases(phases_single_step[:, jnp.newaxis, :])

            # update weights based on popularity
            popularity_factors = self.get_popularity_factor_per_node() # shape (nbatch, noscillators)
            weights_original = copy.deepcopy(self.weights_original) # already scaled with weight_scale (noscillators, noscillators)
            weights_tiled = jnp.tile(weights_original[None, :, :], (self.nbatch, 1, 1))  # shape (nbatch, noscillators, noscillators)
            new_weights = weights_tiled * popularity_factors[:, None, :]   # shape (nbatch, noscillators, noscillators)
            cpg_state = replace_weights_across_batch(cpg_state, new_weights)

        self.cpg_state = cpg_state
        print(f"Stepped CPG for {nsteps} steps. Control shape: {self.control.shape}, Phases shape: {self.phases.shape}")


    def count_phase_mismatches(
            self,
            error_threshold: float = 1e-1,
            get_pairs: bool = False,
            time_index: int = -1,
            ): # -> Tuple[chex.Array[int], chex.Array[float], list, list]:
        """
        Only considers difference between current state and modulation params.
        self.cpg_state and self.modulation_params must be set and are assumed to have nbatch dim.
        If get_pairs is False, return None as third and fourth outputs.
        Returns:
            count_phase_mismatches: (nbatch,) int
            relative_fraction_mismatches: (nbatch,) float
            mismatched_pairs_list: list length nbatch; each element is (2, k_i) array of indices
            mismatched_errors_list: list length nbatch; each element is (k_i,) array of error values
        """
        phases_final = self.phases[:,time_index,:]  # shape (nbatch, noscillators)
        modulation_params = self.modulation_params

        @jax.vmap
        def _count_phase_mismatches_single(
                phases: chex.Array,
                modulation_params: dict,
                ): # -> Tuple[chex.Array[int], chex.Array[float], chex.Array[bool], chex.Array]:
            """
            No batch dimension. Uses static shapes.
            phases: (noscillators,) (final timestep)
            modulation_params: dict with "rhos": (noscillators, noscillators)
            """
            n = modulation_params["rhos"].shape[0]
            pre_idx, post_idx = jnp.tril_indices(n)  # all lower-tri pairs

            rhos_lower = modulation_params["rhos"][pre_idx, post_idx]
            connections_mask = rhos_lower != 0.0
            ncouplings = jnp.sum(connections_mask)

            phase_diffs = (phases[pre_idx] - phases[post_idx]) % (2 * jnp.pi)
            modulation_rhos = -rhos_lower
            difference = jnp.abs(phase_diffs - modulation_rhos) % (2 * jnp.pi)

            mismatched_mask = (difference > error_threshold) & (difference < (2 * jnp.pi - error_threshold)) & connections_mask
            count_phase_mismatches = jnp.sum(mismatched_mask)
            relative_fraction_mismatches = jnp.where(
                ncouplings > 0,
                count_phase_mismatches / ncouplings,
                0.0
            )
            return count_phase_mismatches, relative_fraction_mismatches, mismatched_mask, difference

        counts, fracs, masks, differences = _count_phase_mismatches_single(phases_final, modulation_params)

        if get_pairs == True:
            # post-process on host: extract ragged mismatched pairs per batch
            mismatched_pairs_list = []
            mismatched_errors_list = []
            n = modulation_params["rhos"].shape[-1]
            pre_idx_full, post_idx_full = jnp.tril_indices(n)
            masks_np = np.asarray(masks)
            differences_np = np.asarray(differences)
            
            for b in range(masks_np.shape[0]):
                mask_b = masks_np[b]
                pre_b = np.asarray(pre_idx_full)[mask_b]
                post_b = np.asarray(post_idx_full)[mask_b]
                errors_b = differences_np[b][mask_b]
                
                mismatched_pairs_list.append(np.stack([pre_b, post_b], axis=0))  # shape (2, k_b)
                mismatched_errors_list.append(errors_b)  # shape (k_b,)
        else:
            mismatched_pairs_list = None
            mismatched_errors_list = None

        return counts, fracs, mismatched_pairs_list, mismatched_errors_list
    

    def get_popularity_factor_per_node(
            self,
            popularity_factor_normalization: int = 1,
            error_threshold: float = 1e-1,
            time_index: int = -1,) -> chex.Array:
        """
        A value between 0. and 1. of how much of the maximum possible weight will be used in the matrix.
        Dimensions of output: (nbatch, noscillators)
        The popularity factor is calculated based on the size of the subnetwork each node belongs to,
        """
        
        subnetwork_sizes = self.get_size_of_subnetwork_per_node(
            error_threshold=error_threshold,
            time_index=time_index
        )
        difference_from_max = jnp.max(subnetwork_sizes, axis=1, keepdims=True) - subnetwork_sizes

        popularity_factors = popularity_factor(difference_from_max, alpha=popularity_factor_normalization)
        return popularity_factors

    def get_size_of_subnetwork_per_node(
            self,
            error_threshold: float = 1e-1,
            time_index: int = -1,
    ) -> chex.Array:
        """
        Calculate the popularity (subnetwork size) for each node across all batch elements.
        Uses vectorized operations for efficiency.
        
        Args:
            error_threshold (float): Threshold for phase mismatch detection. Defaults to 1e-1.
            time_index (int): Timestep to analyze (-1 for final). Defaults to -1.
            
        Returns:
            chex.Array: Array of shape (nbatch, noscillators) where each element is the size
                    of the subnetwork that node belongs to.
        """
        # Get mismatched pairs at this timestep
        _, _, mismatched_pairs, _ = self.count_phase_mismatches(
            error_threshold=error_threshold,
            get_pairs=True,
            time_index=time_index
        )
        
        def compute_popularity_single_batch(batch_idx):
            """Compute popularity for a single batch element"""
            # Extract mismatched pairs for this batch
            if mismatched_pairs is not None and batch_idx < len(mismatched_pairs):
                mismatched_pairs_batch = mismatched_pairs[batch_idx]
                if mismatched_pairs_batch.shape[1] > 0:
                    phase_mismatches = [
                        (int(mismatched_pairs_batch[0, i]), int(mismatched_pairs_batch[1, i]))
                        for i in range(mismatched_pairs_batch.shape[1])
                    ]
                else:
                    phase_mismatches = []
            else:
                phase_mismatches = []
            
            # Get adjacency matrix from modulation params
            adjacency_matrix = jnp.where(
                self.modulation_params["rhos"][batch_idx] != 0,
                1.0,
                0.0
            )
            
            # Build undirected graph from adjacency matrix
            matrix = np.asarray(adjacency_matrix)
            edges_u, edges_v = np.nonzero(np.abs(matrix) > 0.0)
            G = nx.Graph()
            G.add_nodes_from(range(self.noscillators))
            
            # Add edges from both triangles (undirected)
            for u, v in zip(edges_u, edges_v):
                if u != v:
                    G.add_edge(int(u), int(v))
            
            # Remove mismatched edges
            mismatched_edges_set = set()
            for (a, b) in phase_mismatches:
                ia = int(a)
                ib = int(b)
                e = (min(ia, ib), max(ia, ib))
                mismatched_edges_set.add(e)
            
            G_clean = G.copy()
            for (u, v) in mismatched_edges_set:
                if G_clean.has_edge(u, v):
                    G_clean.remove_edge(u, v)
            
            # Get connected components
            components = list(nx.connected_components(G_clean))
            
            # Build popularity array: each node gets the size of its component
            popularity = np.zeros(self.noscillators, dtype=np.int32)
            for comp in components:
                comp_size = len(comp)
                for node in comp:
                    popularity[node] = comp_size
            
            return popularity
        
        # Vectorize over batch dimension
        popularity_all_batches = np.array([
            compute_popularity_single_batch(b) for b in range(self.nbatch)
        ])
        
        return jnp.array(popularity_all_batches)

    def get_mismatches_over_time(
            self,
            error_threshold: float = 1e-1,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Returns a list with dimension (nbatch, nsteps) with number of phase mismatches at each timestep.
        """
        nsteps = self.phases.shape[1]
        absolute_mismatches = jnp.zeros((self.nbatch, nsteps), dtype=jnp.int32)
        relative_mismatches = jnp.zeros((self.nbatch, nsteps), dtype=jnp.float32)

        for t in range(nsteps):
            counts, fracs, _, _ = self.count_phase_mismatches(
                error_threshold=error_threshold,
                get_pairs=False,
                time_index=t
            )
            absolute_mismatches = absolute_mismatches.at[:, t].set(counts)
            relative_mismatches = relative_mismatches.at[:, t].set(fracs)

        return absolute_mismatches, relative_mismatches
    

    def get_time_to_convergence(
            self,
            error_threshold: float = 1e-1,
            fraction_converged: float = 1.0
    ):
        """
        Returns time to convergence (in timesteps) for each batch element.
        If not converged, returns nsteps.
        fraction_converged is the percentage of oscillators that needs to be converged.
        """
        _, relative_mismatches = self.get_mismatches_over_time(
            error_threshold=error_threshold
        )
        nsteps = self.phases.shape[1]

        # Find first timestep where mismatches == 0
        # argmax returns first True, or 0 if all False
        converged_mask = relative_mismatches <= (1.0 - fraction_converged)
        first_zero_idx = jnp.argmax(converged_mask, axis=1)
        
        # If never converged (all False), argmax returns 0, but we want nsteps
        never_converged = ~jnp.any(converged_mask, axis=1)
        time_to_convergence = jnp.where(never_converged, nsteps, first_zero_idx)
        
        return time_to_convergence



    def visualize_subnetworks_graph_at_timestep(
            self,
            time_index: int = -1,
            nbatch_index: int = 0,
            error_threshold: float = 1e-1,
            path: Optional[str] = None,
            show: bool = False,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = "tab20",
            title: str = None,
            disable_node_number=False,
            fontsize: int = 12,
            ignore_coloring: bool = False,
        ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        """
        Visualize convergence at a specific timestep by highlighting mismatched phase biases.
        
        Args:
            time_index (int): Timestep to analyze (-1 for final). Defaults to -1.
            error_threshold (float): Threshold for phase mismatch detection. Defaults to 1e-1.
            nbatch_index (int): Which batch element to visualize. Defaults to 0.
            path (Optional[str]): Path to save figure. Defaults to None.
            show (bool): Whether to display the plot. Defaults to False.
            figsize (Tuple[int, int]): Figure size. Defaults to (10, 10).
            cmap (str): Colormap name. Defaults to "tab20".
            ignore_coloring (bool): If True, render all nodes in default blue and all edges in solid gray, ignoring components/mismatches. Defaults to False.
            
        Returns:
            Tuple[List[List[int]], List[Tuple[int, int]]]: Connected components and mismatched edges.
        """
        # Get mismatched pairs at this timestep
        _, _, mismatched_pairs, _ = self.count_phase_mismatches(
            error_threshold=error_threshold,
            get_pairs=True,
            time_index=time_index
        )
        
        # Extract mismatched pairs for this batch
        if mismatched_pairs is not None and nbatch_index < len(mismatched_pairs):
            mismatched_pairs_batch = mismatched_pairs[nbatch_index]
            # Convert (2, k) array to list of (i, j) tuples
            if mismatched_pairs_batch.shape[1] > 0:
                phase_mismatches = [
                    (int(mismatched_pairs_batch[0, i]), int(mismatched_pairs_batch[1, i]))
                    for i in range(mismatched_pairs_batch.shape[1])
                ]
            else:
                phase_mismatches = []
        else:
            phase_mismatches = []
        
        # Get adjacency matrix from modulation params
        adjacency_matrix = jnp.where(
            self.modulation_params["rhos"][nbatch_index] != 0,
            1.0,
            0.0
        )
        
        # Visualize
        _visualize_subnetworks_graph(
            node_positions=self.graph_node_positions,
            adjacency_matrix=adjacency_matrix,
            phase_mismatches=phase_mismatches,
            path=path,
            show=show,
            figsize=figsize,
            cmap=cmap,
            title=title if title is not None else f"Convergence at timestep {time_index} (batch {nbatch_index})",
            disable_node_number=disable_node_number,
            fontsize=fontsize,
            ignore_coloring=ignore_coloring,
        )


    def visualize_subnetworks_evolution_video(
            self,
            path: str,
            error_threshold: float = 1e-1,
            nbatch_index: int = 0,
            timestep_interval: int = 10,
            fps: int = None,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = "tab20",
        ) -> None:
        if fps is None:
            fps = 1 / (CPG_DEFAULT_DT * timestep_interval)
            print(f"fps: {fps}")

        # absolute tmp dir
        tmp_frames_dir = os.path.abspath("tmp_frames_dir")
        os.makedirs(tmp_frames_dir, exist_ok=True)

        nsteps_total = self.phases.shape[1]
        timesteps = range(0, nsteps_total, timestep_interval)
        print(f"Generating {len(timesteps)} frames for convergence video...")

        frame_paths = []
        for frame_idx, time_idx in enumerate(timesteps):
            frame_path = os.path.join(tmp_frames_dir, f"frame_{frame_idx:04d}.png")
            try:
                self.visualize_subnetworks_graph_at_timestep(
                    time_index=time_idx,
                    error_threshold=error_threshold,
                    nbatch_index=nbatch_index,
                    path=frame_path,
                    show=False,
                    figsize=figsize,
                    cmap=cmap,
                )
                frame_paths.append(frame_path)
                print(f"  Frame {frame_idx+1}/{len(timesteps)} saved: t={time_idx}")
            except Exception as e:
                print(f"  Error generating frame at t={time_idx}: {e}")

        if not frame_paths:
            print("✗ No frames were generated; aborting video creation.")
            return

        # verify frames exist
        missing = [p for p in frame_paths if not os.path.isfile(p)]
        if missing:
            print(f"✗ Missing frame files, aborting: {missing[:3]} ...")
            return

        frames_pattern = os.path.join(tmp_frames_dir, "frame_%04d.png")
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",      # suppress banner/progress
            "-framerate", str(fps),
            "-i", frames_pattern,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            os.path.abspath(path),
        ]

        print(f"\nStitching frames into video: {path}")
        try:
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,  # or keep stderr=None if you want errors printed
            )
            print(f"✓ Video created successfully at: {path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error creating video: {e}")
            print("  Ensure ffmpeg is installed (e.g., sudo apt-get install ffmpeg)")
            return
        except FileNotFoundError:
            print("✗ ffmpeg not found. Install it (e.g., sudo apt-get install ffmpeg).")
            return

        print(f"\nCleaning up frame directory: {tmp_frames_dir}")
        shutil.rmtree(tmp_frames_dir, ignore_errors=True)
        print("✓ Frames deleted")


    def clear(self):
        """
        Clear all stored states and controls.
        """
        if hasattr(self, '_control'):
            del self._control
        if hasattr(self, '_phases'):
            del self._phases
        if hasattr(self, 'cpg_state'):
            del self.cpg_state





class BS_CPG(CPG_Ring_Arms):
    """
    Initializes a CPG_Ring_Arms with specific constraints to use on brittle star simulations.
    Constraints:
    - Always 5 arms
    - Every arm segment results in 2 ring segments
    - Ganglion controls are omitted as they are not linked to a physical actuator.

    Note: arm_setup != ring_setup. Arm_setup is only used in brittle star context.
    """
    def __init__(
            self,
            arm_setup: list | jnp.ndarray,
            control_timestep: float = CONTROL_TIMESTEP,
            **kwargs
        ):
            """
            **kwargs are passed to CPG_Ring_Arms constructor.
            """
            arm_setup = jnp.array(arm_setup)
            ring_setup = jnp.array([2 * n_segments for n_segments in arm_setup])  # each arm segment corresponds to 2 ring segments
            self.arm_setup = arm_setup
            super().__init__(
                ring_setup=ring_setup,
                **kwargs
            )
            self.control_timestep = control_timestep
            self.cpg_timestep = self._dt
            
    @property
    def control_for_simulator(self, control_timestep: float = CONTROL_TIMESTEP) -> jnp.ndarray:
        "shape (nbatch, nsteps, noscillators). Ganglion controls are omitted."
        control = self.control  # shape (nbatch, nsteps, noscillators)
        # remove ganglion controls (first len(arm_setup) oscillators)
        control_no_ganglia = control[:, :, len(self.arm_setup):]  # shape (nbatch, nsteps, noscillators - len(arm_setup))
        subsampled_control = self._subsample_control(control_no_ganglia, control_timestep=control_timestep, cpg_timestep=self._dt)
        return subsampled_control
    
    @staticmethod
    def _subsample_control(control, control_timestep: float = CONTROL_TIMESTEP, cpg_timestep: float = CPG_DEFAULT_DT) -> jnp.ndarray:
        """
        Subsample control from CPG timestep to the actual control timestep of the simulator.
        control: shape (nbatch, nsteps_cpg, noscillators)
        Returns:
            control_subsampled: shape (nbatch, nsteps_control, noscillators)
        """
        # resample control
        downsample_rate = control_timestep / cpg_timestep # e.g. if CPG_DT is 100 times smaller than CONTROL_TIMESTEP, downsample by factor 100
        control = control[:, ::int(downsample_rate), :]
        return control




# ===========================
# Static functions
# ===========================

def _independent_params_clipping_and_rescaling(independent_params: dict) -> dict:
    independent_params["R"] = clip_and_rescale(independent_params["R"], PARAM_CLIP_MIN, PARAM_CLIP_MAX, R_SCALE_MIN, R_SCALE_MAX)
    independent_params["X"] = clip_and_rescale(independent_params["X"], PARAM_CLIP_MIN, PARAM_CLIP_MAX, X_SCALE_MIN, X_SCALE_MAX)
    if "omegas" in independent_params:
        independent_params["omegas"] = clip_and_rescale(independent_params["omegas"], PARAM_CLIP_MIN, PARAM_CLIP_MAX, OMEGA_SCALE_MIN, OMEGA_SCALE_MAX)
    return independent_params


def _generate_tree_without_loops(independent_params: dict, ring_setup: chex.Array, omega: float | None) -> dict:
    """
    No batch dimension.
    Generate modulation params for tree-like CPG structure without loops.
    Shape of independent_params:
    - R: jnp.ndarray (1D) of shape (num_oscillators,) - Amplitudes for the oscillators
    - X: jnp.ndarray (1D) of shape (num_oscillators,) - Offsets for the oscillators
    - [optional] omegas: jnp.ndarray (1D) of shape (1,) - Frequency (in radians per second) for all oscillators
    - rhos: jnp.ndarray (1D) of shape (num_oscillators - 1,) - Phase biases for the oscillators
    """
    modulation_params = {}
    modulation_params["R"] = independent_params["R"]
    modulation_params["X"] = independent_params["X"]

    if omega is None:
        omega = independent_params["omegas"][0]

    modulation_params = _add_omegas_to_modulation_params(modulation_params, omega)

    modulation_params["rhos"] = _generate_rhos_tree_without_loops(independent_params, ring_setup)

    return modulation_params

def _generate_rhos_tree_without_loops(independent_params: jnp.ndarray, ring_setup: chex.Array) -> jnp.ndarray:
    # Fix rhos matrix tree structure
    noscillators = len(ring_setup) + sum(ring_setup)
    ring_indices = jnp.arange(len(ring_setup))
    arm_indices = jnp.arange(len(ring_setup), noscillators)
    

    adjacency_matrix_tree = jnp.zeros((noscillators, noscillators))
    # connect ring oscillators counterclockwise
    adjacency_matrix_tree = adjacency_matrix_tree.at[ring_indices[:-1], ring_indices[1:]].set(1.)
    
    # connect arms to ring
    if sum(ring_setup) > 0:
        # connect ring to arms
        first_osc_per_arm_indices = (jnp.cumsum(jnp.array([0] + list(ring_setup[:-1])))) + len(ring_setup)
        nonzero_arm_indices = jnp.nonzero(ring_setup)[0]
        first_osc_per_arm_indices = jnp.unique(first_osc_per_arm_indices)
        adjacency_matrix_tree = adjacency_matrix_tree.at[ring_indices[nonzero_arm_indices], first_osc_per_arm_indices[:len(nonzero_arm_indices)]].set(1.)

        # connect arms internally. Note: last index per arm does not connect to anything further
        arm_indices_to_delete = jnp.cumsum(ring_setup) + (len(ring_setup) - 1) # last index per arm
        arm_indices_without_last_arm_per_arm = jnp.delete(arm_indices, jnp.where(jnp.isin(arm_indices, arm_indices_to_delete))[0])
        adjacency_matrix_tree = adjacency_matrix_tree.at[arm_indices_without_last_arm_per_arm, arm_indices_without_last_arm_per_arm + 1].set(1.)

    # at the indices of the non-zero connections, assign the rhos from independent_params
    nonzero_indices = jnp.nonzero(jnp.triu(adjacency_matrix_tree))
    
    rhos_nonsymmetric = jnp.zeros((noscillators, noscillators))
    rhos_nonsymmetric = rhos_nonsymmetric.at[nonzero_indices].set(independent_params["rhos"])

    rhos = rhos_nonsymmetric - rhos_nonsymmetric.T  # make antisymmetric

    return rhos


def _add_omegas_to_modulation_params(modulation_params: dict, omega: float) -> dict:
    """
    add preset or optimized omega to every oscillator in modulation params
    Does not include nbatch. Vmapping can be done in outer loop
    """
    noscillators = modulation_params["R"].shape[0]
    omegas = jnp.ones(noscillators) * omega
    
    modulation_params["omegas"] = omegas
    return modulation_params

def _close_ring_connection_in_modulation_params(modulation_params: dict, ring_setup: chex.Array) -> dict:
    """
    Close the ring connection by adjusting the phase bias between the last and first ganglion oscillators.    
    No batch dim for modulation_params.
    """
    rhos = modulation_params["rhos"]
    first_ring_idx = 0
    last_ring_idx = len(ring_setup) - 1

    rho_ring_close = _calculate_rho_between_oscillators(
        ring_setup,
        rhos,
        first_ring_idx,
        last_ring_idx
    )

    modulation_params["rhos"] = modulation_params["rhos"].at[first_ring_idx, last_ring_idx].set(rho_ring_close)
    modulation_params["rhos"] = modulation_params["rhos"].at[last_ring_idx, first_ring_idx].set(-rho_ring_close)
    return modulation_params


def _add_cobweb_connections_to_modulation_params(
        modulation_params: dict,
        ring_setup: chex.Array,
    ) -> dict:
    """
    Add cobweb connections between neighbouring arms by adjusting the phase biases accordingly.
    No batch dim for modulation_params.
    1. Identify neighbouring arms based on ring_setup.
    2. For each pair of neighbouring arms, add connections between corresponding oscillators.
    3. Phase biases to add are calculated based on the already established phase biases to maintain
       consistency in the CPG dynamics.
    """
    # determine number of ip oscillators to connect per arm.
    # An arm can only connect to neighbouring arms at the same segment, if that segment is also present in that arm.
    narms = len(ring_setup)
    rhos = modulation_params["rhos"]

    ccw_nosc = jnp.min(jnp.array([ring_setup, jnp.roll(ring_setup, -1)]), axis=0)  # number of oscillators to connect counterclockwise

    # add connections: calculate the difference between them, modulate that difference to that value -> no additional degrees of freedom
    for arm in range(narms):
        for osc in range(0, ccw_nosc[arm]):
            osc_1_idx = len(ring_setup) + sum(ring_setup[:arm]) + osc
            osc_2_idx = len(ring_setup) + sum(ring_setup[:(arm+1)%narms]) + osc
            rho_1_2 = _calculate_rho_between_oscillators(ring_setup, rhos, osc_1_idx, osc_2_idx)
            rhos = rhos.at[osc_1_idx, osc_2_idx].set(rho_1_2)
            rhos = rhos.at[osc_2_idx, osc_1_idx].set(-rho_1_2)


    modulation_params["rhos"] = rhos
    
    return modulation_params  # placeholder for future implementation


def _add_full_connections_to_modulation_params(
        modulation_params: dict,
        ring_setup: chex.Array,
    ) -> dict:
    """
    Add connections between all oscillators by calculating the phase biases according to the tree structure.
    No batch dim for modulation_params.
    ring_setup required to calculate phase biases correctly.
    """
    rhos = modulation_params["rhos"]
    noscillators = rhos.shape[0]
    
    pre_connection_idx, post_connection_idx = _get_fully_connected_indices(noscillators)

    calculated_rhos = _calculate_rho_between_oscillators_for_multiple_oscillators(
        ring_setup,
        rhos,
        pre_connection_idx,
        post_connection_idx
    )
    
    rhos = rhos.at[pre_connection_idx, post_connection_idx].set(calculated_rhos)
    rhos = rhos.at[post_connection_idx, pre_connection_idx].set(-calculated_rhos)

    modulation_params["rhos"] = rhos
    return modulation_params


def _add_fully_connected_ring_to_modulation_params(
        modulation_params: dict,
        ring_setup: chex.Array,
    ) -> dict:
    """
    Add fully connected ring connections by calculating the phase biases according to the tree structure.
    No batch dim for modulation_params.
    ring_setup required to calculate phase biases correctly.
    """
    rhos = modulation_params["rhos"]
    nring_osc = len(ring_setup)
    pre_connection_idx, post_connection_idx = _get_fully_connected_indices(nring_osc)

    calculated_rhos = _calculate_rho_between_oscillators_for_multiple_oscillators(
        ring_setup,
        rhos,
        pre_connection_idx,
        post_connection_idx
    )
    
    rhos = rhos.at[pre_connection_idx, post_connection_idx].set(calculated_rhos)
    rhos = rhos.at[post_connection_idx, pre_connection_idx].set(-calculated_rhos)

    modulation_params["rhos"] = rhos
    return modulation_params


def _add_ratio_couplings_connections_to_modulation_params(
        modulation_params: dict,
        ring_setup: chex.Array,
        ratio_couplings_oscillators: int,
        rng: jax.random.PRNGKey,
    ) -> dict:
        """
        Note: ratio_couplings_oscillators will always be between 1 (base method)
        and (n-1)/2 (fully connected ring). (n=total number of oscillators)
        Also note that we are adding connections on the base connectivity,
        which means that those connections don't need to be added anymore
        Couplings to add = (ratio - 1) * n = (n * ratio) - n
        """
        rhos = modulation_params["rhos"]
        n = rhos.shape[0]
        ncoupling_to_add = (n * ratio_couplings_oscillators) - n  # number of connections to add


        noscillators = len(ring_setup) + sum(ring_setup)
        ring_indices = jnp.arange(len(ring_setup))
        arm_indices = jnp.arange(len(ring_setup), noscillators)
        
        # List all candidate connections to add (fully connected excluding the ones already in base connection)
        all_connections_pre_idx, all_connections_post_idx = _get_fully_connected_indices(n)

        # add ring connections to existing pre/post indices
        existing_connections_pre_idx = ring_indices
        existing_connections_post_idx = jnp.roll(ring_indices, -1)

        # add arm connections to existing pre/post indices
        if sum(ring_setup) > 0:
            # connect ring to arms
            first_osc_per_arm_indices = (jnp.cumsum(jnp.array([0] + list(ring_setup[:-1])))) + len(ring_setup)
            nonzero_arm_indices = jnp.nonzero(ring_setup)[0]
            first_osc_per_arm_indices = jnp.unique(first_osc_per_arm_indices)
            existing_connections_pre_idx = jnp.concatenate([existing_connections_pre_idx, ring_indices[nonzero_arm_indices]])
            existing_connections_post_idx = jnp.concatenate([existing_connections_post_idx, first_osc_per_arm_indices[:len(nonzero_arm_indices)]])

            # connect arms internally. Note: last index per arm does not connect to anything further
            arm_indices_to_delete = jnp.cumsum(ring_setup) + (len(ring_setup) - 1) # last index per arm
            arm_indices_without_last_arm_per_arm = jnp.delete(arm_indices, jnp.where(jnp.isin(arm_indices, arm_indices_to_delete))[0])
            existing_connections_pre_idx = jnp.concatenate([existing_connections_pre_idx, arm_indices_without_last_arm_per_arm])
            existing_connections_post_idx = jnp.concatenate([existing_connections_post_idx, arm_indices_without_last_arm_per_arm + 1])


        # remove existing connections from the full list using JAX only
        all_pairs = jnp.stack([all_connections_pre_idx, all_connections_post_idx], axis=1)  # (m, 2)
        existing_pairs = jnp.stack([existing_connections_pre_idx, existing_connections_post_idx], axis=1)  # (k, 2)

        def _compute_remaining(all_pairs, existing_pairs):
            if existing_pairs.shape[0] == 0:
                return jnp.ones(all_pairs.shape[0], dtype=bool)
            eq = jnp.all(all_pairs[:, None, :] == existing_pairs[None, :, :], axis=-1)  # (m, k)
            return ~jnp.any(eq, axis=1)  # (m,)

        keep_mask = _compute_remaining(all_pairs, existing_pairs)

        remaining_pre_idx = all_pairs[:, 0][keep_mask]
        remaining_post_idx = all_pairs[:, 1][keep_mask]

        index_candidates = jnp.arange(remaining_pre_idx.shape[0])
        selected_indices = jax.random.choice(
            rng,
            index_candidates,
            (ncoupling_to_add,),
            replace=False
        )

        selected_pre_idx = remaining_pre_idx[selected_indices]
        selected_post_idx = remaining_post_idx[selected_indices]

        calculated_rhos = _calculate_rho_between_oscillators_for_multiple_oscillators(
            ring_setup,
            rhos,
            selected_pre_idx,
            selected_post_idx
        )
        
        rhos = rhos.at[selected_pre_idx, selected_post_idx].set(calculated_rhos)
        rhos = rhos.at[selected_post_idx, selected_pre_idx].set(-calculated_rhos)
        modulation_params["rhos"] = rhos
        return modulation_params



def _remove_radially_outward_connections_in_arms_from_modulation_params(
        modulation_params: dict,
        ring_setup: chex.Array,
    ) -> dict:
    """
    Remove radially outward connections in arms by setting their phase biases to zero.
    No batch dim for modulation_params.
    ring_setup required to identify the radially outward connections.

    1. Remove upper triangular part of rhos matrix (outward connections in arms, clockwise direction in ring)
    2. Restore the ring connections
    """
    rhos = modulation_params["rhos"]
    nring_osc = len(ring_setup)
    ring_connections_to_restore = rhos[:nring_osc, :nring_osc].copy()
    
    rhos = jnp.tril(rhos)  # remove upper triangular part (outward connections in arms, clockwise direction in ring)
    rhos = rhos.at[:nring_osc, :nring_osc].set(ring_connections_to_restore)  # restore ring connections

    modulation_params["rhos"] = rhos
    return modulation_params


def _calculate_rho_between_oscillators(ring_setup: list, rhos: chex.Array, osc_1_idx: int, osc_2_idx: int) -> float:
    """
    No batch dim for rhos: rhos dim = (num_oscillators, num_oscillators)
    Rhos are calculated with respect to node 0.

    Compute rho_1_2 = rho_0_2 - rho_0_1 where rho_0_k is the phased bias
    accumulated from ganglion 0 to node k following the connectivity:
        - along the ganglion ring (0 -> 1 -> 2 -> ...)
        - then from the ganglion of the target arm down the arm segments:
            ganglion -> first_ip -> oop -> next_ip -> ...
    This implementation is JAX-friendly (uses jnp only, no Python loops depending on tracers).
    """
    rho_0_1 = _rho_0_to_idx(ring_setup, rhos, osc_1_idx)
    rho_0_2 = _rho_0_to_idx(ring_setup, rhos, osc_2_idx)
    return (rho_0_2 - rho_0_1).astype(jnp.float32)


_calculate_rho_between_oscillators_for_multiple_oscillators = jax.vmap(
        _calculate_rho_between_oscillators,
        in_axes=(None, None, 0, 0)
    )


def _rho_0_to_idx(ring_setup: list, rhos: chex.Array, idx: int) -> jnp.ndarray:
    """
    JAX-friendly accumulator of phase biases from node 0 to node `idx`.
    Returns a jnp scalar (float32).
    """
    mat = jnp.asarray(rhos)
    ring_setup_arr = jnp.asarray(ring_setup)
    ganglion_count = ring_setup_arr.shape[0]
    idx_j = jnp.asarray(idx)

    # precompute diagonal of ganglion ring (k -> k+1) and useful arrays
    diag1 = jnp.diag(mat, k=1)  # length = N-1 -> Elements on the first superdiagonal
    positions_diag = jnp.arange(diag1.shape[0])

    # helper: sum first `n` diagonal elements without slicing by tracer
    def sum_first_n_diag(n):
        mask = positions_diag < n
        return jnp.sum(diag1 * mask)

    # ganglion case
    def ring_case():
        return jnp.where(idx_j == 0, 0.0, sum_first_n_diag(idx_j))

    def arm_segment_case():
        adjusted = idx_j - ganglion_count               # 0-based index over all arm oscillators
        segments_passed = adjusted                       # one oscillator per segment

        segments_cumsum = jnp.cumsum(ring_setup_arr)
        segments_before = segments_cumsum - ring_setup_arr

        # find arm index
        arm_mask = (segments_passed >= segments_before) & (segments_passed < segments_cumsum)
        arm_idx = jnp.argmax(arm_mask)

        segment_in_arm = segments_passed - segments_before[arm_idx]
        first_seg_index = ganglion_count + segments_before[arm_idx]

        # accumulate ganglion 0 -> arm ganglion
        s = sum_first_n_diag(arm_idx)

        # add ganglion -> first segment
        s = s + mat[arm_idx, first_seg_index]

        # walk along the arm (one osc per segment)
        max_segs = jnp.max(ring_setup_arr)
        seg_pos = jnp.arange(max_segs)
        seg_mask = seg_pos < segment_in_arm              # segments strictly before target
        seg_idxs = first_seg_index + seg_pos

        s = s + jnp.sum(mat[seg_idxs, seg_idxs + 1] * seg_mask)

        return s

    result = jnp.where(idx_j < ganglion_count, ring_case(), arm_segment_case())
    return result.astype(jnp.float32)


def _get_fully_connected_indices(nosc: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate pre- and post-synaptic indices for a fully connected network.
    
    Args:
        nosc (int): Number of oscillators
        
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: pre_synaptic_idx, post_synaptic_idx
        
    Example:
        For nosc=4:
        pre_synaptic_idx = [0, 0, 0, 1, 1, 2]
        post_synaptic_idx = [1, 2, 3, 2, 3, 3]
    """
    # Create all pairs (i, j) where i < j using meshgrid
    pre_idx, post_idx = jnp.meshgrid(jnp.arange(nosc), jnp.arange(nosc), indexing='ij')
    
    # Flatten and filter to get only i < j
    pre_flat = pre_idx.flatten()
    post_flat = post_idx.flatten()
    
    mask = pre_flat < post_flat
    
    pre_synaptic_idx = pre_flat[mask]
    post_synaptic_idx = post_flat[mask]
    
    return pre_synaptic_idx, post_synaptic_idx


def visualize_connectivity(
        node_positions,
        ring_setup: chex.Array,
        adjacency_matrix: chex.Array,
        path: Optional[str] = None,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (15, 15),
        clean_layout: bool = False
    ):
    matrix = adjacency_matrix
    ring_setup = np.array(ring_setup)

    if path != None:
        assert (path[-4:] == ".jpg" or path[-4:] == ".png"), "Make sure the extension of the path is either .jpg or .png"

    colours = {"ring": "red", "arm": "green"}

    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    
    if not clean_layout:
        node_colors = len(ring_setup) * [colours["ring"]] + sum(ring_setup) * [colours["arm"]] # ring nodes are red, arm nodes are green
    else:
        node_colors = "#1f78b4"
    # Draw graph
    plt.figure(figsize=figsize)
    nx.draw(
        G,
        node_positions,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        node_size=300,
        font_size=12,
        font_color="white",
        arrows=True,
        )
    
    plt.title("Brittle Star CPG Connectivity")
    
    # Add legend based on node colors
    legend_elements = [
        Patch(facecolor=colours["ring"], edgecolor='black', label='ring'),
        Patch(facecolor=colours["arm"], edgecolor='black', label='arm'),
    ]

    if not clean_layout:
        plt.legend(handles=legend_elements, loc='lower left')
    
    if path != None:
        plt.savefig(path)

    if show_plot == True:
        plt.show()
    else:
        plt.close()


def _visualize_subnetworks_graph(
        node_positions,
        adjacency_matrix: chex.Array,
        phase_mismatches: Optional[List[Tuple[int, int]]] = None,
        path: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (10, 10),
        cmap: str = "tab20",
        title: str = "Connectivity components after removing mismatched couplings",
        disable_node_number: bool = False,
        fontsize: int = 12,
    ignore_coloring: bool = False,
    ):
    """
    Visualize graph and highlight subnetworks after removing mismatched couplings.

    Args:
        node_positions: Positions of nodes for plotting (to ensure nice ring layout)
        adjacency_matrix: Matrix to visualize
        phase_mismatches: List of (i, j) oscillator index tuples that are mismatched
        path: Path to save figure
        show: Whether to display plot
        figsize: Figure size
        cmap: Colormap name
        title: Plot title
        ignore_coloring: If True, draw all nodes in default blue and all edges in solid gray (ignore component coloring and mismatched edges)
        
    Returns:
        (components, mismatched_edges_normalized)
    """
    matrix = np.asarray(adjacency_matrix)
    # Build undirected graph from non-zero entries
    edges_u, edges_v = np.nonzero(np.abs(matrix) > 0.0)
    G = nx.Graph()
    n = matrix.shape[0]
    G.add_nodes_from(range(n))
    # Add edges once (undirected)
    for u, v in zip(edges_u, edges_v):
        if u != v:
            G.add_edge(int(u), int(v))

    # Normalize mismatches to set of undirected edges
    mismatched_edges_set = set()
    mismatched_edges_normalized = []
    if phase_mismatches:
        for (a, b) in phase_mismatches:
            ia = int(a)
            ib = int(b)
            e = (min(ia, ib), max(ia, ib))
            mismatched_edges_set.add(e)
        mismatched_edges_normalized = sorted(list(mismatched_edges_set))

    # Create graph with mismatched edges removed
    G_clean = G.copy()
    for (u, v) in mismatched_edges_set:
        if G_clean.has_edge(u, v):
            G_clean.remove_edge(u, v)

    # Get connected components
    components = [sorted(list(c)) for c in nx.connected_components(G_clean)]
    components = sorted(components, key=lambda x: (len(x) == 0, -len(x), x))

    # Build a mapping from node to component index
    node_to_component = {}
    for idx, comp in enumerate(components):
        for node in comp:
            node_to_component[node] = idx

    # Build color map or default colors if requested
    if ignore_coloring:
        node_colors = ["#1f78b4"] * n  # matplotlib default blue
        normal_edges = list(G.edges())
        mismatched_edges = []
    else:
        palette = list(plt.get_cmap(cmap).colors) if hasattr(plt.get_cmap(cmap), "colors") else None
        if palette is None:
            palette = list(plt.cm.tab20.colors)
        
        node_color_map = {}
        if len(components) == 1 and len(mismatched_edges_normalized) == 0:
            single_color = palette[0]
            for node in G.nodes:
                node_color_map[node] = single_color
        else:
            for idx, comp in enumerate(components):
                color = palette[idx % len(palette)]
                for node in comp:
                    node_color_map[node] = color
            for node in G.nodes:
                if node not in node_color_map:
                    node_color_map[node] = (0.7, 0.7, 0.7, 1.0)

        node_colors = [node_color_map[i] for i in range(n)]

        # Classify edges: red dotted if between different components, gray solid if within same component
        normal_edges = []
        mismatched_edges = []
        for e in G.edges():
            u, v = int(e[0]), int(e[1])
            # Check if the edge connects nodes in different components
            u_comp = node_to_component.get(u, -1)
            v_comp = node_to_component.get(v, -1)
            if u_comp != v_comp:
                mismatched_edges.append(e)
            else:
                normal_edges.append(e)

    # Draw
    plt.figure(figsize=figsize)
    ax = plt.gca()

    if normal_edges:
        nx.draw_networkx_edges(G, node_positions, edgelist=normal_edges, edge_color="gray", width=0.8, style="solid", alpha=0.8, ax=ax)
    if mismatched_edges:
        nx.draw_networkx_edges(G, node_positions, edgelist=mismatched_edges, edge_color="red", width=2.0, style="dashed", alpha=0.9, ax=ax)

    nx.draw_networkx_nodes(G, node_positions, node_color=node_colors, node_size=300, edgecolors="k", linewidths=0.5, ax=ax)
    
    if not disable_node_number:
        nx.draw_networkx_labels(G, node_positions, labels={i: str(i) for i in G.nodes()}, font_color="white", font_size=9, ax=ax)

    # Legend
    if not ignore_coloring:
        legend_handles = []
        if len(components) == 1 and len(mismatched_edges_normalized) == 0:
            legend_handles.append(Patch(facecolor=palette[0], edgecolor="black", label="All nodes (fully converged)"))
        else:
            for idx, comp in enumerate(components):
                lbl = f"component {idx} (size={len(comp)})"
                legend_handles.append(Patch(facecolor=palette[idx % len(palette)], edgecolor="black", label=lbl))
            if mismatched_edges_normalized:
                legend_handles.append(Line2D([0], [0], color="red", lw=2, linestyle="--", label="mismatched edge"))
    plt.title(title, fontsize=fontsize)
    plt.axis("off")
    # plt.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, -0.05))
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()




def plot_matrix_as_annotated_heatmap(
    matrix,
    title: str = "Matrix Heatmap",
    xlabel: str = "Post-connection Oscillator Index",
    ylabel: str = "Pre-connection Oscillator Index",
    scale: bool = True,
    scale_label: str = "Value",
    path: Optional[str] = None,
    show_plot: bool = False,
    figsize: Tuple[int, int] = (30, 30),
):
    """
    Matrix should be square.
    Used to visualize adjacency matrix and rhos matrix.
    """
    assert matrix.ndim == 2, "matrix should be 2D"
    assert matrix.shape[0] == matrix.shape[1], "matrix should be square"

    mat = np.asarray(matrix)
    n = mat.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.max(np.abs(mat)) if np.max(np.abs(mat)) > 0 else 1.0
    im = ax.imshow(mat, cmap="bwr", vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="equal")
    if scale:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(scale_label)

    # ticks: simple enumeration
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(np.arange(n))
    ax.set_yticklabels(np.arange(n))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with formatted numbers, pick text color for contrast
    thresh = vmax * 0.5
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if abs(val) > thresh else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if path is not None:
        plt.savefig(path)
    if show_plot:
        plt.show()
    else:
        plt.close()


@functools.partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0)
def _reset_state_vectorized(
        rng: chex.PRNGKey,
        cpg_self: CPG, 
        cpg_reset_phase_ranges: Tuple[float, float]=CPG_RESET_PHASE_RANGES,
        disable_randomness: bool=DISABLE_RANDOMNESS,
        ) -> CPGState:
    """
    Generate a novel initialized CPGState object.
    Since it is vmapped, returns a batch of CPGState objects.
    Args:
        rng: chex.PRNGKey, optional -> not required if disable randomness == True
        phase_init_range: Tuple[float, float], optional -> range for uniform initialization of the phases of the oscillators
        If disable_randomness is True, the rng key is ignored and a fixed seed=0 is used.

    """
    state = cpg_self.reset(
        rng=rng,
        cpg_reset_phase_ranges=cpg_reset_phase_ranges,
        disable_randomness=disable_randomness
    )

    return state


@functools.partial(jax.vmap, in_axes=(0, 0, None), out_axes=0)
def _modulate_state_vectorized(
        state: CPGState,
        modulation_params: dict,
        cpg_self: CPG,
        ) -> CPGState:
    """
    Modulate the CPG state with new parameters. They can change at any moment,
    because they only affect the steady state solution of the CPG.
    Since it is vmapped, returns a batch of CPGState objects.
    Args:
        state (CPGState): The current state of the CPG. (with nbatch dim)
        modulation_params (dict): dict containing the modulation parameters with nbatch dim.
            - R (jnp.ndarray (nbatch, num_oscillators,), optional): Modulation for the amplitude
            - X (jnp.ndarray (nbatch, num_oscillators,), optional): Modulation for the offset.
            - omegas (jnp.ndarray (nbatch, num_oscillators,), optional): New frequencies for the oscillators.
            - rhos (jnp.ndarray (nbatch, num_oscillators, num_oscillators), optional): New phase biases for the oscillators.
    Returns:
        CPGState: The updated CPG state with the new modulation parameters.
    """
    state = cpg_self.modulate(
        state=state,
        R=modulation_params["R"],
        X=modulation_params["X"],
        omegas=modulation_params["omegas"],
        rhos=modulation_params["rhos"]
    )
    return state


def popularity_factor(x, alpha: int = 1):
    """
    function can be applied on scalars or arrays
    Alpha represents the number for which the popularity fraction is 37%
    1/exp(1) = 0.37
    """
    return jnp.exp(- x / alpha)



# Helper to run a single configuration and return time to convergence and relative non-converged fractions
def run_one_config_with_random_modulation(ring_setup: jnp.ndarray,
                rng: chex.PRNGKey,
                method: str,
                nbatch: int,
                nsteps: int,
                weight_coupling: float = CPG_DEFAULT_WEIGHT_SCALE,
                dt: float = CPG_DEFAULT_DT,
                solver: str = "rk4",
                ratio_couplings_oscillators: int | str | None = None,

 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float, int]:
    # Build model
    cpg = CPG_Ring_Arms(
        ring_setup=ring_setup,
        rng=rng,
        method=method,
        ratio_couplings_oscillators=ratio_couplings_oscillators,
        weight_scale=weight_coupling,
        dt=dt,
        solver=solver,
    )
    rng, rng_modulation, rng_reset = jax.random.split(rng, 3)
    # Random modulation params for NBATCH identical ring_setup
    cpg.set_random_modulation_params(rng=rng_modulation, nbatch=nbatch)
    # Reset state/control tracking
    cpg.reset_state(rng_reset)
    cpg.reset_control()
    cpg.reset_phases()
    # Modulate state (weights/rhos applied)
    cpg.modulate_state()
    # Step for a fixed horizon (choose modest steps to keep runtime reasonable)

    cpg.step_state_n_times(nsteps)
    # Metrics per batch
    tconv_p50 = cpg.get_time_to_convergence(error_threshold=1e-1, fraction_converged=0.50)
    tconv_p75 = cpg.get_time_to_convergence(error_threshold=1e-1, fraction_converged=0.75)
    tconv_p90 = cpg.get_time_to_convergence(error_threshold=1e-1, fraction_converged=0.90)
    tconv_p100 = cpg.get_time_to_convergence(error_threshold=1e-1, fraction_converged=1.0)  # (nbatch,) -> timesteps [0, nsteps]
    _, fracs, _, _ = cpg.count_phase_mismatches(error_threshold=1e-1, get_pairs=False, time_index=-1)
    rel_non_converged = fracs  # fraction of connections not converged (nbatch,) -> [0.0, 1.0]
    spectral_gap = cpg.spectral_gap
    induced_norm = cpg.induced_norm
    num_couplings = cpg.num_couplings

    cpg.clear()  # free memory
    del cpg # free memory
    return tconv_p50, tconv_p75, tconv_p90, tconv_p100, rel_non_converged, spectral_gap, induced_norm, num_couplings



if __name__ == "__main__":
    import os
    import sys
    from cpg_convergence.defaults import DISABLE_RANDOMNESS, OMEGA, CPG_RESET_PHASE_RANGES, CPG_DEFAULT_DT, CPG_DEFAULT_SOLVER
    import time
    # Type 1 font
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    time_start = time.time()
    # ==============================
    # configurables
    brittle_star = False
    ring_setup = 5 * [5]  # 5 arms with 5 segments each
    # ring_setup = [5, 4, 0, 6, 7]   # mixed arms
    # ring_setup = [1, 0, 0, 3, 2]   # sparse arm
    # ring_setup = [0] * 20          # big ring
    solver = CPG_DEFAULT_SOLVER
    solver_dt = CPG_DEFAULT_DT
    weight_scale = 5000
    omega = OMEGA
    nbatch = 10
    method = "cobweb"  # "base", "cobweb", "fully_connected", "leader_follower", "popularity", "modified_de", "ratio_couplings_oscillators"
    ratio_couplings_oscillators = None # must be int >= 1, "max", or None

    seed = 43
    mu = 0.0
    sigma = 0.5
    alpha = 3
    
    # ==============================
    # tmp directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, "../../tmp/")
    tmp_dir = os.path.abspath(tmp_dir)  # Normalize the path
    print(f"tmp_dir: {tmp_dir}")

    # ==============================
    rng = jax.random.PRNGKey(seed)
    rng, rng_cpg_init = jax.random.split(rng)

    if brittle_star == True:
        print("Initializing Brittle Star CPG...")
        cpg = BS_CPG(arm_setup=ring_setup,
                        rng=rng_cpg_init,
                        dt=solver_dt,
                        solver=solver,
                        weight_scale=weight_scale,
                        omega=omega,
                        method=method,
                        ratio_couplings_oscillators=ratio_couplings_oscillators
                        )
    else:
        cpg = CPG_Ring_Arms(ring_setup=ring_setup,
                            rng=rng_cpg_init,
                            dt=solver_dt,
                            solver=solver,
                            weight_scale=weight_scale,
                            omega=omega,
                            method=method,
                            ratio_couplings_oscillators=ratio_couplings_oscillators
                            )
    
    print(f"cpg.noscillators: {cpg.noscillators}")
    print(f"cpg.num_oscillators: {cpg.num_oscillators}")
    print(f"cpg.num_couplings: {cpg.num_couplings}")
    print(f"actual ratio couplings to oscillators: {cpg.num_couplings/cpg.num_oscillators:.4f}")

    print(f"cpg.parameter_reshaper.total_params: {cpg.parameter_reshaper.total_params}")

    cpg.visualize_clean_adjacency_matrix(show_plot=False, clean_layout=True, path=os.path.join(tmp_dir, "clean_adjacency_matrix.png"))

    rng, rng_params = jax.random.split(rng)
    cpg.set_random_modulation_params(rng_params, nbatch, mu, sigma)

    print(f"modulation_params_shape: {jax.tree_util.tree_map(lambda x: x.shape, cpg.modulation_params)}")


    cpg.visualize_modulated_adjacency_matrix(nbatch_index=0, show_plot=True)
    cpg.print_modulation_params_ranges()

    print(f"Eigenvalues of Laplacian sorted: {jnp.sort(jnp.real(cpg.eigenvalues_of_laplacian)).round(4)}")
    print(f"number of eigenvalues: {len(cpg.eigenvalues_of_laplacian)}")
    print(f"eigenvalues: {cpg.eigenvalues_of_laplacian.round(4)}")
    print(f"Smallest eigenvalue: {cpg.smallest_eigenvalue}")
    print(f"Spectral gap - Algebraic connectivity (2nd smallest eigenvalue): {cpg.spectral_gap}")
    print(f"max eigenvalue: {cpg.max_eigenvalue}")
    print(f"Induced_norm: {cpg.induced_norm}")
    print(f"Is laplacian symmetric: {cpg.is_laplacian_symmetric}")
    print(f"Are all eigenvalues real: {cpg.are_all_eigenvalues_real}")

    # sys.exit() # Stop here to visualize everything up until the point where rollouts are done

    # Reset, modulate and step CPG states
    rng, rng_reset = jax.random.split(rng)
    cpg.reset_state(rng_reset, cpg_reset_phase_ranges=CPG_RESET_PHASE_RANGES, disable_randomness=DISABLE_RANDOMNESS)
    cpg.reset_control()
    cpg.reset_phases()
    print(f"cpg.state structure: {jax.tree_util.tree_map(lambda x: x.shape, cpg.cpg_state)}")

    cpg.plot_clean_adjacency_matrix_heatmap(show_plot=False, path=os.path.join(tmp_dir, "clean_adjacency_matrix_heatmap.png"))
    cpg.plot_modulated_adjacency_matrix_heatmap(nbatch_index=0, show_plot=False, path=os.path.join(tmp_dir, "modulated_adjacency_matrix_heatmap.png"))
    cpg.plot_modulated_rhos_matrix_heatmap(nbatch_index=0, show_plot=False, path=os.path.join(tmp_dir, "modulated_rhos_matrix_heatmap.png"))


    cpg.modulate_state()
    nsteps_1 = 2
    cpg.step_state_n_times(nsteps=nsteps_1)
    nmismatches, fraction_mismatches, mismatched_pairs, mismatched_values = cpg.count_phase_mismatches(error_threshold=1e-1, get_pairs=False)
    print(f"Number of phase differences mismatches after {nsteps_1} steps: {nmismatches}, fraction of mismatches: {fraction_mismatches}")
    print(f"Subnetwork size per node: {cpg.get_size_of_subnetwork_per_node()}")
    print(f"popularity factor per node: {cpg.get_popularity_factor_per_node(alpha)}")

    cpg.plot_modulated_adjacency_matrix_heatmap(nbatch_index=0, show_plot=False, path=os.path.join(tmp_dir, f"modulated_adjacency_matrix_heatmap_{nsteps_1}_steps.png"))

    cpg.visualize_subnetworks_graph_at_timestep(time_index=-1,
                                                nbatch_index=0,
                                                # path=os.path.join(tmp_dir, "subnetworks_graph.png"),
                                                path=os.path.join(tmp_dir, f"connectivity_no_color"),
                                                show=True,
                                                figsize=(5, 5),
                                                cmap="tab20",
                                                title=" ",
                                                disable_node_number=True,
                                                # fontsize=14,
                                                ignore_coloring=True
                                                )
    # sys.exit()
    
    
    nsteps_2 = 200
    cpg.step_state_n_times(nsteps=nsteps_2)
    nmismatches, fraction_mismatches, mismatched_pairs, mismatched_values = cpg.count_phase_mismatches(error_threshold=1e-1, get_pairs=True)
    print(f"Number of phase differences mismatches after {nsteps_2} steps: {nmismatches}, fraction of mismatches: {fraction_mismatches}")

    print(f"Control shape: {cpg.control.shape}")
    print(f"Phases shape: {cpg.phases.shape}\n")

    cpg.visualize_subnetworks_graph_at_timestep(time_index=-1,
                                                nbatch_index=0,
                                                # path=os.path.join(tmp_dir, "subnetworks_graph.png"),
                                                path=os.path.join(tmp_dir, f"subnetworks_graph_unstable.png"),
                                                show=True,
                                                figsize=(5, 5),
                                                cmap="tab20",
                                                # title=" ",
                                                disable_node_number=True,
                                                # fontsize=14
                                                )
    
    cpg.visualize_subnetworks_evolution_video(path=os.path.join(tmp_dir, "subnetworks_evolution_video.mp4"),
                                                timestep_interval=10,
                                                error_threshold=0.1,
                                                nbatch_index=0,
                                                figsize=(10, 10),
                                                cmap="tab20"
                                                )


    mismatches_over_time_absolute, mismatches_over_time_relative = cpg.get_mismatches_over_time(error_threshold=1e-1)
    print(f"mismatches_over_time_absolute.shape: {mismatches_over_time_absolute.shape}")
    print(f"mismatches_over_time_relative.shape: {mismatches_over_time_relative.shape}")

    tconv_p25 = cpg.get_time_to_convergence(error_threshold=1e-1, fraction_converged=0.25)
    tconv_p50 = cpg.get_time_to_convergence(error_threshold=1e-1, fraction_converged=0.50)
    tconv_p75 = cpg.get_time_to_convergence(error_threshold=1e-1, fraction_converged=0.75)
    tconv_p100 = cpg.get_time_to_convergence(error_threshold=1e-1, fraction_converged=1.00)
    print(f"time_to_convergence at 25% fraction converged: {tconv_p25}")
    print(f"time_to_convergence at 50% fraction converged: {tconv_p50}")
    print(f"time_to_convergence at 75% fraction converged: {tconv_p75}")
    print(f"time_to_convergence at 100% fraction converged: {tconv_p100}")
    print(f"time_to_convergence.shape: {tconv_p100.shape}")



    # Calculate percentiles of absolute mismatches over time
    median_mismatches = jnp.median(mismatches_over_time_absolute, axis=0)
    percentile_5_mismatches = jnp.percentile(mismatches_over_time_absolute, 5, axis=0)
    percentile_95_mismatches = jnp.percentile(mismatches_over_time_absolute, 95, axis=0)
    
    # Plot
    timesteps = jnp.arange(mismatches_over_time_absolute.shape[1])
    
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, median_mismatches, label='Median', linewidth=2, color='blue')
    plt.fill_between(timesteps, percentile_5_mismatches, percentile_95_mismatches, 
                     alpha=0.3, color='blue', label='5th-95th percentile')
    plt.xlabel('Timestep')
    plt.ylabel('Absolute Number of Mismatches')
    plt.title('Phase Mismatch Convergence Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(tmp_dir, "mismatches_over_time.png"))
    plt.show()



    # Plot histogram of time to convergence
    nsteps = mismatches_over_time_absolute.shape[1]
    plt.figure(figsize=(10, 6))
    plt.hist(tconv_p100, bins=np.arange(0, nsteps + 1) - 0.5, edgecolor='black', alpha=0.7, color='steelblue', align='mid')
    plt.xlabel('Time to Convergence (timesteps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time to Convergence')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(tmp_dir, "time_to_convergence_histogram.png"))
    plt.show()



    print(f"\nTotal time taken: {time.time() - time_start:.2f} seconds")


    if brittle_star:
        print(f"CPG based control shape: {cpg.control.shape}")
        print(f"Brittle Star Controls shape: {cpg.control_for_simulator.shape}")
        # plt.plot(cpg.control_for_simulator[0, :, 0], label="Arm 0 segment 0")
        # plt.plot(cpg.control_for_simulator[0, :, 1], label="Arm 0 segment 1")
        # plt.plot(cpg.control_for_simulator[0, :, 2], label="Arm 0 segment 2")
        # plt.show()