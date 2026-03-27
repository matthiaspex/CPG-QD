import os
import sys

import chex
from typing import Optional, Union, Dict, Any, Tuple, List
import numpy as np
import jax
from pprint import pprint
import functools

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import default_qd_metrics


from cpg_convergence.optimizer.base import BaseOptimizer
from cpg_convergence.control_generator import CPGControlGeneratorBS
from cpg_convergence.simulator import BrittleStarSimulator
from cpg_convergence.behavioral_descriptors import BehavioralDescriptorsExtractor
from cpg_convergence.wandb_utils import WandbLogger
from cpg_convergence.visualization import plot_2d_map_elites_repertoire
from cpg_convergence.defaults import PARAM_CLIP_MIN, PARAM_CLIP_MAX, ISO_SIGMA, LINE_SIGMA, \
    GAIT_FREQUENCY


class MAPElitesOptimizer(BaseOptimizer):
    def __init__(
            self,
            control_generator: CPGControlGeneratorBS,
            simulator: BrittleStarSimulator,
            evolution_cfg: dict,
            map_elites_cfg: dict,
            evaluation_cfg: dict,
            simulation_cfg: dict,
            rng: chex.PRNGKey,
            wandb_logger: Optional[WandbLogger] = None,
            ):
        """
        Instantiates attributes
            strategy (MAPElites): Functions as the backbone of the ask-tell loops.
            wandb_logger: allows logging intermediate metrics

        args:
            cpg (BS_CPG): The CPG instance to use for the optimization.
            simulator (BrittleStarSimulator): Required to make initial fill of the archive.
            evolution_cfg (dict): Configuration dictionary for the evolution process. Contains:
                - "popsize": Population size for the evolution strategy.
                - "ngen": Number of generations for the evolution strategy.
            map_elites_cfg (dict): Configuration dictionary for the MAP-Elites algorithm. Contains:
                - "axis_0": Configuration for the first axis of the MAP-Elites grid.
                    - "descriptor": The descriptor for the axis (e.g., "bilateral_contralateral_score", "leading_or_trailing", "ground_contact_fraction").
                    - "min_val": The minimum value for the axis.
                    - "max_val": The maximum value for the axis.
                    - "nbins": The number of bins for the axis.
                - "axis_1": 
                    ...
                - ...
            evaluation_cfg (dict): Configuration dictionary for the evaluation process. Contains:
                - "reward_expr": Expression for the reward calculation.
                - "cost_expr": Expression for the cost calculation.
                - "penalty_expr": Expression for the penalty calculation.
                - "additive_or_multiplicative": Whether the reward, cost, and penalty are combined additively or multiplicatively.
                - "alpha": parameter of the cost in fitness aggregation
                - "beta": parameter of the penalty in fitness aggregation
            simulation_cfg (dict): Configuration dictionary for the simulation process. Contains:
                - "nsteps": Number of steps for the simulation. (required for evaluation)
            rng (chex.PRNGKey): The random number generator key.
            wandb_logger (Optional[WandbLogger]): Logger for Weights & Biases.
        """
        self.control_generator = control_generator
        self.simulator = simulator
        self.evolution_cfg = evolution_cfg
        self.map_elites_cfg = map_elites_cfg
        self.evaluation_cfg = evaluation_cfg
        self.simulation_cfg = simulation_cfg
        self.wandb_logger = wandb_logger

        self.iso_sigma = self.map_elites_cfg.get("iso_sigma", ISO_SIGMA)
        self.line_sigma = self.map_elites_cfg.get("line_sigma", LINE_SIGMA)

        rng, rng_init = jax.random.split(rng, 2)
        self.init_strategy(rng_init)


    @property
    def grid_shape(self):
        if not hasattr(self, "_grid_shape"):
            self._process_map_elites_cfg()
        return self._grid_shape
    
    @property
    def min_descriptor(self):
        if not hasattr(self, "_min_descriptor"):
            self._process_map_elites_cfg()
        return self._min_descriptor

    @property
    def max_descriptor(self):
        if not hasattr(self, "_max_descriptor"):
            self._process_map_elites_cfg()
        return self._max_descriptor

    @property
    def descriptor_types(self):
        if not hasattr(self, "_descriptor_types"):
            self._process_map_elites_cfg()
        return self._descriptor_types
    
    @property
    def ndescriptors(self):
        if not hasattr(self, "_ndescriptors"):
            self._process_map_elites_cfg()
        return self._ndescriptors

    def _process_map_elites_cfg(self):
        grid_shapes = []
        min_descriptor = []
        max_descriptor = []
        descriptor_types = []
        for axis in self.map_elites_cfg.keys():
            if "axis" in axis:
                grid_shapes.append(self.map_elites_cfg[axis]["nbins"])
                min_descriptor.append(self.map_elites_cfg[axis]["min_val"])
                max_descriptor.append(self.map_elites_cfg[axis]["max_val"])
                descriptor_types.append(self.map_elites_cfg[axis]["descriptor"])
        self._grid_shape = tuple(grid_shapes)
        self._min_descriptor = np.array(min_descriptor)
        self._max_descriptor = np.array(max_descriptor)
        self._descriptor_types = descriptor_types
        self._ndescriptors = len(self._grid_shape)

    @property
    def centroids(self):
        if not hasattr(self, "_centroids"):
            self._centroids = compute_euclidean_centroids(
                grid_shape=self.grid_shape,
                minval=self.min_descriptor,
                maxval=self.max_descriptor,
            )
        return self._centroids


    @property
    def nbatch(self):
        return self.evolution_cfg["popsize"]
    

    def init_strategy(self, rng):
        """
        Initialize the MAPElites strategy by generating random genotypes
        and distributing them correctly along the archive.
        1. Generate random (uniformly sampled) genotypes
        2. Convert them into controls for simulator
        3. Evaluate the genotypes via simulator to get state, sensordata
        4. Compute fitness and behavioral metrics
        5. Feed everything to MAPElites.init_ask_tell
        """
        # Generate initial genotypes
        rng, rng_init_genotypes, rng_control, rng_init_ask_tell = jax.random.split(rng, 4)

        init_genotypes = np.random.uniform(
            low=PARAM_CLIP_MIN,
            high=PARAM_CLIP_MAX,
            size=(self.nbatch, self.control_generator.parameter_reshaper.total_params),
        )

        init_genotypes = jax.random.uniform(
            rng_init_genotypes,
            shape=(self.nbatch, self.control_generator.parameter_reshaper.total_params),
            minval=PARAM_CLIP_MIN,
            maxval=PARAM_CLIP_MAX,
        )


        # Evaluate initial genotypes
        control, genotypes_clipped = self.control_generator.generate_control_from_genotype(rng_control, init_genotypes, self.simulation_cfg["nsteps"])
        nthread = self.simulation_cfg.get("nthread", None)

        state, sensordata = super().simulate(self.simulator, control, nthread=nthread)
        sensordata_dict = self.simulator.extract_sensor_dict(sensordata)
        fitnesses, _ = super().evaluate(self.simulator, state, sensordata, self.evaluation_cfg)

        
        # initialize behavioral descriptor extractor object
        descriptors = self._get_descriptors(state, sensordata_dict)


        # Define emitter
        variation_fn = functools.partial(
            isoline_variation,
            iso_sigma=self.iso_sigma,
            line_sigma=self.line_sigma,
            minval=PARAM_CLIP_MIN,
            maxval=PARAM_CLIP_MAX,
        )

        mixing_emitter = MixingEmitter(
            mutation_fn=None, # mutation_fn=lambda x, y: (x, y),
            variation_fn=variation_fn,
            variation_percentage=1.0, # do not apply the mutation
            batch_size=self.nbatch,
        )

        # Define a metrics function
        metrics_fn = functools.partial(
            default_qd_metrics,
            qd_offset=0.0,
        )

        # Instantiate MAP-Elites
        strategy = MAPElites(
            scoring_function=None,
            emitter=mixing_emitter,
            metrics_function=metrics_fn,
        )

        # Note: for mixing_emitter, emitter state is obsolete.
        repertoire, emitter_state, metrics = strategy.init_ask_tell(
            # genotypes=genotypes_clipped,
            genotypes=init_genotypes,
            fitnesses=fitnesses,      # shape (nbatch,)
            descriptors=descriptors,  # shape (nbatch, ndescriptors)
            centroids=self.centroids,           # shape (ncentroids, ndescriptors)
            key=rng_init_ask_tell,
        )
        # # Note: for mixing_emitter, emitter state is obsolete.
        # repertoire_controls, emitter_state_controls, metrics_controls = strategy.init_ask_tell(
        #     genotypes=control,        # shape (nbatch, nsteps, njoints)
        #     fitnesses=fitnesses,      # shape (nbatch,)
        #     descriptors=descriptors,  # shape (nbatch, ndescriptors)
        #     centroids=self.centroids,           # shape (ncentroids, ndescriptors)
        #     key=rng_init_ask_tell,
        # )

        self.strategy = strategy
        self.repertoire = repertoire
        self.emitter_state = emitter_state # will be None for mixing_emitter
        self.metrics = metrics

        # self.repertoire_controls = repertoire_controls
        # self.emitter_state_controls = emitter_state_controls
        # self.metrics_controls = metrics_controls

    
    def _get_behavioral_metrics(self, state, sensordata):
        """
        Computes the behavioral metrics of the given state and sensor data.
        Type of metrics described in map_elites_cfg[<axis>]["descriptor"]
        Args:
        - state: np.ndarray of shape (nbatch, nstate)
        - sensordata: np.ndarray of shape (nbatch, nsensordata)
        Outputs:
        - metrics: dict containing the behavioral metrics
        """
        raise NotImplementedError
        # Implement the behavioral metrics computation logic here
        return None
    

    def get_control(self, nsteps: int, rng: chex.PRNGKey, nrep_per_genotype: int=1) -> np.ndarray:
        """
        Get control parameters for the Simulator.
        Args:
            nsteps (int): Number of steps for which to generate control parameters.
            rng (chex.PRNGKey): Random key for strategy.ask and resetting the CPG state.
        Returns:
            control (np.ndarray) (nreps, nbatch, nsteps, ncontrol): Control parameters for the simulator.

        Side effects:
            - Updates the `self.genotypes` with the new CLIPPED genotypes.
            - Updates the `self.control` with the control parameters generated from the CPG.
            - Updates the `self.cpg_modulation_params` with the modulation parameters extracted from the CPG.
        """
        rng, rng_ask, rng_control, rng_control_reps = jax.random.split(rng, 4)
        genotypes, _ = self.strategy.ask(
            repertoire=self.repertoire,
            emitter_state=self.emitter_state,
            key=rng_ask
        )

        control, genotypes_clipped = self.control_generator.generate_control_from_genotype(rng_control, genotypes, nsteps)
        if nrep_per_genotype > 1:
            # query the self.control_generator again to get multiple repetitions per genotype
            rngs = jax.random.split(rng_control_reps, nrep_per_genotype-1)
            controls_repeated = [
                self.control_generator.generate_control_from_genotype(rng_i, genotypes, nsteps)[0]
                for rng_i in rngs
            ]
            #concatenate along new first axis into shape (nrep_per_genotype, nbatch, nsteps, ncontrol)
            control = np.concatenate([control[np.newaxis, :, :, :]] + [ctrl[np.newaxis, :, :, :] for ctrl in controls_repeated], axis=0)
        else:
            # expand with a new axis along at axis index 0
            control = control[np.newaxis, :, :, :]

        # self.genotypes = genotypes_clipped
        self.genotypes = genotypes
        self.control = control

        return control

    
    def tell(self, fitness: np.ndarray, state: np.ndarray, sensordata_dict: dict):
        """
        Tell the evolution strategy about the fitness of the current population.
        Also includes the state and sendordata_dicts for possible extraction of descriptors
        necessary for the MAP-Elites algorithm.
        Args:
            fitness (np.ndarray): Fitness values for the current population.
            state (np.ndarray): The current state of the population.
            sensordata_dict (dict): The sensor data collected during the simulation.
        Returns:
            None
        Side-effects:
            - Updates the `repertoire` with the updated archive
            - Updates the `emitter_state` with the new emitter state, Typically None for mixing_emitter
            - Updates the `metrics` with the new metrics from the archive.
            - Updates the `state` with the new simulation data.
            - Updates the `sensordata_dict` with the new sensordata dictionary from the simulation.
            - Updates the `fitness` with the new fitness values OF THE BATCH, NOT OF THE ARCHIVE
                    Note: self.repertoire.fitnesses contains fitnesses of the archive.
        """

        descriptors = self._get_descriptors(state, sensordata_dict)


        repertoire, emitter_state, metrics = self.strategy.tell(
            genotypes=self.genotypes,  # shape (nbatch, nparams)
            fitnesses=fitness,         # shape (nbatch,)
            descriptors=descriptors,   # shape (nbatch, ndescriptors)
            repertoire=self.repertoire,
            emitter_state=self.emitter_state,
        )

        # repertoire_controls, emitter_state_controls, metrics_controls = self.strategy.tell(
        #     genotypes=self.control,  # shape (nbatch, nsteps, njoints)
        #     fitnesses=fitness,         # shape (nbatch,)
        #     descriptors=descriptors,   # shape (nbatch, ndescriptors)
        #     repertoire=self.repertoire_controls,
        #     emitter_state=self.emitter_state_controls,
        # )

        if self.wandb_logger is not None:
            self.wandb_logger.log(metrics)

            fig, ax = plot_2d_map_elites_repertoire(
            repertoire=self.repertoire,
            descriptor_types=self.descriptor_types,
            minval=self.min_descriptor,
            maxval=self.max_descriptor,
            vmin=0, # set colorscale minimum value at 0.
            )
            ax.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
            image_path = f"archive_of_elites.jpg"
            fig.savefig(image_path)
            self.wandb_logger.log_image_per_step(image_name="Archive of Elites", image_path=image_path)

        self.repertoire = repertoire
        self.emitter_state = emitter_state
        self.metrics = metrics

        # self.repertoire_controls = repertoire_controls
        # self.emitter_state_controls = emitter_state_controls
        # self.metrics_controls = metrics_controls

        self.state = state
        self.sensordata_dict = sensordata_dict
        self.fitness = fitness


    def get_solution_control(self):
        """Only returns the control of the best performing genotype for quick visualization.
        If multiple reps per genotype, only the first rep is returned."""
        index_in_batch_dim = np.argmax(self.fitness)
        best_genotype_control = self.control[0, index_in_batch_dim, :, :] # (nrep_per_genotype, popsize, nsteps, ncontrol)
        return best_genotype_control[np.newaxis, :, :] # (1, nsteps, ncontrol) -> batch dim required for simulator.rollout

    
    def get_solution_policy_params(self):
        """returns a repertoire containing genotypes, fitnesses, descriptors and centroids.
           For further analyses: everything can be reconstructed from this repertoire."""
        return self.repertoire
    
    # def get_solution_repertoire_controls(self):
    #     """returns a repertoire containing controls, fitnesses, descriptors and centroids.
    #        For further analyses: everything can be reconstructed from this repertoire."""
    #     return self.repertoire_controls
       

    
    
    def _get_descriptors(self, state, sensordata):
        """
        Args:
            state (np.ndarray) (nbatch, nsteps, nsensordata): The current state of the system.
            sensordata (dict): The sensor data collected during the simulation.
        Returns:
            descriptors (np.ndarray) (nbatch, ndescriptors)
        """
        descriptor_extractor = BehavioralDescriptorsExtractor(
            state=state,
            sensordata=sensordata,
            arm_setup=self.simulator.morph_cfg["num_segments_per_arm"],
        )
        descriptors = np.zeros((self.nbatch, 0))
        for descriptor_type in self.descriptor_types:
            if hasattr(descriptor_extractor, descriptor_type):
                axis_descriptors = getattr(descriptor_extractor, descriptor_type)
                descriptors = np.concatenate((descriptors, axis_descriptors[:, np.newaxis]), axis=1)
            else:
                raise ValueError(f"Descriptor {descriptor_type} not found as attribute in BehavioralDescriptorsExtractor.")
        assert self.ndescriptors == descriptors.shape[1], f"Expected {self.ndescriptors} descriptors, but got {descriptors.shape[1]}"

        return descriptors      
    


    


if __name__ == "__main__":
    """Example usage of the MAPElitesOptimizer with CPG control generator."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import jax
    
    from cpg_convergence.cpg import BS_CPG
    from cpg_convergence.simulator import BrittleStarSimulator
    from cpg_convergence.optimizer.map_elites import MAPElitesOptimizer
    from cpg_convergence.wandb_utils import WandbLogger
    import pickle

    # Example use of the OpenESOptimizer subclass:
    rng = jax.random.PRNGKey(0)
    tmp_dir = os.path.join(os.path.dirname(__file__), "../../../tmp/")
    results_dir = os.path.join(os.path.dirname(__file__), "../../../results/")

    morph_cfg = {
            "num_arms": 5,
            "num_segments_per_arm": [7, 7, 7, 7, 7],
            "use_p_control": True
        }
    
    arena_cfg = {
        "size": [4., 4.],
        "sand_ground_color": False
    }

    simulation_cfg = {
        "nsteps": 135,
        "nthread": 4,
    }

    rendering_cfg = {
        "camera": [0, 1],
        "shape": ( 480, 640 ),
        "transparent": False,
        "color_contacts": True,
    }

    evolution_cfg = {
        "popsize": 100,
        "ngen": 70,
    }

    evaluation_cfg = {
        "reward_expr": "xy-distance",
        "cost_expr": "actuator_forces",
        "penalty_expr": None,
        "additive_or_multiplicative": "multiplicative",
        "alpha": 10.0,
        "beta": 1.0,
    }

    map_elites_cfg = {
        "axis_0": {
            "descriptor": "bilateral_contralateral_score",
            "min_val": -1.0,
            "max_val": 1.0,
            "nbins": 10
        },
        "axis_1": {
            "descriptor": "assistive_score",
            "min_val": 1.0,
            "max_val": 5.0,
            "nbins": 10
        }
    }

    nsteps = simulation_cfg["nsteps"]
    ngen = evolution_cfg["ngen"]
    nrep_per_genotype = 1
    method = "base"
    weight_scale = 5.0
    wandb_logger = None

    # wandb_logger = WandbLogger(
    #     project="cpg_convergence",
    #     group="test",
    #     run_name=f"MAPElites_{method}_ws{weight_scale}",
    # )

    simulator = BrittleStarSimulator(morph_cfg=morph_cfg, arena_cfg=arena_cfg)

    rng, rng_cpg = jax.random.split(rng)
    cpg = BS_CPG(arm_setup=morph_cfg["num_segments_per_arm"],
                 rng=rng_cpg,
                 method=method,
                 weight_scale=weight_scale,
                 )

    rng, rng_control_generator = jax.random.split(rng, 2)
    control_generator = CPGControlGeneratorBS(cpg=cpg)

    rng, rng_strategy_init, rng_optimize = jax.random.split(rng, 3)
    optimizer = MAPElitesOptimizer(
        control_generator=control_generator,
        simulator=simulator,
        evolution_cfg=evolution_cfg,
        map_elites_cfg=map_elites_cfg,
        evaluation_cfg=evaluation_cfg,
        simulation_cfg=simulation_cfg,
        rng=rng_strategy_init,
        wandb_logger=wandb_logger,
    )

    print(f"optimizer.simulator.morphology.arm_setup: {optimizer.simulator.morph_cfg['num_segments_per_arm']}")
    print(f"optimizer.grid_shape: {optimizer.grid_shape}")
    print(f"optimizer.nbatch: {optimizer.nbatch}")

    optimizer.optimize(
        simulator=simulator,
        nsteps=nsteps,
        rng=rng_optimize,
        ngen=ngen,
        evaluation_cfg=evaluation_cfg,
        nrep_per_genotype=nrep_per_genotype,
        aggregation_over_reps="max",
        wandb_logger=wandb_logger,
    )

    

    print(optimizer.metrics)


    # optimizer.visualize_solution(
    #     simulator=simulator,
    #     video_path=os.path.join(tmp_dir, "map_elites_optimizer_solution.mp4"),
    #     rendering_cfg=rendering_cfg,
    #     wandb_logger=wandb_logger,
    # )

    # Save the optimizer's repertoire using pickle
    # repertoire_path = os.path.join(results_dir, "b02_r33_r35", "medium_repertoire.pkl")
    repertoire_path = "/home/idlab515/OneDrive/Documents/DOCUMENTEN/4_PhD/BioCodespace/CPGConvergence/results/b02_r33_35/slow_repertoire.pkl"
    # repertoire_control_path = os.path.join(results_dir, "b02_r33_r35", "medium_repertoire_controls.pkl")
    with open(repertoire_path, "wb") as f:
        pickle.dump(optimizer.repertoire, f)
    # with open(repertoire_control_path, "wb") as f:
    #     pickle.dump(optimizer.repertoire_controls, f)
