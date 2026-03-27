import numpy as np
from typing import Tuple, Optional, Literal

from evosax import OpenES, FitnessShaper
import chex
import jax


from cpg_convergence.defaults import CENTERED_RANK, MAXIMIZE_FITNESS
from cpg_convergence.optimizer.base import BaseOptimizer
from cpg_convergence.control_generator import CPGControlGeneratorBS
from cpg_convergence.wandb_utils import WandbLogger



class OpenESOptimizer(BaseOptimizer):
    def __init__(
            self,
            control_generator: CPGControlGeneratorBS,
            evolution_cfg: dict,
            rng_strategy_init: chex.PRNGKey,
            wandb_logger: Optional[WandbLogger] = None,
            ):
        """
        Instantiates attributes
            strategy (OpenES): The evolution strategy instance.
            cpg_parameter_reshaper (CPGParameterReshaper): The reshaper that allows smooth interaction between raw policy params
             and modulation params for the CPG, with structure:
                keys:
                    - "omegas": The oscillation frequencies.
                    - "rhos": The coupling strengths.
                    - "X": The offsets.
                    - "R": The amplitudes
            fitness_shaper (FitnessShaper): The fitness shaper for the evolution strategy.
            es_state: the initial state of the evolution strategy.
            wandb_logger: allows logging intermediate metrics

        args:
            control_generator (CPGControlGeneratorBS): The CPG instance to use for the optimization.
            evolution_cfg (dict): Configuration dictionary for the evolution process. Contains:
                - "popsize": Population size for the evolution strategy.
                - "ngen": Number of generations for the evolution strategy. 
        """
        self.evolution_cfg = evolution_cfg
        self.wandb_logger = wandb_logger
        self.control_generator = control_generator
        self.strategy = OpenES(
            popsize=self.evolution_cfg["popsize"],
            num_dims=self.control_generator.cpg.parameter_reshaper.total_params,
        )
        self.fitness_shaper = FitnessShaper(
            centered_rank=CENTERED_RANK,
            maximize=MAXIMIZE_FITNESS,  # This is consistent with the way the simulator calculates fitness
        )

        self.es_state = self.strategy.initialize(rng_strategy_init, self.strategy.params_strategy)


    def get_control(self, nsteps: int, rng: chex.PRNGKey, nrep_per_genotype=1) -> np.ndarray:
        """
        Get control parameters for the Simulator.
        Args:
            nsteps (int): Number of steps for which to generate control parameters.
            rng (chex.PRNGKey): Random key for strategy.ask and resetting the CPG state.
        Returns:
            control (np.ndarray) (nreps, nbatch, nsteps, ncontrol): Control parameters for the simulator.

        Side effects:
            - Updates the `self.es_state` with the new state of the evolution strategy.
            - Updates the `self.params_evosax_flat_clipped` with the clipped parameters.
            - Updates the `self.control` with the control parameters generated from the CPG.
            - Updates the `self.modulation_params` with the modulation parameters extracted from the CPG.
        """
        rng, rng_ask, rng_control, rng_control_reps = jax.random.split(rng, 4)
        genotypes_flat, es_state = self.strategy.ask(rng_ask, self.es_state)

        control, genotypes_clipped = self.control_generator.generate_control_from_genotype(rng_control, genotypes_flat, nsteps)
        if nrep_per_genotype > 1:
            # query the self.control_generator again to get multiple repetitions per genotype
            rngs = jax.random.split(rng_control_reps, nrep_per_genotype-1)
            controls_repeated = [
                self.control_generator.generate_control_from_genotype(rng_i, genotypes_flat, nsteps)[0]
                for rng_i in rngs
            ]
            #concatenate along new first axis into shape (nrep_per_genotype, nbatch, nsteps, ncontrol)
            control = np.concatenate([control[np.newaxis, :, :, :]] + [ctrl[np.newaxis, :, :, :] for ctrl in controls_repeated], axis=0)
        else:
            # expand with a new axis along at axis index 0
            control = control[np.newaxis, :, :, :]

        self.es_state = es_state
        self.genotypes = genotypes_clipped
        self.control = control

        return control
    


    def tell(self, fitness: np.ndarray, state: np.ndarray, sensordata_dict: np.ndarray):
        """
        Tell the evolution strategy about the fitness of the current population.
        Also includes the state and sensordata_dict for possible further metrics extraction if required.
        Args:
            fitness (np.ndarray): Fitness values for the current population.
            state (np.ndarray): Mujoco state information for the current population.
            sensordata_dict (dict): Mujoco sensordata dictionary for the current population
        Returns:
            None
        Side-effects:
            - updates the `es_state` with new evolution strategy state.
            - Updates the `self.fitness` with the new fitness values.
        """
        # Reshape fitness to match the population size
        fitness_shaped = self.fitness_shaper.apply(self.genotypes, fitness)
        es_state = self.strategy.tell(self.genotypes, fitness_shaped, self.es_state, self.strategy.params_strategy)

        self.fitness = fitness # NOT FITNESS SHAPED, fitness shaped is just for internal processing within the tell method
        self.es_state = es_state
    

    def get_solution_control(self) -> np.ndarray:
        """
        Get the control for the best solution found by the evolution strategy.
        Returns:
            np.ndarray (1, nsteps, ncontrol): Control parameters for the best solution.
        """
        index_in_batch_dim = np.argmax(self.fitness)
        solution_control = self.control[0, index_in_batch_dim, :, :] # (nrep_per_genotype, popsize, nsteps, ncontrol)
        return solution_control[np.newaxis, :, :]  # (1, nsteps, ncontrol) -> nbatch dimension required for simulator.rollout

    def get_solution_policy_params(self):
        """For open_es, the policy params required to reproduce the results are the modulation parameters of the CPG."""
        index_in_batch_dim = np.argmax(self.fitness)
        solution_genotype = jax.tree_util.tree_map(
            lambda x: x[index_in_batch_dim:index_in_batch_dim + 1],
            self.genotypes
        )
        return solution_genotype





if __name__ == "__main__":
    import jax
    import os
    
    from cpg_convergence.cpg import BS_CPG
    from cpg_convergence.simulator import BrittleStarSimulator
    from cpg_convergence.control_generator import CPGControlGeneratorBS

    # Example use of the OpenESOptimizer subclass:
    rng = jax.random.PRNGKey(0)
    tmp_dir = os.path.join(os.path.dirname(__file__), "../../../tmp/")

    morph_cfg = {
            "num_arms": 8,
            "num_segments_per_arm": [0, 0, 8, 13, 2, 5, 0, 7],
            "use_p_control": True
        }
    
    arena_cfg = {
        "size": [4., 4.],
        "sand_ground_color": False
    }
    evolution_cfg = {
        "popsize": 4,
        "ngen": 10,
    }
    evaluation_cfg = {
        "reward_expr": "x-distance",
        "cost_expr": None,
        "penalty_expr": None,
        "additive_or_multiplicative": "multiplicative",
        "alpha": 1.0,
        "beta": 1.0,
    }

    nsteps = 125 # 5 seconds of control at 25 FPS
    ngen = evolution_cfg["ngen"]
    nrep_per_genotype = 2


    simulator = BrittleStarSimulator(morph_cfg=morph_cfg, arena_cfg=arena_cfg)

    rng, rng_cpg = jax.random.split(rng)
    cpg = BS_CPG(arm_setup=morph_cfg["num_segments_per_arm"],
            rng=rng_cpg,
            method="base"
        )

    rng, rng_control_generator = jax.random.split(rng)
    control_generator = CPGControlGeneratorBS(
        cpg=cpg,
    )

    rng, rng_strategy_init, rng_optimize = jax.random.split(rng, 3)
    optimizer = OpenESOptimizer(
        control_generator,
        evolution_cfg,
        rng_strategy_init)
    
    optimizer.optimize(
        simulator=simulator,
        nsteps=nsteps,
        rng=rng_optimize,
        ngen=ngen,
        evaluation_cfg=evaluation_cfg,
        nrep_per_genotype=nrep_per_genotype,
        aggregation_over_reps="max"
    )

    optimizer.visualize_solution(
        simulator=simulator,
        video_path=os.path.join(tmp_dir, "open_es_optimizer_solution.mp4"),
    )


