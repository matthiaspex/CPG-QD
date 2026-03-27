"""
All functionalities to run a complete optimization.
Generally consists of 4 components:
1. Evolutionary parameter generation: evosax-OpenES, QDax-MAPElites, etc.
2. Controller: generate the controls (CPGs) for the simulator based on the optimized parameters.
3. Simulation: run the simulator with the generated controls and collect data.
4. Evaluation: assess the performance of the simulation and update the optimizer.

These 4 components are looped iteratively to obtain the best parameters for the controller and the simulation.
"""
from matplotlib.transforms import ScaledTranslation
import numpy as np
import time
from typing import Tuple, Optional, Literal
from abc import ABC, abstractmethod
import chex
import os
import sys
import pickle
import jax

from cpg_convergence.simulator import BrittleStarSimulator
from cpg_convergence.wandb_utils import WandbLogger
from cpg_convergence.defaults import CPG_OUTPUT_CLIP_MIN, CPG_OUTPUT_CLIP_MAX, CPG_OUTPUT_SCALE_MIN, CPG_OUTPUT_SCALE_MAX
from cpg_convergence.utils import clip_and_rescale, save_to_pickle
from cpg_convergence.experiment_utils.cpg import create_csv, add_csv_entry
from cpg_convergence.behavioral_descriptors import BehavioralDescriptorsExtractor


class BaseOptimizer(ABC):
    @abstractmethod
    def get_control(self, nsteps: int, rng: chex.PRNGKey) -> np.ndarray:
        #Still look how to treat this with f
        """
        Get control parameters for the Simulator.
        Args:
            nsteps (int): Number of steps for which to generate control parameters.
            rng (chex.PRNGKey): Random key for strategy.ask and resetting the CPG state.
        Returns:
            control (np.ndarray) (nbatch, nsteps, ncontrol): Control parameters for the simulator.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    

    @staticmethod
    def simulate(
            simulator: BrittleStarSimulator,
            control: np.ndarray,
            **kwargs,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the simulation with the given control parameters.
        Args:
            simulator (BrittleStarSimulator): The simulator instance to run the simulation.
            control (np.ndarray): Control parameters for the simulation.
            **kwargs: Additional keyword arguments for the simulator.rollout method:
                - nthread (int, default: None): Number of threads to use for parallel simulation.
                  if not specified: uses all available CPU cores. (multiprocessing.cpu_count())
        Returns:
            state (np.ndarray) (nbatch, nsteps, nstate): The state data from the mujoco.rollout.rollout
            sensordata (np.ndarray) (nbatch, nsteps, nsensordata): The sensor data from the mujoco.rollout.rollout.
        """
        control = clip_and_rescale(
            control,
            clip_min=CPG_OUTPUT_CLIP_MIN,
            clip_max=CPG_OUTPUT_CLIP_MAX,
            rescale_min=CPG_OUTPUT_SCALE_MIN,
            rescale_max=CPG_OUTPUT_SCALE_MAX,
        )
        state, sensordata = simulator.rollout(control=control, **kwargs)

        return state, sensordata

    @staticmethod
    def evaluate(
            simulator: BrittleStarSimulator,
            state: np.ndarray,
            sensordata: np.ndarray,
            evaluation_cfg: dict
        ) -> np.ndarray:
        """
        Evaluate the performance of the simulation.
        Args:
            state (np.ndarray) (nbatch, nsteps, nstate): State data from the simulation.
            sensordata (np.ndarray) (nbatch, nsteps, nsensordata): Sensor data from the simulation.
            evaluation_cfg: Additional keyword arguments for evaluation, includes but not limited to:
                - reward_expr (str, default: "x-distance"): Expression to calculate the reward, e.g., "x-distance".
                - cost_expr (str, optional): Expression to calculate the cost, e.g., "torques".
                - penalty_expr (str, optional): Expression to calculate the penalty, e.g., "stability".
                - additive_or_multiplicative (Literal["additive", "multiplicative"], default: "multiplicative"):
                - alpha (float, default: 1.): Weighting factor for the cost in the fitness calculation.
        Returns:
            fitness (np.ndarray) (nbatch): Fitness value for each batch in the rollout.
        """
        fitness, metrics = simulator.get_fitness(state, sensordata, **evaluation_cfg)
        return fitness, metrics
    
    @abstractmethod
    def tell(self, fitness: np.ndarray, state: np.ndarray, sensordata: np.ndarray):
        """Tell the optimizer about the fitness of the current control parameters."""
        raise NotImplementedError("This method should be implemented by subclasses.")
        

    def optimize(
            self,
            simulator: BrittleStarSimulator,
            nsteps: int,
            rng: chex.PRNGKey,
            ngen: int,
            evaluation_cfg: dict,
            nrep_per_genotype: int = 1,
            aggregation_over_reps: Literal["max", "min"] = "max",
            wandb_logger: Optional[WandbLogger] = None,
            nthread: Optional[int] = None,
            csv_header: Optional[list] = None,
            prespecified_csv_entries: Optional[dict] = None,
            csv_path: Optional[str] = None,
        ):
        """
        Run a full optimization loop.
        Every loop consists of:
        1. Get control parameters - `get_control()`.
        3. Give controls to simulator (includes clipping), get state and sensordata back - `simulate()`.
        4. Give state and sensordata to evaluate, get fitness back - `evaluate()`
        5. Give fitness back to the optimizer - `tell()`

        Args:
            simulator (BrittleStarSimulator): The simulator instance.
            nsteps (int): Number of steps for which to generate control parameters.
            rng (chex.PRNGKey): Random key for the optimization process.
            ngen (int): Number of generations to run the optimization.
            nrep_per_genotype (int): Number of repetitions per generation.
            aggregation_over_reps (Literal["mean", "max", "min"]): Method to aggregate fitness.
            evaluation_cfg (dict): Configuration dictionary for the evaluation process. How fitness is computed
            wandb_logger (Optional[WandbLogger]): WandB logger instance for logging metrics.
            nthread (Optional[int]): Number of threads to use for parallel simulation.
            csv_header (Optional[list]): Header for the CSV file to log results.
                Must contain: "generation" and "genotype_id" as well as the
                automatically recognized headers: "fitness", "disk_elevation", "ground_contact_fraction",
                                                  "sine_total_displacement", "cosine_total_displacement"
            prespecified_csv_entries (Optional[dict]): Prespecified entries to include in each CSV row
            csv_path (Optional[str]): Path to the CSV file to log results.
        """
        print("Optimization started.")
        if wandb_logger is not None:
            wandb_logger.reset_step()
        
        if csv_path is not None:
            assert csv_header is not None, "CSV header must be provided if CSV path is specified."
            if not os.path.exists(csv_path):
                create_csv(csv_path, csv_header)

        for gen in range(ngen):
            start_gen_time = time.time()
            rng, step_rng = jax.random.split(rng, 2)
            control = self.get_control(nsteps, rng=step_rng, nrep_per_genotype=nrep_per_genotype)
            # control shape: (nrep_per_genotype, popsize, nsteps, ncontrol)
            print(f"Generation {gen}/{ngen-1}: Control shape: {control.shape}")

            fitness_stack = None            
            state_stack = None
            sensordata_stack = None
            for rep in range(nrep_per_genotype):
                control_rep = control[rep, :, :, :]  # shape: (popsize, nsteps, ncontrol)

                simulate_time_start = time.time()
                state, sensordata = self.simulate(simulator, control_rep, nthread=nthread)
                if state_stack is None or sensordata_stack is None:
                    state_stack = np.expand_dims(state, axis=0)  # shape: (1, popsize, nsteps, nstate)
                    sensordata_stack = np.expand_dims(sensordata, axis=0)  # shape: (1, popsize, nsteps, nsensordata)
                else:
                    state_stack = np.concatenate([state_stack, np.expand_dims(state, axis=0)], axis=0)
                    sensordata_stack = np.concatenate([sensordata_stack, np.expand_dims(sensordata, axis=0)], axis=0)

                simulate_time = time.time() - simulate_time_start

                evaluate_time_start = time.time()
                fitness, metrics = self.evaluate(simulator, state, sensordata, evaluation_cfg=evaluation_cfg)
                if fitness_stack is None:
                    fitness_stack = fitness[np.newaxis, :]  # shape: (1, popsize)
                else:
                    fitness_stack = np.concatenate([fitness_stack, fitness[np.newaxis, :]], axis=0)

                evaluate_time = time.time() - evaluate_time_start

                sensordata_dict = simulator.extract_sensor_dict(sensordata)
                
                descriptors_extractor = BehavioralDescriptorsExtractor(state, sensordata_dict, arm_setup=simulator.morph_cfg["num_segments_per_arm"])
                
                csv_time_start = time.time()
                if csv_path is not None:
                    # add popsize independent entries
                    csv_entry_qd_metrics = {}
                    for head in csv_header:
                        if head in ["qd_score", "coverage", "max_fitness"] and hasattr(self, "metrics"):
                            csv_entry_qd_metrics.update({f"{head}": self.metrics[head]})
                    # add per genotype entries (across popsize)
                    for i in range(fitness.shape[0]): # iterate over popsize
                        entry_tmp = prespecified_csv_entries.copy() if prespecified_csv_entries is not None else {}
                        entry_tmp.update(csv_entry_qd_metrics)
                        entry_tmp.update({"generation": gen})
                        entry_tmp.update({"genotype_id": i})
                        entry_tmp.update({"genotype_rep": rep})
                        for head in csv_header:
                            if not head in prespecified_csv_entries.keys():
                                if head == "fitness":
                                    entry_tmp.update({f"{head}": fitness[i]})
                                elif head in ["disk_elevation", "ground_contact_fraction",
                                              "sine_total_displacement", "cosine_total_displacement",
                                              "assistive_score", "bilateral_contralateral_score",
                                              "bilateral_score", "contralateral_score",
                                              "bilateral_score_grf", "contralateral_score_grf",]:
                                    entry_tmp.update({f"{head}": getattr(descriptors_extractor, head)[i]})
                                else:
                                    pass


                        add_csv_entry(
                            file_path=csv_path,
                            entry=entry_tmp
                        )
                csv_time = time.time() - csv_time_start
                print(f"\t\trep_per_genotype: {rep}, simulate: {simulate_time:.2f}s, evaluate: {evaluate_time:.2f}s, csv: {csv_time:.2f}s)")
            
            aggr_fn = {"max": np.max,
                       "min": np.min
                        }[aggregation_over_reps]
            aggr_ind_fn = {"max": np.argmax,
                           "min": np.argmin
                            }[aggregation_over_reps]

            aggregated_fitness = aggr_fn(fitness_stack, axis=0)
            fitness_idx = aggr_ind_fn(fitness_stack, axis=0)

            popsize = state_stack.shape[1]
            state_tell = state_stack[fitness_idx, np.arange(popsize), :, :]  # shape: (popsize, nsteps, nstate)
            sensordata_tell = sensordata_stack[fitness_idx, np.arange(popsize), :, :] # shape: (popsize, nsteps, nsensordata)

            sensordata_tell_dict = simulator.extract_sensor_dict(sensordata_tell)

            self.tell(aggregated_fitness, state_tell, sensordata_tell_dict)
            gen_time = time.time() - start_gen_time

            if wandb_logger is not None:
                wandb_logger.log(metrics)
                wandb_logger.log({"gen_time": gen_time})
                wandb_logger.log_histogram("fitness distribution", fitness)
   
                wandb_logger.advance_step()


            print(f"""\tGeneration {gen}/{ngen-1} completed in {gen_time:.2f} seconds.
                  """)
        
        print("Optimization completed.")


    @abstractmethod
    def get_solution_control(self) -> np.ndarray:
        """
        Get the best control parameters found during the optimization.
        This should be used after an optimization loop has been completed.
        """
        raise NotImplementedError("get_solution_control method is not implemented yet. Please implement the required method in this class.")
    
    @abstractmethod
    def get_solution_policy_params(self):
        """
        Get the relevant parameters to store, with which the results can be reproduced.
        """
        raise NotImplementedError("get_policy_params method is not implemented yet. Please implement the required method in this class.")
            

    def visualize_solution(
            self,
            simulator: BrittleStarSimulator,
            video_path: Optional[str] = None,
            wandb_logger: Optional[WandbLogger] = None,
            video_name: Optional[str] = None,
            caption: Optional[str] = None,
            show: bool = False,
            rendering_cfg: Optional[dict] = None,
    ):
        """
        Visualize the solution by logging relevant data to WandB.
        Args:
            simulator (BrittleStarSimulator): The simulator instance.
            solution_control (np.ndarray): The optimized control parameters to visualize.
            wandb_logger (Optional[WandbLogger]): The WandB logger instance.
            video_name (Optional[str]): Name of the video to log: must be specified if wandb_logger is not None.
            video_path (Optional[str]): Path to the video file to log: must be specified if wandb_logger is not None.
            caption (Optional[str]): Caption for the video.
            show (bool): Whether to show the video in a window. Defaults to False. Only works in interactive python environments.
            rendering_cfg (Optional[dict]): Additional keyword arguments for rendering, includes but not limited to:
                - camera (int): Camera index to use for rendering. Default is 0.
                - shape (Tuple[int, int]): Shape of the rendered frames (height, width). Default is (480, 640).
                - transparent (bool): Whether to render with a transparent background. Default is False.
                - light_pos (Optional[List[float]]): Position of the light source in the scene. If None, no light is added. Default is None.
                - color_contacts (bool): Whether to color segments which touch the ground red in visualisation
        """
        print("Visualizing solution... \n ")
        if rendering_cfg is None:
            rendering_cfg = {}
        solution_control = self.get_solution_control()
        state, sensordata = self.simulate(simulator, solution_control)
        simulator.visualize_rollout(state, sensordata, show=show, path=video_path, **rendering_cfg)
        if wandb_logger is not None and video_path is not None:
            # Try if video can be logged with wandb_logger, do nothing if it does not exist
            try:
                wandb_logger.log_video(video_name, video_path, caption)
            except Exception as e:
                print(f"MuJoCo video not rendered, so not logged to WandB: {e}")

        self.solution_sensordata = sensordata
        self.solution_state = state
        print("Solution visualization completed.")

    
    def save_solution(
            self,
            simulator: BrittleStarSimulator,
            save_dir: str,
            policy_path: Optional[str] = None,
            control_path: Optional[str] = None,
            state_path: Optional[str] = None,
            sensordata_path: Optional[str] = None,
            repertoire_controls_path: Optional[str] = None,
        ):
        """
        Save the solution to disk.
        Args:
            simulator (BrittleStarSimulator): The simulator instance.
            save_dir (str): Directory where the solution will be saved.
            policy_path (Optional[str]): Path to save the policy parameters (CPG modulation parameters).
            control_path (Optional[str]): Path to save the control parameters.
            state_path (Optional[str]): Path to save the state data.
            sensordata_path (Optional[str]): Path to save the sensor data.
        """
        if policy_path is None:
            policy_path=os.path.join(save_dir, "solution_policy_params.pkl") # former solution_cpg_modulation_params.pkl
        if control_path is None:
            control_path=os.path.join(save_dir, "solution_control.npy")
        if state_path is None:
            state_path=os.path.join(save_dir, "solution_state.npy")
        if sensordata_path is None:
            sensordata_path=os.path.join(save_dir, "solution_sensordata.pkl")
        if repertoire_controls_path is None:
            repertoire_controls_path=os.path.join(save_dir, "solution_repertoire_controls.pkl")

        solution_policy_params = self.get_solution_policy_params()
        solution_control = self.get_solution_control()

        solution_state = self.solution_state
        solution_sensordata = simulator.extract_sensor_dict(self.solution_sensordata)

        save_to_pickle(solution_policy_params, policy_path)
        np.save(control_path, solution_control)

        np.save(state_path, solution_state)
        save_to_pickle(solution_sensordata, sensordata_path)

        try:
            repertoire_controls = self.get_solution_repertoire_controls()
            save_to_pickle(repertoire_controls, repertoire_controls_path)
            print("Repertoire with controls saved succesfully.")
        except:
            print("Failed to save repertoire with controls.")


        
    