"""Evaluate CPGs by chaining them to a simulator and optimizer and assessing performance metrics."""
import argparse
parser = argparse.ArgumentParser(description="Run brittle star CPG sweeps")
parser.add_argument("--exp_id", type=str, help="Experiment id")
parser.add_argument("--nrepetitions", type=int, help="Number of repetitions")
parser.add_argument("--popsize", type=int, help="Population size")
parser.add_argument("--ngen", type=int, help="Number of generations")
parser.add_argument("--nrep_per_genotype", type=int, help="Repeats per genotype")

parser.add_argument("--method", type=str, help="Method (str)")
parser.add_argument("--arm_size", type=int, help="Arm size (int)")
parser.add_argument("--weight_coupling", type=float, help="Weight coupling (float)")

parser.add_argument("--cuda_devices", type=str, help="CUDA_VISIBLE_DEVICES (empty for CPU)")
parser.add_argument("--max_threads", type=int, help="Max threads to use")
args = parser.parse_args()

# propagate parsed values into globals so the rest of the script can use them
exp_id = args.exp_id
NREPETITIONS = args.nrepetitions
POPSIZE = args.popsize
NGEN = args.ngen
NREP_PER_GENOTYPE = args.nrep_per_genotype

method = args.method
arm_size = args.arm_size
weight_coupling = args.weight_coupling

CUDA_VISIBLE_DEVICES = args.cuda_devices
NTHREADS_MAXIMUM = args.max_threads



# ==================== Computational resources setup ====================
# Always keep sufficient cores open for others to use (try to not go above 50% of available cores)
# Needs to be done before system imports that use threading (e.g., numpy, jax)
from multiprocessing import cpu_count
import os
import time

os.environ["MUJOCO_GL"] = "egl"  # for headless rendering on servers
# CUDA_VISIBLE_DEVICES = "0"  # e.g., "0", "1", ... for specific GPU, "" to use CPU only
# NTHREADS_MAXIMUM = 12  # set an upper limit to avoid overloading systems with many cores, max olifant 32, max dell 14


print(f"Number of CPU cores found for threading: {cpu_count()}")
print(f"Setting maximum threads to (defaults): {NTHREADS_MAXIMUM}")
print(f"Setting CUDA_VISIBLE_DEVICES to: {CUDA_VISIBLE_DEVICES} (empty string means no GPU)")
threads = min(cpu_count()-2, NTHREADS_MAXIMUM)
max_threads = max(1, threads)
print(f"Acutal threads used: {max_threads}")
os.environ["OMP_NUM_THREADS"] = str(max_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
os.environ["MKL_NUM_THREADS"] = str(max_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # limit GPU memory usage to 90%
if CUDA_VISIBLE_DEVICES == "":
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    print("No GPU will be used for this run.")
else:
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    print(f"GPU {CUDA_VISIBLE_DEVICES} will be used for this run.")

# ==================== System imports ===================================
import sys

from datetime import datetime
from pprint import pprint
import time

import numpy as np
import chex
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import jax
from jax import numpy as jnp

from cpg_convergence.control_generator import CPGControlGeneratorBS
from cpg_convergence.optimizer.map_elites import MAPElitesOptimizer
from cpg_convergence.cpg import BS_CPG
from cpg_convergence.simulator import BrittleStarSimulator
from cpg_convergence.wandb_utils import WandbLogger

from cpg_convergence.experiment_utils.cpg import create_experiment_directory, write_metadata_dict_to_txt, \
    create_csv
from cpg_convergence.defaults import CPG_DEFAULT_DT, CPG_DEFAULT_SOLVER
import argparse

# ==============================
# CONFIGURABLES
# exp_id = "b02_r50"
seed = 42

# NREPETITIONS = 1 # complete experiment (intended for batch 02)
# POPSIZE = 4 # read via argparse
# NGEN = 10 # read via argparse
NSTEPS_CONTROL = 135
# NREP_PER_GENOTYPE = 1 # repeats per genotype (intended for batch 03)
SOLVER = CPG_DEFAULT_SOLVER
CPG_DT = CPG_DEFAULT_DT
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# method = "base"
ring_size = 5
# arm_size = 2
# weight_coupling = 5



csv_header = ["method", "ring_size", "arm_size", "weight_coupling", "run_id", # unique for entire experiment
              "spectral_gap", "induced_norm", "n_oscillators", "n_couplings", # unique for entire experiment
              "qd_score", "coverage", "max_fitness",                          # per generation
              "generation", "genotype_id", "genotype_rep",                    # per individual rollout (also following lines)
              "fitness",
            #   "disk_elevation", "ground_contact_fraction",
            #   "sine_total_displacement", "cosine_total_displacement",
              "assistive_score", "bilateral_contralateral_score",
            #   "bilateral_score", "contralateral_score",
            #   "bilateral_score_grf", "contralateral_score_grf"
              ]



# ==============================
# SIMULATION AND OPTIMIZATION CONFIGS (fixed)
arena_cfg = {
    "size": [7., 7.],
    "sand_ground_color": False
}
evolution_cfg = {
    "popsize": POPSIZE,
    "ngen": NGEN,
}
evaluation_cfg = {
    "reward_expr": "xy-distance",
    "cost_expr": "actuator_forces",
    "penalty_expr": None,
    "additive_or_multiplicative": "multiplicative",
    "alpha": 10.0,
    "beta": 1.0,
}

simulation_cfg = {
    "nsteps": NSTEPS_CONTROL,
    "nthread": max_threads,
}

rendering_cfg = {
    "camera": [0, 1],
    "shape": (480, 640),
    "transparent": False,
    "color_contacts": False, # can only be True if Sand Ground Color is true
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

# ==============================
# SETUP EXPERIMENT DIR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NUMERICAL_METRICS_DIR, PLOTS_DIR, RUN_DIR, VIDEOS_DIR = create_experiment_directory(base_dir=BASE_DIR, experiment_id=exp_id)

# store metadata
metadata = {
    "experiment_id": exp_id,
    "seed": seed,
    "timestamp": timestamp,
    "POPSIZE": POPSIZE,
    "NGEN": NGEN,
    "NREPETITIONS": NREPETITIONS,
    "NREP_PER_GENOTYPE": NREP_PER_GENOTYPE,
    "NSTEPS_CONTROL": NSTEPS_CONTROL,
    "SOLVER": SOLVER,
    "CPG_DT": CPG_DT,
    }

write_metadata_dict_to_txt(metadata, os.path.join(RUN_DIR, "metadata.txt"))

create_csv(os.path.join(NUMERICAL_METRICS_DIR, "bs_simulation_results.csv"), csv_header)
# ==============================
# RUN EXPERIMENT
rng = jax.random.PRNGKey(seed)

print("Starting parameter sweeps...")
print(f"method: {method}, ring_size: {ring_size}, arm_size: {arm_size}, weight_coupling: {weight_coupling}")
start_time = time.time()

arm_setup = int(ring_size) * [int(arm_size)]
n_oscillators = 2*sum(arm_setup) + len(arm_setup) # for brittle stars: number of nodes is doubled + 1 per arm

morph_cfg = {
        "num_arms": len(arm_setup),
        "num_segments_per_arm": arm_setup,
        "use_p_control": True
    }


for run_id in range(NREPETITIONS):
    wandb_logger = WandbLogger(
        project="cpg_convergence",
        group=f"{exp_id}",
        run_name=f"{exp_id}_run_{run_id}_{method}_weight_{weight_coupling}_arm_{arm_size}_ring_{ring_size}",
    )
    simulator = BrittleStarSimulator(morph_cfg=morph_cfg, arena_cfg=arena_cfg)

    rng, rng_cpg = jax.random.split(rng)
    cpg = BS_CPG(arm_setup=morph_cfg["num_segments_per_arm"],
                    rng=rng_cpg,
                    method=method,
                    weight_scale=weight_coupling,
                    dt=CPG_DT,
                    solver=SOLVER,
                )
    
    prespecified_csv_entries = {
        "method": method,
        "ring_size": ring_size,
        "arm_size": arm_size,
        "weight_coupling": weight_coupling,
        "run_id": run_id,
        "spectral_gap": cpg.spectral_gap,
        "induced_norm": cpg.induced_norm,
        "n_oscillators": cpg.num_oscillators,
        "n_couplings": cpg.num_couplings,
    }
    
    rng, rng_control_generator = jax.random.split(rng)
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
    
    optimizer.optimize(
        simulator=simulator,
        nsteps=NSTEPS_CONTROL,
        rng=rng_optimize,
        ngen=NGEN,
        nrep_per_genotype=NREP_PER_GENOTYPE,
        evaluation_cfg=evaluation_cfg,
        nthread=max_threads,
        csv_header=csv_header,
        prespecified_csv_entries=prespecified_csv_entries,
        csv_path=os.path.join(NUMERICAL_METRICS_DIR, "bs_simulation_results.csv"),
        wandb_logger=wandb_logger,
    )

    if run_id == 0:
        optimizer.visualize_solution(
            simulator=simulator,
            rendering_cfg=rendering_cfg,
            video_path=os.path.join(VIDEOS_DIR, f"solution_episode_{method}_ring_size_{ring_size}_arm_size_{arm_size}_weight_{weight_coupling}.mp4"),
        )
    cpg.clear()



elapsed = int(time.time() - start_time)
hrs = elapsed // 3600
mins = (elapsed % 3600) // 60
secs = elapsed % 60
print(f"Total time: {hrs:02d}:{mins:02d}:{secs:02d} (HH:MM:SS)")



print("Parameter sweeps completed. Script Finished.")
