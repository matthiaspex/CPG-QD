"""Evaluate Convergence of CPGs with random parameters without simulations."""
# ==================== Computational resources setup ====================
# Always keep sufficient cores open for others to use (try to not go above 50% of available cores)
# Needs to be done before system imports that use threading (e.g., numpy, jax)
from multiprocessing import cpu_count
import os
import time


CUDA_VISIBLE_DEVICES = "0"  # e.g., "0", "1", ... for specific GPU, "" to use CPU only
NTHREADS_MAXIMUM = 16 # set an upper limit to avoid overloading systems with many cores


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
import os
from datetime import datetime
from pprint import pprint
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import jax
from jax import numpy as jnp

from cpg_convergence.experiment_utils.cpg import generate_ring_setup_variations, create_csv, add_csv_entry, \
    create_experiment_directory, write_metadata_dict_to_txt
from cpg_convergence.cpg import run_one_config_with_random_modulation
from cpg_convergence.defaults import CPG_DEFAULT_DT, CPG_DEFAULT_SOLVER

# ==============================
# CONFIGURABLES
exp_id = "b01_r06"
seed = 42

NBATCH = 100
NBATCH_SUBDIVISIONS = 1  # 1 default and minimum to avoid OOM issues, subdivide NBATCH into smaller batches
NSTEPS = 200 # 200 default
SOLVER = CPG_DEFAULT_SOLVER
CPG_DT = CPG_DEFAULT_DT
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


csv_header = ["method", "ring_size", "arm_size", "weight_coupling", "run_id",\
    "n_oscillators", "n_couplings", "spectral_gap", "induced_norm", \
    "step_conv_p50", "step_conv_p75", "step_conv_p90", "step_conv_p100", "fraction_not_converged"]

# parameter sweeps
method_list = ["base","cobweb", "fully_connected"] # 3 variations
ring_size_list = list(jnp.arange(3, 11, 1)) # 3 4 5 ... 10 # 8 variations
arm_size_list = list(jnp.arange(0, 11, 1)) # 0 1 2 ... 10 # 11 variations 
weight_coupling_list = [0.05, 0.5, 5, 25, 50, 100, 500] # 7 variations

# morphology_list = [[1, 0, 0], [1, 1, 0],\
#                    [2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1]\
#                    [3, 0, 0], [3, 1, 0], [3, 1, 1], [3, 2, 0], [3, 2, 1], [3, 2, 2], [3, 3, 0], [3, 3, 1], [3, 3, 2]]

# old variations
# method_list = ["base", "cobweb", "fully_connected", "leader_follower", "modified_de", "popularity"]
# morphology_type_list = ["ring", "equal_arms", "varying_arms"]
# morphology_parameter_lists = {
#     "ring": list(jnp.arange(3, 54, 5)), # 3 8 13 ... 53 # 11 variations
#     "equal_arms": list(jnp.arange(1, 21, 2)), # 1 3 5 ... 19 # 10 variations
#     "varying_arms": list(jnp.arange(0.05, 1.05, 0.05)) # 0.05 0.1 ... 1.0 # 20 variations
# }


# ==============================
# SETUP EXPERIMENT DIR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NUMERICAL_METRICS_DIR, PLOTS_DIR, RUN_DIR, VIDEOS_DIR = create_experiment_directory(base_dir=BASE_DIR, experiment_id=exp_id)

# store metadata
metadata = {
    "experiment_id": exp_id,
    "seed": seed,
    "timestamp": timestamp,
    "NBATCH": NBATCH,
    "NSTEPS": NSTEPS,
    "SOLVER": SOLVER,
    "CPG_DT": CPG_DT,
    "method_list": method_list,
    "ring_size_list": ring_size_list,
    "arm_size_list": arm_size_list,
    "weight_coupling_list": weight_coupling_list,
    "NBATCH_SUBDIVISIONS": NBATCH_SUBDIVISIONS,
    }

write_metadata_dict_to_txt(metadata, os.path.join(RUN_DIR, "metadata.txt"))

create_csv(os.path.join(NUMERICAL_METRICS_DIR, "convergence_results.csv"), csv_header)
# ==============================
# PARAMETER SWEEPS
rng = jax.random.PRNGKey(seed)

print("Starting parameter sweeps...")
for method in method_list:
    time_method_start = time.time()
    print(f"Starting methodology: {method} ...")
    for ring_size in ring_size_list:
        print(f"New ring size: {ring_size}")
        time_ring_size_start = time.time()

        for arm_size in arm_size_list:
            ring_setup = int(ring_size) * [int(arm_size)]
            n_oscillators = sum(ring_setup) + len(ring_setup)
            print(f"\tNew arm_size: {arm_size}")
            time_arm_size_start = time.time()

            for weight_coupling in weight_coupling_list:
                time_weight_coupling_start = time.time()
                
                # further divide in smaller batches of size 100 to avoid OOM issues
                for sub_batch in range(NBATCH_SUBDIVISIONS):
                    sub_batch_size = NBATCH // NBATCH_SUBDIVISIONS
                    rng, run_rng = jax.random.split(rng)
                    steps_to_convergence_p50, steps_to_convergence_p75, steps_to_convergence_p90, steps_to_convergence_p100,\
                        fraction_not_converged, spectral_gap, induced_norm, n_couplings = run_one_config_with_random_modulation(
                            rng=run_rng,
                            ring_setup=ring_setup,
                            method=method,
                            nbatch=sub_batch_size,
                            nsteps=NSTEPS,
                            weight_coupling=weight_coupling,
                            dt=CPG_DT,
                            solver=SOLVER,
                            )
                    b_offset = sub_batch * sub_batch_size
                    for b_incr in range(sub_batch_size):
                        b = b_offset + b_incr
                        steps_to_convergence_b_p50 = int(steps_to_convergence_p50[b])
                        steps_to_convergence_b_p75 = int(steps_to_convergence_p75[b])
                        steps_to_convergence_b_p90 = int(steps_to_convergence_p90[b])
                        steps_to_convergence_b_p100 = int(steps_to_convergence_p100[b])
                        fraction_not_converged_b = round(float(fraction_not_converged[b]), 4)
                        # store results in csv
                        entry = {
                            "method": method,
                            "ring_size": ring_size,
                            "arm_size": arm_size,
                            "weight_coupling": weight_coupling,
                            "run_id": b,
                            "n_oscillators": n_oscillators,
                            "n_couplings": n_couplings,
                            "spectral_gap": spectral_gap,
                            "induced_norm": induced_norm,
                            "step_conv_p50": steps_to_convergence_b_p50,
                            "step_conv_p75": steps_to_convergence_b_p75,
                            "step_conv_p90": steps_to_convergence_b_p90,
                            "step_conv_p100": steps_to_convergence_b_p100,
                            "fraction_not_converged": fraction_not_converged_b,
                        }
                        add_csv_entry(os.path.join(NUMERICAL_METRICS_DIR, "convergence_results.csv"), entry)
                time_weight_coupling_end = time.time()
                print(f"\t\tWeight coupling: {weight_coupling} completed in {time_weight_coupling_end - time_weight_coupling_start:.2f} seconds.")
            time_arm_size_end = time.time()
            print(f"\tFinished arm_size: {arm_size} in {time_arm_size_end - time_arm_size_start:.2f} seconds.")

        time_ring_size_end = time.time()
        print(f"Finished ring_size: {ring_size} in {time_ring_size_end - time_ring_size_start:.2f} seconds.")

    time_method_end = time.time()
    print(f"Finished methodology: {method} in {time_method_end - time_method_start:.2f} seconds.")

print("Parameter sweeps completed. Script Finished.")
