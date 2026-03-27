"""Evaluate Convergence of CPGs with random parameters without simulations.
Now only vary:
- morphology size (ring size and arm size will be equal)
- ratio of n_couplings to n_oscillators
- coupling weight
Don't use the strict methodologies anymore."""
# ==================== Computational resources setup ====================
# Always keep sufficient cores open for others to use (try to not go above 50% of available cores)
# Needs to be done before system imports that use threading (e.g., numpy, jax)
from multiprocessing import cpu_count
import os
import time


CUDA_VISIBLE_DEVICES = ""  # e.g., "0", "1", ... for specific GPU, "" to use CPU only
NTHREADS_MAXIMUM = 20  # set an upper limit to avoid overloading systems with many cores


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
exp_id = "b04_r05"
seed = 42

NBATCH = 100
NBATCH_SUBDIVISIONS = None  # to avoid OOM issues, subdivide NBATCH into smaller batches -> Set automatically during runtime
NSTEPS = 200
SOLVER = CPG_DEFAULT_SOLVER
CPG_DT = CPG_DEFAULT_DT
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


csv_header = ["morphology_size", "ratio_couplings_oscillators", "weight_coupling", "run_id",\
    "n_oscillators", "n_couplings", "spectral_gap", "induced_norm", \
    "step_conv_p50", "step_conv_p75", "step_conv_p90", "step_conv_p100", "fraction_not_converged"]

# parameter sweeps
method = "ratio_couplings_oscillators"
# ratio_couplings_oscillators determined at rollout (while loop)
# morphology_size_list = list(jnp.arange(3, 28, 1)) # 25 variations
morphology_size_list = list(jnp.arange(26, 28, 1))  # tmp
weight_coupling_list = [0.05, 0.5, 5, 50, 500] # 5 variations

# ==============================
# SETUP EXPERIMENT DIR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NUMERICAL_METRICS_DIR, PLOTS_DIR, RUN_DIR, VIDEOS_DIR = create_experiment_directory(base_dir=BASE_DIR, experiment_id=exp_id)

# store metadata
metadata = {
    "method": method,
    "morphology_size_list": morphology_size_list,
    "weight_coupling_list": weight_coupling_list,
    "experiment_id": exp_id,
    "seed": seed,
    "timestamp": timestamp,
    "NBATCH": NBATCH,
    "NSTEPS": NSTEPS,
    "SOLVER": SOLVER,
    "CPG_DT": CPG_DT,
    }

write_metadata_dict_to_txt(metadata, os.path.join(RUN_DIR, "metadata.txt"))

create_csv(os.path.join(NUMERICAL_METRICS_DIR, "convergence_results.csv"), csv_header)
# ==============================
# PARAMETER SWEEPS
rng = jax.random.PRNGKey(seed)

print("Starting parameter sweeps...")

for morphology_size in morphology_size_list:
    morphology_size = int(morphology_size)
    if NBATCH_SUBDIVISIONS is None:
        # set automatically to avoid OOM issues
        if morphology_size <=15:
            NBATCH_SUBDIVISIONS = 1
        elif morphology_size <=20:
            NBATCH_SUBDIVISIONS = 5
        elif morphology_size <=25:
            NBATCH_SUBDIVISIONS = 10
        else:
            NBATCH_SUBDIVISIONS = 20
    print(f"New morphology size: {morphology_size}, NBATCH_SUBDIVISIONS: {NBATCH_SUBDIVISIONS}")
    time_morphology_size_start = time.time()
    if morphology_size < 3:
        ring_setup = 3*[morphology_size]
        print(f"\tUsing minimum morphology size of 3 for ring setup: {ring_setup}")
    else:
        ring_setup = morphology_size * [morphology_size]

    n_oscillators = len(ring_setup) + sum(ring_setup)
    print(f"morphology_size: {morphology_size}, n_oscillators: {n_oscillators}")

    ratio_couplings_oscillators = []
    power = 0
    ratio = 1
    while ratio < (n_oscillators-1)/2:
        ratio_couplings_oscillators.append(ratio)
        power += 1
        ratio = 2 ** power
    ratio_couplings_oscillators.append("max")  # also add max option/fully_connected
    print(f"\tRatio of n_couplings to n_oscillators to be tested: {ratio_couplings_oscillators}")
    

    for ratio in ratio_couplings_oscillators:
        print(f"\tNew ratio: {ratio}")
        time_ratio_start = time.time()

        for weight_coupling in weight_coupling_list:
            time_weight_coupling_start = time.time()
            
            # further divide in smaller batches of size 100 to avoid OOM issues
            for sub_batch in range(NBATCH_SUBDIVISIONS):
                assert NBATCH % NBATCH_SUBDIVISIONS == 0, "NBATCH must be divisible by NBATCH_SUBDIVISIONS"
                sub_batch_size = NBATCH // NBATCH_SUBDIVISIONS
                rng, run_rng = jax.random.split(rng)
                steps_to_convergence_p50, steps_to_convergence_p75, steps_to_convergence_p90, steps_to_convergence_p100,\
                    fraction_not_converged, spectral_gap, induced_norm, n_couplings = run_one_config_with_random_modulation(
                        rng=run_rng,
                        ring_setup=ring_setup,
                        method=method,
                        ratio_couplings_oscillators=ratio,
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
                        "morphology_size": morphology_size,
                        "ratio_couplings_oscillators": ratio,
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
        
        time_ratio_end = time.time()
        print(f"\tFinished ratio: {ratio} in {time_ratio_end - time_ratio_start:.2f} seconds.")

    time_morphology_size_end = time.time()
    print(f"Finished morphology_size: {morphology_size} in {time_morphology_size_end - time_morphology_size_start:.2f} seconds.")


print("Parameter sweeps completed. Script Finished.")
