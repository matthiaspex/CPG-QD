"""
This module contains default settings for the BSGaits package.
Write (global) constants in all caps, e.g. `SOME_CONSTANT = 42`.
Your not supposed to change them, to make sure traceability of parameters is easier across experiments.
"""
from math import pi
import numpy as np
from PIL import ImageColor


# Brittle Star simulation settings
PHYSICS_TIMESTEP = 0.004 # Brittle star physics time step is default 2ms, but we scale it up for faster simulation
CONTROL_STEPS_PER_PHYSICS_STEP = 10  # Number of control steps per physics step
CONTROL_TIMESTEP = PHYSICS_TIMESTEP * CONTROL_STEPS_PER_PHYSICS_STEP # 0.004 * 10 = 0.04 seconds per control step
FPS = 1/CONTROL_TIMESTEP # 25 FPS for control step of 0.04 seconds, used for video rendering
GAIT_FREQUENCY = 1 # Default gait frequency in Hz

# CPGs
CPG_DEFAULT_DT = CONTROL_TIMESTEP/10  # Default time step for CPGs, needs to be smaller when weight_scale increases (original: 0.04)
CPG_DEFAULT_SOLVER = "rk4"  # Default solver for CPGs (rk4 or Euler)
CPG_DEFAULT_AMPLITUDE_GAIN = 20.0  # Default gain for amplitude modulation
CPG_DEFAULT_OFFSET_GAIN = 20.0  # Default gain for offset modulation
CPG_DEFAULT_WEIGHT_SCALE = 50.0  # Default weight scale for CPG connections (original: 5)
DISABLE_RANDOMNESS = False  # Whether to disable randomness in CPG initialization, disabled if True
CPG_RESET_PHASE_RANGES = (-0.01, 0.01)  # Range for resetting CPG phases during initialization or reset
OMEGA = GAIT_FREQUENCY * 2 * pi  # Default frequency for CPGs (1 Hz converted to radians per second)
                            # OR -> base on video recording of Ophiarachna incrassata?                

# Optimization settings
# Evosax - OpenES
CENTERED_RANK = True  # Use centered rank for fitness shaping
MAXIMIZE_FITNESS = True  # Maximize fitness in optimization

# Evolutionary Optimization clipping and rescaling -> this enforces isotropic optimization landscape for every parameter
PARAM_CLIP_MIN = -1.0  # Minimum value for raw policy parameters sampling (isotropic distribution) (think OpenES and MAPElites)
PARAM_CLIP_MAX =  1.0  # Maximum value for raw policy parameters sampling (isotropic distribution) (think OpenES and MAPElites)

# MAP-Elites
# Emitter
ISO_SIGMA = 0.05 # original: 0.05
LINE_SIGMA = 0.10 # original: 0.10


# CPG Modulation Parameter scaling (clipping is done with PARAM_CLIP_MIN and PARAM_CLIP_MAX)
X_SCALE_MIN = -1.0
X_SCALE_MAX = 1.0
R_SCALE_MIN = 0.0
R_SCALE_MAX = 1.0
OMEGA_SCALE_MIN = 0.0
OMEGA_SCALE_MAX = 5 * 2 * pi  # 5 Hz converted to radians per second

# CPG Output clipping and scaling
CPG_OUTPUT_CLIP_MIN = -1.       # related to the CPG modulation ranges
CPG_OUTPUT_CLIP_MAX = 1.        # related to the CPG modulation ranges
CPG_OUTPUT_SCALE_MIN = -pi / 6  # Scale the CPG output to [-30°, 30°] = [-0.5236 rad, 0.5236 rad]
CPG_OUTPUT_SCALE_MAX = pi / 6   # Scale the CPG output to [-30°, 30°] = [-0.5236 rad, 0.5236 rad]

# Analysis metrics
CONTACT_THRESHOLD = 0.1

# Parallelization (for e.g. evosax)
# Below parameters need to be set BEFORE importing jax!
NTHREADS_MAXIMUM = 16  # Maximum number of threads to use for parallelization
CUDA_VISIBLE_DEVICES = "" # GPU device index to use, Choose "" (no GPU) or "0", "1", ... for specific GPU



# Colors

hex_green = "#7db5a8"
hex_bright_green = "#83f28f"
hex_red = "#b75659"
hex_orange = "#d6ae72"
hex_gray = "#595959"
hex_blue = "#73c2fb"

plt_red_1 = "#fb3000"
plt_orange_1 = "#ffa01c"
plt_green_1 = "#89de00"
plt_red_2 = "#c64f55"
plt_orange_2 = "#f8cb7e"
plt_green_2 = "#6bb8a8"


rgba_green = np.array(ImageColor.getcolor(hex_green, "RGBA")) / 255
rgba_bright_green = np.array(ImageColor.getcolor(hex_bright_green, "RGBA")) / 255
rgba_red = np.array(ImageColor.getcolor(hex_red, "RGBA")) / 255
rgba_orange = np.array(ImageColor.getcolor(hex_orange, "RGBA")) / 255
rgba_gray = np.array(ImageColor.getcolor(hex_gray, "RGBA")) / 255
rgba_blue = np.array(ImageColor.getcolor(hex_blue, "RGBA")) / 255

rgba_tendon_relaxed = np.array([214, 174, 114, 190]) / 255.0
rgba_tendon_contracted = np.array([183, 86, 89, 210]) / 255.0

rgba_sand = np.array([225, 191, 146, 255]) / 255.0