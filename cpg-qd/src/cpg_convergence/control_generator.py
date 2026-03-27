from typing import Any, Dict, Optional, List, Tuple
import numpy as np
from evosax import ParameterReshaper
import jax
from jax import numpy as jnp
import chex
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


from cpg_convergence.cpg import BS_CPG
from cpg_convergence.wandb_utils import WandbLogger
from cpg_convergence.utils import clip_and_rescale
from cpg_convergence.defaults import GAIT_FREQUENCY, PARAM_CLIP_MIN, PARAM_CLIP_MAX, \
    CPG_RESET_PHASE_RANGES, DISABLE_RANDOMNESS, OMEGA




class CPGControlGeneratorBS:
    """
    Control generator that uses a Central Pattern Generator (CPG) approach.
    """

    def __init__(
            self,
            cpg: BS_CPG,
            wandb_logger: Optional[WandbLogger] = None
        ):
        self.cpg = cpg
        self.wandb_logger = wandb_logger


    @property
    def parameter_reshaper(self) -> ParameterReshaper:
        return self.cpg.parameter_reshaper
    

    def generate_control_from_genotype(self,
                                         rng: chex.PRNGKey,
                                         genotypes: np.ndarray,
                                         nsteps_control: int
                                    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Generate control signals from given genotypes using a CPG approach.
        Genotypes and controls will have batch dimension

        Args:
            genotypes (np.ndarray) (nbatch, nparams): The genotype representation of the control parameters.
            nsteps_control (int): Number of control steps to generate.

        Returns:
            Dict[str, Any]: A dictionary containing the generated control signals.
            np.ndarray: The clipped genotypes -> Important to report back to the tell method of the optimizer.
        """
        rng, rng_reset = jax.random.split(rng, 2)

        genotypes_clipped = self.cpg.set_modulation_params_from_independent_params(genotypes, returned_clipped_genotypes=True)

        self.cpg.reset_state(rng_reset, cpg_reset_phase_ranges=CPG_RESET_PHASE_RANGES, disable_randomness=DISABLE_RANDOMNESS)
        self.cpg.reset_control()
        self.cpg.reset_phases()
        self.cpg.modulate_state()

        cpg_to_control_steps_ratio = int(self.cpg.control_timestep / self.cpg.cpg_timestep)
        self.cpg.step_state_n_times(nsteps=int(nsteps_control * cpg_to_control_steps_ratio))

        control = self.cpg.control_for_simulator

        modulation_params = self.cpg.modulation_params  # Store for potential later use

        if self.wandb_logger is not None:
            self.wandb_logger.log_histogram(histogram_name="Offset X modulation distribution", data=modulation_params["X"])
            self.wandb_logger.log_histogram(histogram_name="Amplitude R modulation distribution", data=modulation_params["R"])
            self.wandb_logger.log_histogram(histogram_name="Omega modulation distribution", data=modulation_params["omegas"])
            # self.wandb_logger.log_heatmap(heatmap_name="Phase biases modulation heatmap", data=modulation_params["rhos"][0])


        return control, genotypes_clipped


if __name__ == "__main__":
    import jax
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    rng = jax.random.PRNGKey(42)
    # # Test CPGControlGenerator
    # print("Testing CPGControlGenerator...")
    nbatch = 8
    nsteps_control = 135
    arm_setup = [5] * 3
    rng, rng_cpg_init, rng_control = jax.random.split(rng, 3)

    cpg = BS_CPG(arm_setup=arm_setup,
                rng=rng_cpg_init,
                dt=0.004,
                solver="rk4",
                weight_scale=50,
                omega=OMEGA,
                method="base"
            )
    
    control_generator = CPGControlGeneratorBS(
        cpg = cpg,
        wandb_logger= None
    )

    # print("Finished testing CPGControlGenerator.")

    print(f"cpg.parameter_reshaper.total_params: {cpg.parameter_reshaper.total_params}")
    print(f"number of oscillators: {cpg.num_oscillators}")

    genotypes_test = np.random.normal(loc=0, scale=1, size=(nbatch, cpg.parameter_reshaper.total_params))
    print("genotypes shape: ", genotypes_test.shape)
    print("genotypes minimum and maximum values: ", np.min(genotypes_test), np.max(genotypes_test))

    rng, rng_control_generator = jax.random.split(rng, 2)
    control, genotype_clipped_flat = control_generator.generate_control_from_genotype(
        rng=rng_control_generator,
        genotypes=genotypes_test,
        nsteps_control=nsteps_control
    )

    print("genotypes clipped shape: ", genotype_clipped_flat.shape)
    print("genotypes clipped minimum and maximum values: ", np.min(genotype_clipped_flat), np.max(genotype_clipped_flat))

    print("control shape: ", control.shape)


    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    fig.suptitle("Generated Control Signal Examples", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < control.shape[1]:  # Ensure we don't exceed the number of joints
            ax.plot(control[0, :, i], label=f"Joint {i}")
            ax.set_title(f"Joint {i}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Joint Angle (rad)")
            ax.legend()
        else:
            ax.axis('off')  # Turn off unused subplots

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.savefig("control_generator_test.png")
    plt.show() # will not work on headless server



