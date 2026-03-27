"""
Generate brittle star mjModel, mjData and other requirements to launch a rollout in Mujoco.
This method does not require anything related to jax
"""

import pprint
from typing import Tuple, Optional, Literal, Union
from xml.parsers.expat import model
import numpy as np
import mujoco
from mujoco import rollout
import mediapy as media
from multiprocessing import cpu_count
import copy
import sys
import os
import xmltodict

# morphology imports
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology

# arena imports
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena

from moojoco.mjcf.component import MJCFRootComponent

from cpg_convergence.defaults import PHYSICS_TIMESTEP, CONTROL_STEPS_PER_PHYSICS_STEP, CONTROL_TIMESTEP, FPS,\
    rgba_green, rgba_red
from cpg_convergence.visualization import show_video, save_video_from_raw_frames
from cpg_convergence.utils import quaternion_to_axis_angle


class BrittleStarSimulator():
    """
    A simulator for the brittle star morphology in Mujoco.
    This simulator generates the mjModel and mjData required to run a simulation
    It provides methods to generate rollout and visualize the simulation.
    """

    def __init__(self, morph_cfg: dict, arena_cfg: dict):
        """
        Initialize the simulator with morphology and arena configurations.
        args:
            morph_cfg (dict): Configuration dictionary for the morphology. Can contain:
                num_arms: int,
                num_segments_per_arm: Union[int, List[int]],
                use_tendons: bool,
                use_p_control: bool,
                use_torque_control: bool,
                radius_to_strength_factor: float,
                num_contact_sensors_per_segment: int,
            arena_cfg (dict): Configuration dictionary for the arena. Can contain:
                name: str,
                size: Tuple[int, int],
                sand_ground_color: bool,
                light_map_resolution: int,
                attach_target: bool,
                wall_height: float,
                wall_thickness: float,
        Initialized attributes:
            morph_cfg (dict): The configuration dictionary for the morphology.
            arena_cfg (dict): The configuration dictionary for the arena.
            morphology (MJCFBrittleStarMorphology): The morphology of the brittle star.
            arena (MJCFAquariumArena): The arena in which the brittle star will be simulated.
            mj_model (mujoco.MjModel): The Mujoco model of the brittle star and arena.
            mj_data (mujoco.MjData): The Mujoco data for the simulation.
        """
        self.morph_cfg = morph_cfg
        self.arena_cfg = arena_cfg

        self.morphology = self._create_morphology(morph_cfg)
        self.arena = self._create_arena(arena_cfg)

        self.mj_model, self.mj_data = self._get_mjModel_and_mjData(self.morphology, self.arena)
        self.mj_model = self._modify_model_timestep(self.mj_model)

    @property
    def control_dim(self) -> int:
        """Returns the control dimension based on the defined mj_model."""
        return self.mj_model.nu
    
    
    @staticmethod
    def _create_morphology(
            morph_cfg: dict
            ) -> MJCFBrittleStarMorphology:
        morphology_specification = default_brittle_star_morphology_specification(**morph_cfg) # unpacking the morph_cfg dictionary
        # morphology specification is an instance of BrittleStarMorphologySpecification
        morphology = MJCFBrittleStarMorphology(
                specification=morphology_specification
                )
        return morphology


    @staticmethod
    def _create_arena(
            arena_cfg: dict
            ) -> MJCFAquariumArena:
        arena_configuration = AquariumArenaConfiguration(**arena_cfg)
        # arena configuration is an instance of AquariumArenaConfiguration
        arena = MJCFAquariumArena(
                configuration=arena_configuration
                )
        return arena
    
    @staticmethod
    def _get_mjModel_and_mjData(
            morphology: MJCFBrittleStarMorphology,
            arena: MJCFAquariumArena,
            ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        
        # model should contain the morphology and arena
        arena.attach(morphology, free_joint=True) # Note: this is in place modification

        xml_extended = BrittleStarSimulator._add_ground_reaction_force_sensors(arena.get_mjcf_str())

        mj_model = mujoco.MjModel.from_xml_string(xml_extended)
        mj_data = mujoco.MjData(mj_model)

        return mj_model, mj_data
    
    @staticmethod
    def _add_ground_reaction_force_sensors(
            xml: str
            ) -> str:
        """
        Add ground reaction force sensors to the xml string of the model.
        """
        model_prior = mujoco.MjModel.from_xml_string(xml)
        mjcf_dict = xmltodict.parse(xml)

        xml_addition = []
        for i in range(model_prior.ngeom):
            geom_name = model_prior.geom(i).name
            if "capsule" in geom_name:
                sensor_name = geom_name.replace("capsule", "ground_reaction_force")

                xml_addition.append({"@name": sensor_name,
                                    "@geom1": "groundplane",
                                    "@geom2": geom_name,
                                    "@data": "force torque",
                                    "@reduce": "netforce"
                                    })

        mjcf_dict['mujoco']['sensor']['contact'] = xml_addition
        return xmltodict.unparse(mjcf_dict)
    
    
    @staticmethod
    def _modify_model_timestep(
            mj_model: mujoco.MjModel
            ) -> mujoco.MjModel:
        """
        Modify the model timestep to defaults.PHYSICS_TIMESTEP
        """
        mj_model.opt.timestep = PHYSICS_TIMESTEP
        return mj_model

    
    @staticmethod
    def _visualize_mjcf(
            mjcf: MJCFRootComponent,
            background_white: bool = False
            ) -> None:
        model = mujoco.MjModel.from_xml_string(mjcf.get_mjcf_str())
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model)

        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
        if background_white:
            # Create a white background by replacing only black pixels
            black_mask = np.all(pixels[:, :, :2] == 0, axis=2)
            pixels[black_mask, :3] = [255, 255, 255]  # Replace RGB channels with white
        media.show_image(pixels)

    def visualize_morphology(self, background_white: bool = False) -> None:
        """
        Visualize the morphology of the brittle star.
        Only displays image in interactive python or jupyter notebook.
        """
        self._visualize_mjcf(self.morphology, background_white=background_white)

    def visualize_arena(self, background_white: bool = False) -> None:
        """
        Visualize the arena in which the brittle star will be simulated.
        Only displays image in interactive python or jupyter notebook.
        """
        self._visualize_mjcf(self.arena, background_white=background_white)


    def export_xml(self, file_path: str) -> None:
        """
        Export the mjcf xml string to a file.
        args:
            file_path (str): The path to the file where the xml string will be saved.
        """
        assert file_path.endswith('.xml'), "File path must end with .xml"

        print(type(self.mj_model))
        mujoco.mj_saveLastXML(file_path, self.mj_model)
        # with open(file_path, 'w') as f:
        #     f.write(self.morphology.get_mjcf_str())
        #     f.write(self.arena.get_mjcf_str())
    

    def rollout(self,
                control: np.ndarray,
                nthread: Optional[int] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        As currently implemented, this method performs a rollout for nbatch parallel simulations.
        mjModel and mjData are identical across all simulations.
        Control inputs differ across simulations.
        Method internally handles the upsampling of control and downsampling of state and sensor data
            to match the physics timestep.
        args:
            control (nbatch x nsteps x ncontrol): open-loop control inputs for the simulation.
            nthread (optional): Number of threads to use for the simulation. If None, uses all available CPU cores.
        returns:
            state: State output array, (nbatch x nsteps x nstate).
            sensordata: Sensor data output array, (nbatch x nsteps x nsensordata).

        Possible future extensions:
            Allow for different mjModel and for each simulation (providing lists of models and data).
                Note: mjData doesn't vary: only initial state varies,
                      mjData just needs to have dimension nthreads to speed up the simulation.
        """
        if isinstance(self.mj_model, list) and len(self.mj_model) != 1:
            raise ValueError("BrittleStarSimulator currently only supports a single mjModel for all rollouts. ")
        
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        control = self._upsample_control(control) # Upsample control to match the physics timestep

        nbatch = control.shape[0]
        nstep = control.shape[1]
        ncontrol = control.shape[2]

        if nthread is None:
            nthread = cpu_count() # Used for mjData generation
        else:
            nthread = min(nthread, cpu_count()) # nthread should not exceed the number of available CPU cores

        assert ncontrol == self.mj_model.nu, \
            f"Control dimension {ncontrol} does not match model control dimension {self.mj_model.nu}."
        
        
        # Set correct inputs to allow skip_checks = True in rollout.rollout
        # e.g., If skip_checks = True, we need to provide mj_model as a list of models
        #       If skip_checks = False, the below code would happen within rollout.rollout
        if isinstance(self.mj_model, mujoco.MjModel):
            models = [self.mj_model]*nbatch
        
        initial_states = self._get_state(self.mj_model, self.mj_data, nbatch=nbatch)

        datas = [copy.copy(self.mj_data) for _ in range(nthread)] # 1 MjData per thread

        state = np.zeros((nbatch, nstep, initial_states.shape[-1]))
        sensordata = np.zeros((nbatch, nstep, self.mj_model.nsensordata))

        state, sensordata = rollout.rollout(model=models,                   # list (nbatch) of mujoco.MjModel
                                            data=datas,                     # list (nthread) of mujoco.MjData
                                            nstep=nstep,
                                            initial_state=initial_states,   # np.ndarray (nbatch x nstate)
                                            control=control,                # np.ndarray (nbatch x nsteps x ncontrol)
                                            skip_checks=True,               # Skip checks for faster simulation
                                            state=state,                    # np.ndarray (nbatch x nsteps x nstate)
                                            sensordata=sensordata,          # np.ndarray (nbatch x nsteps x nsensordata)
                                            )
        
        state, sensordata = self._downsample_state_sensordata(state, sensordata)

        # Note: returning mj_data useless, because you would only get the last mj_data of the last set of threads.
        return state, sensordata
        
    @staticmethod
    def _upsample_control(control: np.ndarray,
                         control_steps_per_physics_step: int = CONTROL_STEPS_PER_PHYSICS_STEP
                    ) -> np.ndarray:
        """
        Upsample control inputs to match the physics timestep.
        args:
            control (nbatch x nsteps_control x ncontrol): Open-loop control inputs for the simulation.
            steps_per_physics_step (int): Number of control steps per physics step.
        returns:
            upsampled_control (nbatch x nsteps_physics x ncontrol): Upsampled control inputs.
        """
        upsampled_control = np.repeat(control, control_steps_per_physics_step, axis=1)
        return upsampled_control
    
    @staticmethod
    def _downsample_state_sensordata(state: np.ndarray,
                                    sensordata: np.ndarray,
                                    control_steps_per_physics_step: int = CONTROL_STEPS_PER_PHYSICS_STEP
                                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Downsample state and sensor data to match the control timestep.
        Keep every CONTROL_STEPS_PER_PHYSICS_STEP-th step. (i.e., the last physics step in each control step)
        args:
            state (nbatch x nsteps_physics x nstate): State data from the simulation.
            sensor_data (nbatch x nsteps_physics x nsensordata): Sensor data from the simulation.
        returns:
            downsampled_state (nbatch x nsteps_control x nstate): Downsampled state data.
            downsampled_sensor_data (nbatch x nsteps_control x nsensordata): Downsampled sensor data.
        """
        downsampled_state = state[:, control_steps_per_physics_step-1::control_steps_per_physics_step, :]
        downsampled_sensor_data = sensordata[:, control_steps_per_physics_step-1::control_steps_per_physics_step, :]
        return downsampled_state, downsampled_sensor_data
    

    @staticmethod
    def _get_state(model, data, nbatch=1):
        full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
        state = np.zeros((mujoco.mj_stateSize(model, full_physics),))
        mujoco.mj_getState(model, data, state, full_physics)
        return np.tile(state, (nbatch, 1))
    

    def visualize_rollout(
            self,
            state: np.ndarray,
            sensordata: np.ndarray,
            show: bool = True,
            path: Optional[str] = None,
            framerate: int = FPS,
            **kwargs,
        ):
        """
        Visualize the rollout by rendering the state and sensor data.
        args:
            state (nbatch x nsteps x nstate): State data from the simulation, with nsteps control steps
            sensordata (nbatch x nsteps x nsensordata): Sensor data from the simulation, with nsteps control steps
            show (bool): Whether to show the visualization in a window.
            path (str, optional): Path to save the video. If None, does not save.
            framerate (int): Framerate for the video. Default is defaults.FPS
        kwargs:
            camera (int): Camera index to use for rendering. Default is 0. Other possibility: 1 (camera on the top of the brittle star)
            shape (Tuple[int, int]): Shape of the rendered frames (height, width). Default is (480, 640). other possibilities: (720, 1280), (1080, 1920)
            transparent (bool): Whether to render with a transparent background. Default is False.
            light_pos (Optional[List[float]]): Position of the light source in the scene.
                If None, no light is added. Default is None.
            color_contacts (bool): Whether to color segments which touch the ground red in visualisation. default is False.
        """
        if not (os.getenv("FORCE_MJC_RENDER_OFF") == "True"):  # if FORCE_MJC_RENDER_OFF is not set or False, render the video. If True, skip rendering
            # Render the frames
            frames = BrittleStarSimulator._render_many(
                model=self.mj_model,
                data=self.mj_data,
                state=state,
                sensordata=sensordata,
                framerate=framerate,
                **kwargs,
            )
            # Show the video
            if show:
                show_video(frames, fps=framerate)
            # Save the video
            if path is not None:
                save_video_from_raw_frames(frames, fps=framerate, file_path=path)


    @staticmethod
    def _render_many(model, data, state, sensordata, framerate=FPS, camera=0, shape=(480, 640),
                    transparent=False, light_pos=None, color_contacts=False):
        nbatch = state.shape[0]

        if not isinstance(model, mujoco.MjModel):
            model = list(model)

        if isinstance(model, list) and len(model) == 1:
            model = model * nbatch
        elif isinstance(model, list):
            assert len(model) == nbatch
        else:
            model = [model] * nbatch

        # ensure camera is a list:
        if isinstance(camera, int):
            camera = [camera]

        # Visual options
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
        pert = mujoco.MjvPerturb()  # Empty MjvPerturb object
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

        map_geom_id_on_touch_sensor_range = {}
        for geom_id in range(model[0].ngeom):
            geom_name = mujoco.mj_id2name(model[0], mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if 'capsule' in geom_name:
                # print(f"geom_id: {geom_id}, geom_name: {geom_name}")
                # Find sensor(s) that match this geom and are of type TOUCH
                for i in range(model[0].nsensor):
                    sensor_type = model[0].sensor_type[i]
                    sensor_type_str = mujoco.mjtSensor(sensor_type).name
                    obj_type = model[0].sensor_objtype[i]
                    obj_id = model[0].sensor_objid[i]
                    obj_name = mujoco.mj_id2name(model[0], obj_type, obj_id)
                    if sensor_type_str == "mjSENS_TOUCH" and 'contact_site' in obj_name:
                        if (geom_name.replace("_capsule", "") == (obj_name.replace("_contact_site", ""))): 
                            # print(f"sensor {i}: type: {sensor_type_str}, obj_name: {obj_name}")
                            adr = model[0].sensor_adr[i]
                            dim = model[0].sensor_dim[i]
                            # value = data.sensordata[adr : adr + dim]
                            map_geom_id_on_touch_sensor_range[geom_id] = range(adr, adr+dim)


        # Simulate and render.
        frames = []
        with mujoco.Renderer(model[0], *shape) as renderer:
            for i in range(state.shape[1]):
                if len(frames) <= round(i * CONTROL_TIMESTEP * framerate): # if CONTROL_TIMESTEP = PHYSICS_TIMESTEP: model[0].opt.timestep
                    pixels = []
                    for cam in camera:
                        for j in range(state.shape[0]):
                            mujoco.mj_setState(model[j], data, state[j, i, :],
                                                mujoco.mjtState.mjSTATE_FULLPHYSICS)
                            mujoco.mj_forward(model[j], data)

                            # Change color of geoms with mjSENS_TOUCH sensor to red
                            if color_contacts:
                                for geom_id, adress_range in map_geom_id_on_touch_sensor_range.items():
                                    touch_value = sensordata[j, i, adress_range]
                                    # print(f"touch_value: {touch_value}, geom_id: {geom_id}")
                                    model[j].geom_rgba[geom_id] = rgba_red if touch_value > 0.1 else rgba_green # 0.1 threshold for numerical stability

                                # for geom_id in range(model[j].ngeom):
                                #     geom_name = mujoco.mj_id2name(model[j], mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                                #     if geom_name and 'capsule' in geom_name:
                                #         model[j].geom_rgba[geom_id] = [1.0, 0.0, 0.0, 1.0]  # Red color


                            # Use first model to make the scene, add subsequent models
                            if j == 0:
                                renderer.update_scene(data, cam, scene_option=vopt)
                            else:
                                mujoco.mjv_addGeoms(model[j], data, vopt, pert, catmask, renderer.scene)

                        # Add light, if requested
                        if light_pos is not None:
                            light = renderer.scene.lights[renderer.scene.nlight]
                            light.ambient = [0, 0, 0]
                            light.attenuation = [1, 0, 0]
                            light.castshadow = 1
                            light.cutoff = 45
                            light.diffuse = [0.8, 0.8, 0.8]
                            light.dir = [0, 0, -1]
                            light.directional = 0
                            light.exponent = 10
                            light.headlight = 0
                            light.specular = [0.3, 0.3, 0.3]
                            light.pos = light_pos
                            renderer.scene.nlight += 1

                        # Render and add the frame.
                        pixels.append(renderer.render())
                    pixels = np.concatenate(pixels, axis=1)  # Concatenate frames from all cameras horizontally
                    frames.append(pixels)
        return frames
    

    def get_fitness(
            self,
            state: np.ndarray,
            sensordata: np.ndarray,
            reward_expr: str = "x-distance",
            cost_expr: Optional[str] = None,
            penalty_expr: Optional[str] = None,
            additive_or_multiplicative: Literal["additive", "multiplicative"] = "multiplicative",
            alpha: float = 1.0,
            beta: float = 1.0
        ) -> np.ndarray:
        """
        Calculate the fitness of the brittle star based on the state and sensordata from the rollout.
        The fitness is calculated using formula:
            additive: fitness = reward - alpha * cost - beta * penalty
            multiplicative: fitness = reward / (1 + alpha * cost + beta * penalty) -> does not explode for very small costs.
        args:
            state (nbatch x nsteps x nstate): State data from the simulation.
            sensordata (nbatch x nsteps x nsensordata): Sensor data from the simulation.
            reward_expr (str): The expression to calculate the reward. Choose from:
                - "x-distance" (default): Fitness is the x-distance travelled by the brittle star.
                - "xy-distance": Fitness is the total distance travelled by the brittle star.
            cost_expr (str, optional): The expression to calculate the cost. If None, no cost is applied:
                - "actuator_forces": Cost is the sum of the absolute torques applied by the actuators.
            penalty_expr (str, optional): The expression to calculate the penalty. If None, no penalty is applied.
                - "xy_disk_rotation": Penalty is the rotation of the central disk in the xy-plane.
                - "z_disk_rotation": Penalty is the rotation of the central disk around the z-axis.
                - "xyz_disk_rotation": Penalty is the rotation of the central disk in all axes.
            additive_or_multiplicative (Literal["additive", "multiplicative"]): Whether to use additive or multiplicative combination of reward and cost.
                default: "multiplicative" (depends less on normalization of reward and cost).
            alpha (float): Weighting factor for the cost in the fitness calculation.
            
        returns:
            fitness (np.ndarray) (nbatch): Fitness value for each batch in the rollout.
            metrics (dict[str, float]): Dictionary containing the metrics, possibly forwarded to wandb.
        """
        sensordict = self.extract_sensor_dict(sensordata)
        metrics = {}

        if reward_expr == "x-distance":
            # compare begin and end position of the central disk
            reward = sensordict['central_disk']['position'][:, -1, 0] - sensordict['central_disk']['position'][:, 0, 0]
        elif reward_expr == "xy-distance":
            # compare begin and end position of the central disk
            x_dist = sensordict['central_disk']['position'][:, -1, 0] - sensordict['central_disk']['position'][:, 0, 0]
            y_dist = sensordict['central_disk']['position'][:, -1, 1] - sensordict['central_disk']['position'][:, 0, 1]
            reward = np.sqrt(x_dist**2 + y_dist**2)
        else:
            raise ValueError(f"Unknown reward expression: {reward_expr}")

        metrics['mean_reward'] = np.mean(reward)
        metrics['max_reward'] = np.max(reward)

        if cost_expr is not None:
            if cost_expr == "actuator_forces":
                # mean of absolute values across all actuators
                cost = np.mean(np.abs(sensordict['ip_joints']['actuator_force']), axis=-1) \
                       + np.mean(np.abs(sensordict['oop_joints']['actuator_force']), axis=-1)
                # mean over time
                cost = np.mean(cost, axis=-1)

            else:
                raise ValueError(f"Unknown cost expression: {cost_expr}")
            
            metrics['mean_cost'] = np.mean(cost) # across all batches
            metrics['min_cost'] = np.min(cost) # across all batches
        else:
            cost = 0

        
        if penalty_expr is not None:
            if penalty_expr == "xy_disk_rotation":
                # sum absolute values across x and y dimension
                penalty = np.abs((sensordict['central_disk']['rotation'][:, :, 0])) \
                          + np.abs((sensordict['central_disk']['rotation'][:, :, 1]))  # x and y rotation treated independently
                # mean over time
                penalty = np.mean(penalty, axis=-1)

            if penalty_expr == "z_disk_rotation":
                # absolute value of the gradient of the z-rotation
                penalty = np.abs(np.gradient(sensordict['central_disk']['rotation'][:, :, 2], axis=-1))  # z rotation
                # mean over time
                penalty = np.mean(penalty, axis=-1)

            if penalty_expr == "xyz_disk_rotation":
                # sum absolute values across x, y and z dimension
                penalty = np.abs((sensordict['central_disk']['rotation'][:, :, 0])) \
                            + np.abs((sensordict['central_disk']['rotation'][:, :, 1])) \
                            + np.abs((sensordict['central_disk']['rotation'][:, :, 2]))  # x, y and z rotation treated independently
                # mean over time
                penalty = np.mean(penalty, axis=-1)

            else:
                raise ValueError(f"Unknown penalty expression: {penalty_expr}")
            
            metrics['mean_penalty'] = np.mean(penalty)
            metrics['min_penalty'] = np.min(penalty)
        else:
            penalty = 0

        if additive_or_multiplicative == "additive":
            fitness = reward - alpha*cost - beta*penalty
        elif additive_or_multiplicative == "multiplicative":
            fitness = reward / (1 + alpha*cost + beta*penalty)
        else:
            raise ValueError(f"Unknown combination method: {additive_or_multiplicative}")

        assert fitness.ndim == 1, "Fitness should be a 1D array (nbatch)."

        metrics['mean_fitness'] = np.mean(fitness)
        metrics['max_fitness'] = np.max(fitness)

        return fitness, metrics

    

    def extract_sensor_dict(self, sensordata) -> dict:
        """
        Extract sensor data from the sensordata array and organize it into a dictionary.
        args:
            sensordata (nbatch x nshape x nsensordata): Sensor data array: provided by rollout.rollout
        returns:
            sensor_dict (dict): dictionary with structure:
            {
                'central_disk': {
                    'position': np.ndarray (nbatch x nshape x 3),
                    'quaternion': np.ndarray (nbatch x nshape x 4), -> [w, x, y, z] quaternion
                    'angvel': np.ndarray (nbatch x nshape x 3),
                    'linvel': np.ndarray (nbatch x nshape x 3),
                    'rotation': np.ndarray (nbatch x nshape x 3),  -> [x, y, z] euler angles in radians
                },
                'segments': {
                    'position': np.ndarray (nbatch x nshape x nsegments x 3),
                    'contact': np.ndarray (nbatch x nshape x nsegments),
                    'ground_reaction_force': np.ndarray (nbatch x nshape x nsegments x 6), -> [fx, fy, fz, tx, ty, tz]
                },
                'ip_joints': {
                    'position': np.ndarray (nbatch x nshape x nsegments),
                    'velocity': np.ndarray (nbatch x nshape x nsegments),
                    'joint_force': np.ndarray (nbatch x nshape x nsegments),
                    'actuator_force': np.ndarray (nbatch x nshape x nsegments),
                },
                'oop_joints': {
                    'position': np.ndarray (nbatch x nshape x nsegments),
                    'velocity': np.ndarray (nbatch x nshape x nsegments),
                    'joint_force': np.ndarray (nbatch x nshape x nsegments),
                    'actuator_force': np.ndarray (nbatch x nshape x nsegments),
                }
            }"""
        nbatch, nshape, nsensordata = sensordata.shape
        shape = (nbatch, nshape, 0)  # Initialize with zero third dimension
        shape_segments_positions = tuple(list(shape) + [3])  # 3D positions
        shape_segments_grf = tuple(list(shape) + [6])  # 6D contact forces

        # Generate a dictionary with empty placeholder arrays.
        # The dimensions allow concatenating along the last axis
        sensor_dict = {
            'central_disk': {
                'position': np.zeros(shape),
                'quaternion': np.zeros(shape),
                'angvel': np.zeros(shape),
                'linvel': np.zeros(shape),
            },
            'segments': {
                'position': np.zeros(shape_segments_positions),
                'contact': np.zeros(shape),
                'ground_reaction_force': np.zeros(shape_segments_grf),  # 6D force and torque
            },
            'ip_joints': {
                'position': np.zeros(shape),
                'velocity': np.zeros(shape),
                'joint_force': np.zeros(shape),
                'actuator_force': np.zeros(shape),
            },
            'oop_joints': {
                'position': np.zeros(shape),
                'velocity': np.zeros(shape),
                'joint_force': np.zeros(shape),
                'actuator_force': np.zeros(shape),
            }
        }

        model = self.mj_model

        for i in range(model.nsensor):
            sensor_type = model.sensor_type[i]
            sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            obj_type = model.sensor_objtype[i]
            obj_id = model.sensor_objid[i]
            start = model.sensor_adr[i]
            dim = model.sensor_dim[i]
            sensor_type_str = mujoco.mjtSensor(sensor_type).name
            obj_type_str = mujoco.mjtObj(obj_type).name
            obj_name = mujoco.mj_id2name(model, obj_type, obj_id)

            if 'central_disk' in obj_name:
                key = 'central_disk'
                if sensor_type_str == 'mjSENS_FRAMEPOS':
                    sensor_dict[key]['position'] = np.concatenate((sensor_dict[key]['position'], sensordata[:, :, start:start+dim]), axis=-1)
                elif sensor_type_str == 'mjSENS_FRAMEQUAT':
                    sensor_dict[key]['quaternion'] = np.concatenate((sensor_dict[key]['quaternion'], sensordata[:, :, start:start+dim]), axis=-1)
                elif sensor_type_str == 'mjSENS_FRAMELINVEL':
                    sensor_dict[key]['linvel'] = np.concatenate((sensor_dict[key]['linvel'], sensordata[:, :, start:start+dim]), axis=-1)
                elif sensor_type_str == 'mjSENS_FRAMEANGVEL':
                    sensor_dict[key]['angvel'] = np.concatenate((sensor_dict[key]['angvel'], sensordata[:, :, start:start+dim]), axis=-1)
            elif 'capsule' in obj_name or 'contact_site' in obj_name or 'groundplane' in obj_name:
                key = 'segments'
                if sensor_type_str == 'mjSENS_FRAMEPOS':
                    sensor_dict[key]['position'] = np.concatenate((sensor_dict[key]['position'], sensordata[:, :, np.newaxis, start:start+dim]), axis=-2)
                elif sensor_type_str == 'mjSENS_TOUCH':
                    sensor_dict[key]['contact'] = np.concatenate((sensor_dict[key]['contact'], sensordata[:, :, start:start+dim]), axis=-1)
                elif sensor_type_str == 'mjSENS_CONTACT':
                    sensor_dict[key]['ground_reaction_force'] = np.concatenate((sensor_dict[key]['ground_reaction_force'], sensordata[:, :, np.newaxis, start:start+dim]), axis=-2)
            elif 'in_plane' in obj_name or 'out_of_plane' in obj_name:
                key = 'oop_joints' if 'out_of_plane' in obj_name else 'ip_joints'
                if sensor_type_str == 'mjSENS_JOINTPOS':
                    sensor_dict[key]['position'] = np.concatenate((sensor_dict[key]['position'], sensordata[:, :, start:start+dim]), axis=-1)
                elif sensor_type_str == 'mjSENS_JOINTVEL':
                    sensor_dict[key]['velocity'] = np.concatenate((sensor_dict[key]['velocity'], sensordata[:, :, start:start+dim]), axis=-1)
                elif sensor_type_str == 'mjSENS_JOINTACTFRC':
                    sensor_dict[key]['joint_force'] = np.concatenate((sensor_dict[key]['joint_force'], sensordata[:, :, start:start+dim]), axis=-1)
                elif sensor_type_str == 'mjSENS_ACTUATORFRC':
                    sensor_dict[key]['actuator_force'] = np.concatenate((sensor_dict[key]['actuator_force'], sensordata[:, :, start:start+dim]), axis=-1)

        # Quaternion conversion
        rotation = np.zeros((sensor_dict["central_disk"]["quaternion"].shape[0],
                             sensor_dict["central_disk"]["quaternion"].shape[1],
                            3))  # Initialize rotation array
        for i in range(sensor_dict["central_disk"]["quaternion"].shape[0]):
            for j in range(sensor_dict["central_disk"]["quaternion"].shape[1]):
                rotation[i, j] = quaternion_to_axis_angle(sensor_dict["central_disk"]["quaternion"][i, j])
        sensor_dict["central_disk"]["rotation"] = rotation

        return sensor_dict




if __name__ == "__main__":
    # create a configuration dictionary
    config = {
        "morphology": {
            "num_arms": 5,
            "num_segments_per_arm": 5 * [7],
            "use_p_control": True
        },
        "arena": {
            "size": [20., 20.],
            "sand_ground_color": False,
        },
        "render": {
            "camera": [1],
            "shape": ( 480, 640 ), #( 480, 640 ), (600 ✕ 800),(768, 1024), (1440, 1920)
            # "transparent": False,
            # "color_contacts": True,
        }
    }


    simulator = BrittleStarSimulator(
        morph_cfg=config["morphology"],
        arena_cfg=config["arena"]
    )
    simulator.visualize_morphology(background_white=True)  # Only works in interactive python or jupyter notebook
    control_dim = simulator.control_dim
    control_step = 2 # 5 seconds of control at 25 FPS

    # random control input for nbatch
    nbatch = 1
    control = np.random.uniform(-30*np.pi/180, +30*np.pi/180, (nbatch, control_step, control_dim))
    print(f"Control shape: {control.shape}")

    # Run the simulation
    state, sensordata = simulator.rollout(control=control)

    print(f"State shape: {state.shape}")
    print(f"Sensor data shape: {sensordata.shape}")

    # Visualize rollout
    simulator.visualize_rollout(
        state=state,
        sensordata=sensordata,
        show=False,         # Only works in interactive python or jupyter notebook
        path="rollout.mp4",
        color_contacts=False,  # Colors segments which touch the ground red in visualisation
        **config["render"]
    )

    # # generate .xml
    # simulator.export_xml("brittle_star.xml")
