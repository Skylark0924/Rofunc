#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

import copy
import os
from typing import Dict

import hydra
from hydra import compose, initialize
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_automatic_config_search_path
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig

import rofunc as rf
from rofunc.config import *
from rofunc.utils.oslab.path import get_rofunc_path

"""
Config loading rules:
1. If config_path and config_name are both None, load the default config file (Depreciated, designed for LQT)
2. Configs for RL/IL contains two parts: task and train
    - With the config_path and config_name, the config file `config.yaml` will be loaded and some params will be 
      rewritten by the args passed in.
    - args[0].split('=')[1] is the task name, args[1].split('=')[1] is the train name
"""


def get_sim_config(sim_name: str):
    """
    Load the configs stored in agent_name.yaml.

    :param sim_name: Name of the config file under config/simulator for the agent.
    :return: A dict of configs.
    """
    agent_config_file_path = os.path.join(os.path.dirname(__file__), f"simulator/{sim_name}.yaml")
    if not os.path.exists(agent_config_file_path):
        raise FileNotFoundError(f"{agent_config_file_path} does not exist")

    return OmegaConf.load(agent_config_file_path)


def get_config(config_path=None, config_name=None, args=None, debug=False,
               absl_config_path=None) -> DictConfig:
    """
    Load config file for RofuncRL and rewrite some params by args.

    :param config_path: relative path to the config file (only for rofunc package)
    :param config_name: name of the config file (without .yaml)
    :param args: custom args to rewrite some params in the config file
    :param debug: if True, print the config
    :param absl_config_path: absolute path to the folder contains config file (for external user)
    :return:
    """
    # reset current hydra config if already parsed (but not passed in here)
    if HydraConfig.initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    if (config_path is not None and config_name is not None) or (
            absl_config_path is not None and config_name is not None
    ):
        assert None in [
            config_path,
            absl_config_path,
        ], "config_path and absl_config_path cannot be set simultaneously"

        if args is None:
            with initialize(config_path=config_path, version_base=None):
                cfg = compose(config_name=config_name)
        else:
            if absl_config_path is None:
                rofunc_path = get_rofunc_path()
                absl_config_path = os.path.join(rofunc_path, "config/{}".format(config_path))
            search_path = create_automatic_config_search_path(config_name, None, absl_config_path)
            hydra_object = Hydra.create_main_hydra2(task_name='load_isaacgymenv', config_search_path=search_path)

            # Find the available task and train config files
            try:
                cfg = hydra_object.compose_config(config_name, args, run_mode=RunMode.RUN)
                rf.logger.beauty_print('Use task config: {}.yaml'.format(args[0].split('=')[1]), type='info')
                rf.logger.beauty_print('Use train config: {}.yaml'.format(args[1].split('=')[1]), type='info')
            except Exception as e:
                rf.logger.beauty_print(e, type='warning')
                original_task = args[0].split('=')[1]
                if args[0].split('=')[1].split('_')[0] == 'Gym':
                    task = 'GymBaseTask'
                    args[0] = 'task={}'.format(task)
                    rf.logger.beauty_print('Use task config: {}.yaml'.format(task), type='warning')
                try:
                    cfg = hydra_object.compose_config(config_name, args, run_mode=RunMode.RUN)
                    rf.logger.beauty_print('Use train config: {}.yaml'.format(args[1].split('=')[1]),
                                           type='info')
                except Exception as e:
                    rf.logger.beauty_print(e, type='warning')
                    train = 'BaseTask' + args[1].split('=')[1].split(original_task)[1]
                    args[1] = 'train={}'.format(train)
                    cfg = hydra_object.compose_config(config_name, args, run_mode=RunMode.RUN)
                    rf.logger.beauty_print('Use train config: {}.yaml'.format(train), type='warning')
    else:
        with initialize(config_path="./", version_base=None):
            cfg = compose(config_name="lqt")
    if debug:
        print_config(cfg)
    return cfg


def process_omni_config(cfg):
    """
    Load the configs stored in agent_name.yaml.

    :param cfg:
    :return:
    """

    import carb
    import omni.usd

    default_physx_params = {
        ### Per-scene settings
        "use_gpu": False,
        "worker_thread_count": 4,
        "solver_type": 1,  # 0: PGS, 1:TGS
        "bounce_threshold_velocity": 0.2,
        "friction_offset_threshold": 0.04,  # A threshold of contact separation distance used to decide if a contact
        # point will experience friction forces.
        "friction_correlation_distance": 0.025,  # Contact points can be merged into a single friction anchor if the
        # distance between the contacts is smaller than correlation distance.
        # disabling these can be useful for debugging
        "enable_sleeping": True,
        "enable_stabilization": True,

        # GPU buffers
        "gpu_max_rigid_contact_count": 512 * 1024,
        "gpu_max_rigid_patch_count": 80 * 1024,
        "gpu_found_lost_pairs_capacity": 1024,
        "gpu_found_lost_aggregate_pairs_capacity": 1024,
        "gpu_total_aggregate_pairs_capacity": 1024,
        "gpu_max_soft_body_contacts": 1024 * 1024,
        "gpu_max_particle_contacts": 1024 * 1024,
        "gpu_heap_capacity": 64 * 1024 * 1024,
        "gpu_temp_buffer_capacity": 16 * 1024 * 1024,
        "gpu_max_num_partitions": 8,

        ### Per-actor settings ( can override in actor_options )
        "solver_position_iteration_count": 4,
        "solver_velocity_iteration_count": 1,
        "sleep_threshold": 0.0,  # Mass-normalized kinetic energy threshold below which an actor may go to sleep.
        # Allowed range [0, max_float).
        "stabilization_threshold": 0.0,  # Mass-normalized kinetic energy threshold below which an actor may
        # participate in stabilization. Allowed range [0, max_float).

        ### Per-body settings ( can override in actor_options )
        "enable_gyroscopic_forces": False,
        "density": 1000.0,  # density to be used for bodies that do not specify mass or density
        "max_depenetration_velocity": 100.0,

        ### Per-shape settings ( can override in actor_options )
        "contact_offset": 0.02,
        "rest_offset": 0.001
    }
    default_physics_material = {
        "static_friction": 1.0,
        "dynamic_friction": 1.0,
        "restitution": 0.0
    }
    default_sim_params = {
        "gravity": [0.0, 0.0, -9.81],
        "dt": 1.0 / 60.0,
        "substeps": 1,
        "use_gpu_pipeline": True,
        "add_ground_plane": True,
        "add_distant_light": True,
        "use_flatcache": True,
        "enable_scene_query_support": False,
        "enable_cameras": False,
        "disable_contact_processing": False,
        "default_physics_material": default_physics_material
    }
    default_actor_options = {
        # -1 means use authored value from USD or default values from default_sim_params if not explicitly authored in USD.
        # If an attribute value is not explicitly authored in USD, add one with the value given here,
        # which overrides the USD default.
        "override_usd_defaults": False,
        "make_kinematic": -1,
        "enable_self_collisions": -1,
        "enable_gyroscopic_forces": -1,
        "solver_position_iteration_count": -1,
        "solver_velocity_iteration_count": -1,
        "sleep_threshold": -1,
        "stabilization_threshold": -1,
        "max_depenetration_velocity": -1,
        "density": -1,
        "mass": -1,
        "contact_offset": -1,
        "rest_offset": -1
    }

    class OmniConfig:
        def __init__(self, config: dict = None):
            if config is None:
                config = dict()

            self._config = config
            self._cfg = config.get("task", dict())
            self._parse_config()

            if self._config["inference"]:
                self._sim_params["enable_scene_query_support"] = True

            from omni.isaac.core.utils.extensions import enable_extension
            if self._config["headless"] == True and not self._sim_params["enable_cameras"] and not self._config[
                "enable_livestream"]:
                self._sim_params["use_flatcache"] = False
                self._sim_params["enable_viewport"] = False
            else:
                self._sim_params["enable_viewport"] = True
                enable_extension("omni.kit.viewport.bundle")
            enable_extension("omni.replicator.isaac")

            if self._sim_params["disable_contact_processing"]:
                carb.settings.get_settings().set_bool("/physics/disableContactProcessing", True)

            carb.settings.get_settings().set_bool("/physics/physxDispatcher", True)

        def _parse_config(self):
            # general sim parameter
            self._sim_params = copy.deepcopy(default_sim_params)
            self._default_physics_material = copy.deepcopy(default_physics_material)
            sim_cfg = self._cfg.get("sim", None)
            if sim_cfg is not None:
                for opt in sim_cfg.keys():
                    if opt in self._sim_params:
                        if opt == "default_physics_material":
                            for material_opt in sim_cfg[opt]:
                                self._default_physics_material[material_opt] = sim_cfg[opt][material_opt]
                        else:
                            self._sim_params[opt] = sim_cfg[opt]
                    else:
                        print("Sim params does not have attribute: ", opt)
            self._sim_params["default_physics_material"] = self._default_physics_material

            # physx parameters
            self._physx_params = copy.deepcopy(default_physx_params)
            if sim_cfg is not None and "physx" in sim_cfg:
                for opt in sim_cfg["physx"].keys():
                    if opt in self._physx_params:
                        self._physx_params[opt] = sim_cfg["physx"][opt]
                    else:
                        print("Physx sim params does not have attribute: ", opt)

            self._sanitize_device()

        def _sanitize_device(self):
            if self._sim_params["use_gpu_pipeline"]:
                self._physx_params["use_gpu"] = True

            # device should be in sync with pipeline
            if self._sim_params["use_gpu_pipeline"]:
                self._config["sim_device"] = f"cuda:{self._config['device_id']}"
            else:
                self._config["sim_device"] = "cpu"

            # also write to physics params for setting sim device
            self._physx_params["sim_device"] = self._config["sim_device"]

            print("Pipeline: ", "GPU" if self._sim_params["use_gpu_pipeline"] else "CPU")
            print("Pipeline Device: ", self._config["sim_device"])
            print("Sim Device: ", "GPU" if self._physx_params["use_gpu"] else "CPU")

        def parse_actor_config(self, actor_name):
            actor_params = copy.deepcopy(default_actor_options)
            if "sim" in self._cfg and actor_name in self._cfg["sim"]:
                actor_cfg = self._cfg["sim"][actor_name]
                for opt in actor_cfg.keys():
                    if actor_cfg[opt] != -1 and opt in actor_params:
                        actor_params[opt] = actor_cfg[opt]
                    elif opt not in actor_params:
                        print("Actor params does not have attribute: ", opt)

            return actor_params

        def _get_actor_config_value(self, actor_name, attribute_name, attribute=None):
            actor_params = self.parse_actor_config(actor_name)

            if attribute is not None:
                if attribute_name not in actor_params:
                    return attribute.Get()

                if actor_params[attribute_name] != -1:
                    return actor_params[attribute_name]
                elif actor_params["override_usd_defaults"] and not attribute.IsAuthored():
                    return self._physx_params[attribute_name]
            else:
                if actor_params[attribute_name] != -1:
                    return actor_params[attribute_name]

        @property
        def sim_params(self):
            return self._sim_params

        @property
        def config(self):
            return self._config

        @property
        def task_config(self):
            return self._cfg

        @property
        def physx_params(self):
            return self._physx_params

        def get_physics_params(self):
            return {**self.sim_params, **self.physx_params}

        def _get_physx_collision_api(self, prim):
            from pxr import PhysxSchema
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not physx_collision_api:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            return physx_collision_api

        def _get_physx_rigid_body_api(self, prim):
            from pxr import PhysxSchema
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not physx_rb_api:
                physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            return physx_rb_api

        def _get_physx_articulation_api(self, prim):
            from pxr import PhysxSchema
            arti_api = PhysxSchema.PhysxArticulationAPI(prim)
            if not arti_api:
                arti_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            return arti_api

        def set_contact_offset(self, name, prim, value=None):
            physx_collision_api = self._get_physx_collision_api(prim)
            contact_offset = physx_collision_api.GetContactOffsetAttr()
            # if not contact_offset:
            #     contact_offset = physx_collision_api.CreateContactOffsetAttr()
            if value is None:
                value = self._get_actor_config_value(name, "contact_offset", contact_offset)
            if value != -1:
                contact_offset.Set(value)

        def set_rest_offset(self, name, prim, value=None):
            physx_collision_api = self._get_physx_collision_api(prim)
            rest_offset = physx_collision_api.GetRestOffsetAttr()
            # if not rest_offset:
            #     rest_offset = physx_collision_api.CreateRestOffsetAttr()
            if value is None:
                value = self._get_actor_config_value(name, "rest_offset", rest_offset)
            if value != -1:
                rest_offset.Set(value)

        def set_position_iteration(self, name, prim, value=None):
            physx_rb_api = self._get_physx_rigid_body_api(prim)
            solver_position_iteration_count = physx_rb_api.GetSolverPositionIterationCountAttr()
            if value is None:
                value = self._get_actor_config_value(name, "solver_position_iteration_count",
                                                     solver_position_iteration_count)
            if value != -1:
                solver_position_iteration_count.Set(value)

        def set_velocity_iteration(self, name, prim, value=None):
            physx_rb_api = self._get_physx_rigid_body_api(prim)
            solver_velocity_iteration_count = physx_rb_api.GetSolverVelocityIterationCountAttr()
            if value is None:
                value = self._get_actor_config_value(name, "solver_velocity_iteration_count",
                                                     solver_position_iteration_count)
            if value != -1:
                solver_velocity_iteration_count.Set(value)

        def set_max_depenetration_velocity(self, name, prim, value=None):
            physx_rb_api = self._get_physx_rigid_body_api(prim)
            max_depenetration_velocity = physx_rb_api.GetMaxDepenetrationVelocityAttr()
            if value is None:
                value = self._get_actor_config_value(name, "max_depenetration_velocity", max_depenetration_velocity)
            if value != -1:
                max_depenetration_velocity.Set(value)

        def set_sleep_threshold(self, name, prim, value=None):
            physx_rb_api = self._get_physx_rigid_body_api(prim)
            sleep_threshold = physx_rb_api.GetSleepThresholdAttr()
            if value is None:
                value = self._get_actor_config_value(name, "sleep_threshold", sleep_threshold)
            if value != -1:
                sleep_threshold.Set(value)

        def set_stabilization_threshold(self, name, prim, value=None):
            physx_rb_api = self._get_physx_rigid_body_api(prim)
            stabilization_threshold = physx_rb_api.GetStabilizationThresholdAttr()
            if value is None:
                value = self._get_actor_config_value(name, "stabilization_threshold", stabilization_threshold)
            if value != -1:
                stabilization_threshold.Set(value)

        def set_gyroscopic_forces(self, name, prim, value=None):
            physx_rb_api = self._get_physx_rigid_body_api(prim)
            enable_gyroscopic_forces = physx_rb_api.GetEnableGyroscopicForcesAttr()
            if value is None:
                value = self._get_actor_config_value(name, "enable_gyroscopic_forces", enable_gyroscopic_forces)
            if value != -1:
                enable_gyroscopic_forces.Set(value)

        def set_density(self, name, prim, value=None):
            physx_rb_api = self._get_physx_rigid_body_api(prim)
            density = physx_rb_api.GetDensityAttr()
            if value is None:
                value = self._get_actor_config_value(name, "density", density)
            if value != -1:
                density.Set(value)
                # auto-compute mass
                self.set_mass(prim, 0.0)

        def set_mass(self, name, prim, value=None):
            physx_rb_api = self._get_physx_rigid_body_api(prim)
            mass = physx_rb_api.GetMassAttr()
            if value is None:
                value = self._get_actor_config_value(name, "mass", mass)
            if value != -1:
                mass.Set(value)

        def retain_acceleration(self, prim):
            # retain accelerations if running with more than one substep
            physx_rb_api = self._get_physx_rigid_body_api(prim)
            if self._sim_params["substeps"] > 1:
                physx_rb_api.GetRetainAccelerationsAttr().Set(True)

        def make_kinematic(self, name, prim, cfg, value=None):
            # make rigid body kinematic (fixed base and no collision)
            from pxr import UsdPhysics
            stage = omni.usd.get_context().get_stage()
            if value is None:
                value = self._get_actor_config_value(name, "make_kinematic")
            if value:
                # parse through all children prims
                prims = [prim]
                while len(prims) > 0:
                    cur_prim = prims.pop(0)
                    rb = UsdPhysics.RigidBodyAPI.Get(stage, cur_prim.GetPath())

                    if rb:
                        rb.CreateKinematicEnabledAttr().Set(True)

                    children_prims = cur_prim.GetPrim().GetChildren()
                    prims = prims + children_prims

        def set_articulation_position_iteration(self, name, prim, value=None):
            arti_api = self._get_physx_articulation_api(prim)
            solver_position_iteration_count = arti_api.GetSolverPositionIterationCountAttr()
            if value is None:
                value = self._get_actor_config_value(name, "solver_position_iteration_count",
                                                     solver_position_iteration_count)
            if value != -1:
                solver_position_iteration_count.Set(value)

        def set_articulation_velocity_iteration(self, name, prim, value=None):
            arti_api = self._get_physx_articulation_api(prim)
            solver_velocity_iteration_count = arti_api.GetSolverVelocityIterationCountAttr()
            if value is None:
                value = self._get_actor_config_value(name, "solver_velocity_iteration_count",
                                                     solver_position_iteration_count)
            if value != -1:
                solver_velocity_iteration_count.Set(value)

        def set_articulation_sleep_threshold(self, name, prim, value=None):
            arti_api = self._get_physx_articulation_api(prim)
            sleep_threshold = arti_api.GetSleepThresholdAttr()
            if value is None:
                value = self._get_actor_config_value(name, "sleep_threshold", sleep_threshold)
            if value != -1:
                sleep_threshold.Set(value)

        def set_articulation_stabilization_threshold(self, name, prim, value=None):
            arti_api = self._get_physx_articulation_api(prim)
            stabilization_threshold = arti_api.GetStabilizationThresholdAttr()
            if value is None:
                value = self._get_actor_config_value(name, "stabilization_threshold", stabilization_threshold)
            if value != -1:
                stabilization_threshold.Set(value)

        def apply_rigid_body_settings(self, name, prim, cfg, is_articulation):
            from pxr import UsdPhysics, PhysxSchema

            stage = omni.usd.get_context().get_stage()
            rb_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath())
            if not physx_rb_api:
                physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

            # if it's a body in an articulation, it's handled at articulation root
            if not is_articulation:
                self.make_kinematic(name, prim, cfg, cfg["make_kinematic"])
            self.set_position_iteration(name, prim, cfg["solver_position_iteration_count"])
            self.set_velocity_iteration(name, prim, cfg["solver_velocity_iteration_count"])
            self.set_max_depenetration_velocity(name, prim, cfg["max_depenetration_velocity"])
            self.set_sleep_threshold(name, prim, cfg["sleep_threshold"])
            self.set_stabilization_threshold(name, prim, cfg["stabilization_threshold"])
            self.set_gyroscopic_forces(name, prim, cfg["enable_gyroscopic_forces"])

            # density and mass
            mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
            if mass_api is None:
                mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_attr = mass_api.GetMassAttr()
            density_attr = mass_api.GetDensityAttr()
            if not mass_attr:
                mass_attr = mass_api.CreateMassAttr()
            if not density_attr:
                density_attr = mass_api.CreateDensityAttr()

            if cfg["density"] != -1:
                density_attr.Set(cfg["density"])
                mass_attr.Set(0.0)  # mass is to be computed
            elif cfg["override_usd_defaults"] and not density_attr.IsAuthored() and not mass_attr.IsAuthored():
                density_attr.Set(self._physx_params["density"])

            self.retain_acceleration(prim)

        def apply_rigid_shape_settings(self, name, prim, cfg):
            from pxr import UsdPhysics, PhysxSchema
            stage = omni.usd.get_context().get_stage()

            # collision APIs
            collision_api = UsdPhysics.CollisionAPI(prim)
            if not collision_api:
                collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not physx_collision_api:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)

            self.set_contact_offset(name, prim, cfg["contact_offset"])
            self.set_rest_offset(name, prim, cfg["rest_offset"])

        def apply_articulation_settings(self, name, prim, cfg):
            from pxr import UsdPhysics, PhysxSchema

            stage = omni.usd.get_context().get_stage()

            is_articulation = False
            # check if is articulation
            prims = [prim]
            while len(prims) > 0:
                prim_tmp = prims.pop(0)
                articulation_api = UsdPhysics.ArticulationRootAPI.Get(stage, prim_tmp.GetPath())
                physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Get(stage, prim_tmp.GetPath())

                if articulation_api or physx_articulation_api:
                    is_articulation = True

                children_prims = prim_tmp.GetPrim().GetChildren()
                prims = prims + children_prims

            # parse through all children prims
            prims = [prim]
            while len(prims) > 0:
                cur_prim = prims.pop(0)
                rb = UsdPhysics.RigidBodyAPI.Get(stage, cur_prim.GetPath())
                collision_body = UsdPhysics.CollisionAPI.Get(stage, cur_prim.GetPath())
                articulation = UsdPhysics.ArticulationRootAPI.Get(stage, cur_prim.GetPath())
                if rb:
                    self.apply_rigid_body_settings(name, cur_prim, cfg, is_articulation)
                if collision_body:
                    self.apply_rigid_shape_settings(name, cur_prim, cfg)

                if articulation:
                    articulation_api = UsdPhysics.ArticulationRootAPI.Get(stage, cur_prim.GetPath())
                    physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Get(stage, cur_prim.GetPath())

                    # enable self collisions
                    enable_self_collisions = physx_articulation_api.GetEnabledSelfCollisionsAttr()
                    if cfg["enable_self_collisions"] != -1:
                        enable_self_collisions.Set(cfg["enable_self_collisions"])

                    self.set_articulation_position_iteration(name, cur_prim, cfg["solver_position_iteration_count"])
                    self.set_articulation_velocity_iteration(name, cur_prim, cfg["solver_velocity_iteration_count"])
                    self.set_articulation_sleep_threshold(name, cur_prim, cfg["sleep_threshold"])
                    self.set_articulation_stabilization_threshold(name, cur_prim, cfg["stabilization_threshold"])

                children_prims = cur_prim.GetPrim().GetChildren()
                prims = prims + children_prims

    cfg_dict = omegaconf_to_dict(cfg)
    omni_config = OmniConfig(cfg_dict)
    return omni_config


def get_view_motion_config(config_name):
    """Load the configs stored in config_name.yaml.

    Args:
        config_name (str): Name of the config file for viewing motion.

    Returns:
        (dict): A dict of configs.
    """
    config_file_path = os.path.join(
        os.path.dirname(__file__), f"view_motion/{config_name}.yaml"
    )
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"{config_file_path} does not exist")

    return OmegaConf.load(config_file_path)


def load_ikea_config(ikea_name):
    """Load the configs stored in ikea_name.yaml.

    Args:
        ikea_name (str): Name of the config file under config/ikea for the ikea furniture.

    Returns:
        (dict): A dict of configs.
    """
    agent_config_file_path = os.path.join(
        os.path.dirname(__file__), f"ikea/{ikea_name}.yaml"
    )
    if not os.path.exists(agent_config_file_path):
        raise FileNotFoundError(f"{agent_config_file_path} does not exist")

    return OmegaConf.load(agent_config_file_path)


def print_config(config: DictConfig):
    print("-----------------------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------------------")


def omegaconf_to_dict(config: DictConfig) -> Dict:
    """
    Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation.
    """
    d = {}
    for k, v in config.items():
        try:
            d[k] = omegaconf_to_dict(v) if isinstance(v, DictConfig) else v
        except:
            pass
    return d


def dict_to_omegaconf(d: Dict, save_path: str = None) -> DictConfig:
    """
    Converts a python Dict to an omegaconf DictConfig, respecting variable interpolation.
    """
    conf = OmegaConf.create(d)
    if save_path is not None:
        with open(save_path, "w") as fp:
            OmegaConf.save(config=conf, f=fp.name)
            loaded = OmegaConf.load(fp.name)
            assert conf == loaded
    else:
        return conf


if __name__ == "__main__":
    TD3_DEFAULT_CONFIG = {
        "gradient_steps": 1,  # gradient steps
        "batch_size": 64,  # training batch size
        "discount_factor": 0.99,  # discount factor (gamma)
        "polyak": 0.005,  # soft update hyperparameter (tau)
        "actor_learning_rate": 1e-3,  # actor learning rate
        "critic_learning_rate": 1e-3,  # critic learning rate
        "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
        "learning_rate_scheduler_kwargs": {},  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
        "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
        "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})
        "random_timesteps": 0,  # random exploration steps
        "learning_starts": 0,  # learning starts after this many steps
        "exploration": {
            "noise": None,  # exploration noise
            "initial_scale": 1.0,  # initial scale for noise
            "final_scale": 1e-3,  # final scale for noise
            "timesteps": None,  # timesteps for noise decay
        },
        "policy_delay": 2,  # policy delay update with respect to critic update
        "smooth_regularization_noise": None,  # smooth noise for regularization
        "smooth_regularization_clip": 0.5,  # clip for smooth regularization
        "rewards_shaper": None,  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
        "experiment": {
            "directory": "",  # experiment's parent directory
            "experiment_name": "",  # experiment name
            "write_interval": 250,  # TensorBoard writing interval (timesteps)
            "checkpoint_interval": 1000,  # interval for checkpoints (timesteps)
            "store_separately": False,  # whether to store checkpoints separately
        },
    }

    dict_to_omegaconf(
        TD3_DEFAULT_CONFIG,
        save_path="/home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/config/learning/rl/agent/td3_default_config_skrl.yaml",
    )
