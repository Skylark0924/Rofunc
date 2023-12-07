""" Define base environment class FurnitureEnv. """

import logging
import os
import pickle
import time
from collections import OrderedDict
from sys import platform

import numpy as np
import gym.spaces
from pyquaternion import Quaternion
from scipy.interpolate import interp1d
import hjson
import yaml
import mujoco_py

from rofunc.learning.RofuncRL.tasks import transform_utils as T
from rofunc.learning.RofuncRL.tasks.isaacgymenv.base import EnvMeta
from .image_utils import color_segmentation
from .mjcf_utils import xml_path_completion
from .models import (
    background_names,
    furniture_name2id,
    furniture_names,
    furniture_xmls,
)
from .models.grippers import gripper_factory
from .models.objects import MujocoXMLObject
from .unity_interface import UnityInterface
from .controllers.arm_controller import *
from ..util.demo_recorder import DemoRecorder
from ..util.video_recorder import VideoRecorder
from ..util.logger import logger
from ..util import Qpos, PrettySafeLoader


np.set_printoptions(suppress=True)


NEW_CONTROLLERS = [
    "position",
    "position_orientation",
    "joint_impedance",
    "joint_torque",
    "joint_velocity",
]


class FurnitureTask(metaclass=EnvMeta):
    """
    Base class for IKEA furniture assembly environment.
    """

    name = "furniture"

    def __init__(self, config):
        """
        Initializes class with the configuration.
        """
        self._config = config

        # default env config
        self._max_episode_steps = config.max_episode_steps

        self._debug = config.debug
        if logger.getEffectiveLevel() != logging.CRITICAL:
            logger.setLevel(logging.INFO)
        if self._debug:
            logger.setLevel(logging.DEBUG)

        self._rng = np.random.RandomState(config.seed)

        if config.render and not config.unity:
            self._render_mode = "human"
        else:
            self._render_mode = "no"  # ['no', 'human', 'rgb_array']

        self._screen_width = config.screen_width
        self._screen_height = config.screen_height

        self._agent_type = config.agent_type
        self._control_type = config.control_type
        self._control_freq = config.control_freq  # reduce freq -> longer timestep
        self._discrete_grip = config.discrete_grip
        self._rescale_actions = config.rescale_actions
        self._auto_align = config.auto_align

        if self._agent_type == "Baxter":
            self._arms = ["right", "left"]
        else:
            self._arms = ["right"]

        if self._control_type in NEW_CONTROLLERS:
            self._load_controller(
                config.control_type,
                os.path.join(
                    os.path.dirname(__file__), "controllers/controller_config.hjson"
                ),
                {},
            )

        self._robot_ob = config.robot_ob
        self._object_ob = config.object_ob
        self._object_ob_all = config.object_ob_all
        self._visual_ob = config.visual_ob
        self._subtask_ob = config.subtask_ob
        self._segmentation_ob = config.segmentation_ob
        self._depth_ob = config.depth_ob
        self._camera_ids = config.camera_ids
        self._camera_name = "frontview"
        self._is_render = False
        self._furniture_id = None
        self._background = None
        self.init_pos = None
        self.init_quat = None
        self.fixed_parts = []

        self._manual_resize = None
        self._action_on = False
        self._init_qpos = None
        if config.load_demo:
            with open(config.load_demo, "rb") as f:
                demo = pickle.load(f)
                self._init_qpos = demo["states"][0]

        self._load_init_states = None
        if config.load_init_states:
            with open(config.load_init_states, "rb") as f:
                self._load_init_states = pickle.load(f)

        if config.furniture_name:
            furniture_name = config.furniture_name
            config.furniture_id = furniture_name2id[config.furniture_name]
        else:
            furniture_name = furniture_names[config.furniture_id]
        self.file_prefix = self._agent_type + "_" + furniture_name + "_"

        self._record_demo = config.record_demo
        if self._record_demo:
            self._demo = DemoRecorder(config.demo_dir)

        self._record_vid = config.record_vid
        self.vid_rec = None
        if self._record_vid:
            if self._record_demo:
                self.vid_rec = VideoRecorder(
                    record_mode=config.record_mode,
                    prefix=self.file_prefix,
                    demo_dir=config.demo_dir,
                )
            else:
                self.vid_rec = VideoRecorder(
                    record_mode=config.record_mode, prefix=self.file_prefix
                )

        self._num_connect_steps = 0
        self._gravity_compensation = 0

        self._move_speed = config.move_speed
        self._rotate_speed = config.rotate_speed

        self._preassembled = config.preassembled
        self._num_connects = config.num_connects

        if self._agent_type != "Cursor" and self._control_type in [
            "ik",
            "ik_quaternion",
        ]:
            self._min_gripper_pos = np.array([-1.5, -1.5, 0.0])
            self._max_gripper_pos = np.array([1.5, 1.5, 1.5])
            self._action_repeat = 3

        self._viewer = None
        self._unity = None
        self._unity_updated = False
        if config.unity:
            self._unity = UnityInterface(
                config.port, config.unity_editor, config.virtual_display
            )
            # set to the best quality
            self._unity.set_quality(config.quality)

        if config.render and platform == "win32":
            from mujoco_py import GlfwContext

            GlfwContext(offscreen=True)  # create a window to init GLFW

        if self._object_ob_all:
            if config.furniture_name is not None:
                self._furniture_id = furniture_name2id[config.furniture_name]
            else:
                self._furniture_id = config.furniture_id
            self._load_model_object()
            self._furniture_id = None

    def update_config(self, config):
        """ Updates private member variables with @config dictionary. """
        # Not all config can be appropriately updated.
        for k, v in config.items():
            if hasattr(self, "_" + k):
                setattr(self, "_" + k, v)

    def set_subtask(self, subtask, num_connects=None):
        """ Simply sets @self._preassembled to [0, 1, ..., @subtask]. """
        self._preassembled = range(subtask)
        self._num_connects = num_connects

    def num_subtask(self):
        if self._num_connects is not None:
            return self._num_connects
        else:
            return len(self._object_names) - 1

    @property
    def observation_space(self):
        """
        Returns dict where keys are ob names and values are dimensions.
        """
        ob_space = OrderedDict()
        num_cam = len(self._camera_ids)
        if self._visual_ob:
            ob_space["camera_ob"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(num_cam, self._screen_height, self._screen_width, 3),
                dtype=np.uint8,
            )

        if self._object_ob:
            # can be changed to the desired number depending on the task
            if self._object_ob_all:
                ob_space["object_ob"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=((3 + 4) * self.n_objects,),
                )
            else:
                ob_space["object_ob"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=((3 + 4) * 2,),
                )

        if self._subtask_ob:
            ob_space["subtask_ob"] = gym.spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(2,),
            )

        return gym.spaces.Dict(ob_space)

    @property
    def state_space(self):
        """
        Returns mujoco state space.
        """
        state_space = OrderedDict()
        state_space["qpos"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.sim.data.qpos),),
        )
        state_space["qvel"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.sim.data.qvel),),
        )
        return gym.spaces.Dict(state_space)

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with out grippers).
        """
        raise NotImplementedError

    @property
    def max_episode_steps(self):
        """
        Returns maximum number of steps in an episode.
        """
        return self._max_episode_steps

    @property
    def action_size(self):
        """
        Returns size of the action space.
        """
        return gym.spaces.flatdim(self.action_space)

    @property
    def action_space(self):
        """
        Returns action space.
        """
        return gym.spaces.Dict(
            [
                (
                    "default",
                    gym.spaces.Box(
                        shape=(self.dof,),
                        low=-1,
                        high=1,
                        dtype=np.float32,
                    ),
                )
            ]
        )

    def set_max_episode_steps(self, max_episode_steps):
        self._max_episode_steps = max_episode_steps

    def set_init_qpos(self, init_qpos):
        self._init_qpos = init_qpos

    def reset(self, furniture_id=None, background=None):
        """
        Takes in a furniture_id, and background string.
        Resets the environment, viewer, and internal variables.
        Returns the initial observation.
        """
        if self._record_demo:
            self._demo.reset()
        self._reset(furniture_id=furniture_id, background=background)
        self._after_reset()

        ob = self._get_obs()
        if self._record_demo:
            self._demo.add(ob=ob)
            self._demo.add(low_level_ob=self._get_obs(include_qpos=True))

        return ob

    def _init_random(self, size, name):
        """
        Returns initial random distribution.
        """
        if name == "furniture":
            r = self._config.furn_xyz_rand
        elif name == "agent":
            r = self._config.agent_xyz_rand
        elif name == "resize":
            r = self._config.furn_size_rand
        else:
            r = 0

        return self._rng.uniform(low=-r, high=r, size=size)

    def _after_reset(self):
        """
        Reset timekeeping and internal state for episode.
        """
        self._episode_reward = 0
        self._episode_length = 0
        self._episode_time = time.time()

        self._terminal = False
        self._success = False
        self._fail = False
        self._unity_updated = False

    def step(self, action):
        """
        Computes the next environment state given @action.
        Stores env state for demonstration if needed.
        Returns observation dict, reward float, done bool, and info dict.
        """
        self._before_step()
        _action = action
        if isinstance(action, list):
            action = {key: val for ac_i in action for key, val in ac_i.items()}
        if isinstance(action, dict):
            action = np.concatenate(
                [action[key] for key in self.action_space.spaces.keys()]
            )

        ob, reward, done, info = self._step(action)
        done, info, penalty = self._after_step(reward, done, info)
        reward += penalty
        if self._record_demo:
            self._store_state()
            self._demo.add(ob=ob, action=_action, reward=reward)
        return ob, reward, done, info

    def _before_step(self):
        """
        Called before step
        """
        self._unity_updated = False
        self._connected = False

    def _update_unity(self):
        """
        Updates unity rendering with qpos. Call this after you change qpos
        """
        self._unity.set_qpos(self.sim.data.qpos)
        if self._agent_type == "Cursor":
            for cursor_i in range(2):
                cursor_name = "cursor%d" % cursor_i
                cursor_pos = self._get_pos(cursor_name)
                self._unity.set_geom_pos(cursor_name, cursor_pos)

    def _step(self, a):
        """
        Internal step function. Moves agent, updates unity, and then
        returns ob, reward, done, info tuple
        """
        if a is None:
            a = np.zeros(self.dof)

        if self._agent_type == "Cursor":
            self._step_discrete(a.copy())
            self._do_simulation(None)

        elif self._control_type in ["ik", "ik_quaternion", "torque", "impedance"]:
            self._step_continuous(a.copy())

        elif self._control_type in NEW_CONTROLLERS:
            self._step_continuous(a.copy())

        else:
            raise ValueError

        if self._connected_body1 is not None:
            self.sim.forward()
            self._move_objects_target(
                self._connected_body1,
                self._connected_body1_pos,
                self._connected_body1_quat,
                self._gravity_compensation,
            )
            self._connected_body1 = None
            self.sim.forward()
            self.sim.step()

        ob = self._get_obs()
        done = False
        if (
            self._num_connected == self._success_num_conn
            and len(self._object_names) > 1
        ):
            self._success = True
            done = True

        reward = 0
        info = {}
        return ob, reward, done, info

    def _after_step(self, reward, terminal, info):
        """
        Called after _step, adds additional information and calculate penalty
        """
        step_log = dict(info)
        self._terminal = terminal
        penalty = 0

        if reward is not None:
            self._episode_reward += reward
            self._episode_length += 1

        if self._episode_length == self._max_episode_steps or self._fail:
            self._terminal = True
            if self._fail:
                self._fail = False
                penalty = -self._config.unstable_penalty_coef

        if self._terminal:
            total_time = time.time() - self._episode_time
            step_log["episode_success"] = int(self._success)
            step_log["episode_reward"] = self._episode_reward + penalty
            step_log["episode_length"] = self._episode_length
            step_log["episode_time"] = total_time
            step_log["episode_unstable"] = penalty
            step_log["episode_num_connected"] = self._num_connected
            if self._success:
                step_log["episode_success_state"] = self.get_env_state()

        return self._terminal, step_log, penalty

    def _compute_reward(self, ac):
        """
        Computes the reward at the current step
        """
        # Touch & pick reward
        touch_reward = 0
        pick_reward = 0
        if self._agent_type != "Cursor":
            floor_geom_id = self.sim.model.geom_name2id("FLOOR")
            touch_floor = {}
            for arm in self._arms:
                touch_left_finger = {}
                touch_right_finger = {}
                for body_id in self._object_body_ids:
                    touch_left_finger[body_id] = False
                    touch_right_finger[body_id] = False
                    touch_floor[body_id] = False

                for j in range(self.sim.data.ncon):
                    c = self.sim.data.contact[j]
                    body1 = self.sim.model.geom_bodyid[c.geom1]
                    body2 = self.sim.model.geom_bodyid[c.geom2]

                    for geom_id, body_id in [(c.geom1, body2), (c.geom2, body1)]:
                        if not (body_id in self._object_body_ids):
                            continue
                        if geom_id in self.l_finger_geom_ids[arm]:
                            touch_left_finger[body_id] = True
                        if geom_id in self.r_finger_geom_ids[arm]:
                            touch_right_finger[body_id] = True
                        if geom_id == floor_geom_id:
                            touch_floor[body_id] = True

                for body_id in self._object_body_ids:
                    if touch_left_finger[body_id] and touch_right_finger[body_id]:
                        if not self._touched[body_id]:
                            self._touched[body_id] = True
                            touch_reward += self._config.touch_reward
                        if not touch_floor[body_id] and not self._picked[body_id]:
                            self._picked[body_id] = True
                            pick_reward += self._config.pick_reward

        # Success reward
        success_reward = self._config.success_reward * (
            self._num_connected - self._prev_num_connected
        )
        self._prev_num_connected = self._num_connected

        ctrl_penalty = self._ctrl_penalty(ac)

        reward = success_reward + touch_reward + pick_reward + ctrl_penalty
        # do not terminate
        done = False
        info = {
            "success_reward": success_reward,
            "touch_reward": touch_reward,
            "pick_reward": pick_reward,
            "ctrl_penalty": ctrl_penalty,
        }
        return reward, done, info

    def _ctrl_penalty(self, a):
        """
        Control penalty to discourage erratic motions.
        """
        if a is None or self._agent_type == "Cursor":
            return 0
        ctrl_penalty = -self._config.ctrl_penalty_coef * np.square(a).sum()
        return ctrl_penalty

    def _set_camera_position(self, cam_id, cam_pos):
        """
        Sets cam_id camera position to cam_pos
        """
        self.sim.model.cam_pos[cam_id] = cam_pos.copy()

    def _set_camera_rotation(self, cam_id, target_pos):
        """
        Rotates camera to look at target_pos
        """
        cam_pos = self.sim.model.cam_pos[cam_id]
        forward = target_pos - cam_pos
        up = [
            forward[0],
            forward[1],
            (forward[0] ** 2 + forward[1] ** 2) / (-forward[2]),
        ]
        if forward[0] == 0 and forward[1] == 0:
            up = [0, 1, 0]
        q = T.lookat_to_quat(-forward, up)
        self.sim.model.cam_quat[cam_id] = T.convert_quat(q, to="wxyz")

    def _set_camera_pose(self, cam_id, pose):
        """
        Sets unity camera to pose
        """
        self._unity.set_camera_pose(cam_id, pose)

    def _render_callback(self):
        """
        Callback for rendering
        """
        pass

    def render(self, mode="human"):
        """
        Renders the environment. If mode is rgb_array, we render the pixels.
        The pixels can be rgb, depth map, segmentation map
        If the mode is human, we render to the MuJoCo viewer, or for unity,
        do nothing since rendering is handled by Unity.
        """
        self._render_callback()

        # update unity
        if self._unity and not self._unity_updated:
            self._update_unity()
            self._unity_updated = True

        if mode == "rgb_array":
            if self._unity:
                img, _ = self._unity.get_images(self._camera_ids)
            else:
                img = self.sim.render(
                    camera_name=self._camera_name,
                    width=self._screen_width,
                    height=self._screen_height,
                    depth=False,
                )
                img = np.expand_dims(img, axis=0)
            assert len(img.shape) == 4
            # img = img[:, ::-1, :, :] / 255.0
            img = img[:, ::-1, :, :]
            return img

        elif mode == "rgbd_array":
            depth = None
            if self._unity:
                img, depth = self._unity.get_images(self._camera_ids, self._depth_ob)
            else:
                camera_obs = self.sim.render(
                    camera_name=self._camera_name,
                    width=self._screen_width,
                    height=self._screen_height,
                    depth=self._depth_ob,
                )
                if self._depth_ob:
                    img, depth = camera_obs
                else:
                    img = camera_obs
                img = np.expand_dims(img, axis=0)
            # img = img[:, ::-1, :, :] / 255.0
            img = img[:, ::-1, :, :]

            if depth is not None:
                # depth map is 0 to 1, with 1 being furthest
                # infinite depth is 0, so set to 1
                black_pixels = np.all(depth == [0, 0, 0], axis=-1)
                depth[black_pixels] = [255] * 3
                if len(depth.shape) == 4:
                    # depth = depth[:, ::-1, :, :] / 255.0
                    depth = depth[:, ::-1, :, :]
                elif len(depth.shape) == 3:
                    # depth = depth[::-1, :, :] / 255.0
                    depth = depth[::-1, :, :]

            return img, depth

        elif mode == "segmentation" and self._unity:
            img = self._unity.get_segmentations(self._camera_ids)
            return img

        elif mode == "human" and not self._unity:
            if platform != "win32":
                self._get_viewer().render()

        return None

    def _destroy_viewer(self):
        """
        Destroys the current viewer if there is one
        """
        if self._viewer is not None:
            import glfw

            glfw.destroy_window(self._viewer.window)
            self._viewer = None

    def _viewer_reset(self):
        """
        Resets the viewer
        """
        pass

    def _get_viewer(self):
        """
        Returns the viewer instance, or instantiates a new one
        """
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self.sim)
            self._viewer.cam.fixedcamid = self._camera_ids[0]
            self._viewer.cam.type = mujoco_py.generated.const.CAMERA_FIXED
            self._viewer_reset()
        return self._viewer

    def close(self):
        """
        Cleans up the environment
        """
        if self._unity:
            self._unity.disconnect_to_unity()
        self._destroy_viewer()

    def __delete__(self):
        """
        Called to destroy environment
        """
        if self._unity:
            self._unity.disconnect_to_unity()

    def __del__(self):
        """
        Called to destroy environment
        """
        if self._unity:
            self._unity.disconnect_to_unity()

    def _move_cursor(self, cursor_i, move_offset):
        """
        Moves cursor by move_offset amount, takes into account the
        boundary
        """
        cursor_name = "cursor%d" % cursor_i
        cursor_pos = self._get_pos(cursor_name)
        cursor_pos = cursor_pos + move_offset
        boundary = self._config.cursor_boundary
        if (np.abs(cursor_pos) < boundary).all() and cursor_pos[
            2
        ] >= self._move_speed * 0.45:
            self._set_pos(cursor_name, cursor_pos)
            return True
        return False

    def _move_rotate_object(self, obj, move_offset, rotate_offset):
        """
        Used by cursor to move and rotate selected objects
        """
        qpos_base = self._get_qpos(obj)
        target_quat = T.euler_to_quat(rotate_offset, qpos_base[3:])

        part_idx = self._object_name2id[obj]
        old_pos_rot = {}
        for i, obj_name in enumerate(self._object_names):
            if self._find_group(i) == self._find_group(part_idx):
                old_pos_rot[obj_name] = self._get_qpos(obj_name)
                new_pos, new_rot = T.transform_to_target_quat(
                    qpos_base, self._get_qpos(obj_name), target_quat
                )
                new_pos = new_pos + move_offset
                self._set_qpos(obj_name, new_pos, new_rot)

        if self._is_inside(obj):
            return True

        for obj_name, pos_rot in old_pos_rot.items():
            self._set_qpos(obj_name, pos_rot[:3], pos_rot[3:])
        return False

    def _get_bounding_box(self, obj_name):
        """
        Gets the bounding box of the object
        """
        body_ids = []
        part_idx = self._object_name2id[obj_name]
        for i, body_name in enumerate(self._object_names):
            if self._find_group(i) == self._find_group(part_idx):
                body_id = self.sim.model.body_name2id(body_name)
                body_ids.append(body_id)

        body_id = self.sim.model.body_name2id(obj_name)
        min_pos = np.array([0, 0, 0])
        max_pos = np.array([0, 0, 0])
        for i, site in enumerate(self.sim.model.site_names):
            if self.sim.model.site_bodyid[i] in body_ids:
                pos = self._get_pos(site)
                min_pos = np.minimum(min_pos, pos)
                max_pos = np.maximum(max_pos, pos)

        return min_pos, max_pos

    def _is_inside(self, obj_name):
        """
        Determines if object is inside the boundary
        """
        self.sim.forward()
        self.sim.step()
        min_pos, max_pos = self._get_bounding_box(obj_name)
        b = self._config.cursor_boundary
        if (min_pos < np.array([-b, -b, -0.05])).any() or (
            max_pos > np.array([b, b, b])
        ).any():
            return False
        return True

    def _select_object(self, cursor_i):
        """
        Selects an object within cursor_i
        """
        for obj_name in self._object_names:
            is_selected = False
            obj_group = self._find_group(obj_name)
            for selected_obj in self._cursor_selected:
                if selected_obj and obj_group == self._find_group(selected_obj):
                    is_selected = True

            if not is_selected and self.on_collision("cursor%d" % cursor_i, obj_name):
                return obj_name
        return None

    def _step_discrete(self, a):
        """
        Takes a step for the cursor agent
        """
        assert len(a) == 15
        actions = [a[:7], a[7:]]

        for cursor_i in range(2):
            # move
            move_offset = actions[cursor_i][0:3] * self._move_speed
            # rotate
            rotate_offset = actions[cursor_i][3:6] * self._rotate_speed
            # select
            select = actions[cursor_i][6] > 0

            if not select:
                self._cursor_selected[cursor_i] = None

            success = self._move_cursor(cursor_i, move_offset)
            if not success:
                logger.debug("could not move cursor")
                continue
            if self._cursor_selected[cursor_i] is not None:
                success = self._move_rotate_object(
                    self._cursor_selected[cursor_i], move_offset, rotate_offset
                )
                if not success:
                    logger.debug("could not move cursor due to object out of boundary")
                    # reset cursor to original position
                    self._move_cursor(cursor_i, -move_offset)
                    continue

            if select:
                if self._cursor_selected[cursor_i] is None:
                    self._cursor_selected[cursor_i] = self._select_object(cursor_i)

        connect = a[14]
        if connect > 0 and self._cursor_selected[0] and self._cursor_selected[1]:
            logger.debug(
                "try connect ({} and {})".format(
                    self._cursor_selected[0], self._cursor_selected[1]
                )
            )
            self._try_connect(self._cursor_selected[0], self._cursor_selected[1])
        elif self._connect_step > 0:
            self._connect_step = 0

    def _connect(self, site1_id, site2_id, auto_align=True):
        """
        Connects two sites together with weld constraint.
        Makes the two objects are within boundaries
        """
        self._connected_sites.add(site1_id)
        self._connected_sites.add(site2_id)
        self._site1_id = site1_id
        self._site2_id = site2_id
        site1 = self.sim.model.site_names[site1_id]
        site2 = self.sim.model.site_names[site2_id]

        logger.debug("**** connect {} and {}".format(site1, site2))

        body1_id = self.sim.model.site_bodyid[site1_id]
        body2_id = self.sim.model.site_bodyid[site2_id]
        body1 = self.sim.model.body_id2name(body1_id)
        body2 = self.sim.model.body_id2name(body2_id)

        # remove collision
        group1 = self._find_group(body1)
        group2 = self._find_group(body2)
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            if body_name in self._object_names:
                group = self._find_group(body_name)
                if group in [group1, group2]:
                    if self.sim.model.geom_contype[geom_id] != 0:
                        self.sim.model.geom_contype[geom_id] = (
                            (1 << 30) - 1 - (1 << (group1 + 1))
                        )
                        self.sim.model.geom_conaffinity[geom_id] = 1 << (group1 + 1)

        # align site
        if auto_align:
            self._align_connectors(site1, site2, gravity=self._gravity_compensation)

        # move furniture to collision-safe position
        if self._agent_type == "Cursor":
            self._stop_selected_objects()
        self.sim.forward()
        self.sim.step()

        min_pos1, max_pos1 = self._get_bounding_box(body1)
        min_pos2, max_pos2 = self._get_bounding_box(body2)
        min_pos = np.minimum(min_pos1, min_pos2)
        if min_pos[2] < 0:
            offset = [0, 0, -min_pos[2]]
            self._move_rotate_object(body1, offset, [0, 0, 0])
            self._move_rotate_object(body2, offset, [0, 0, 0])

        if self._agent_type == "Cursor":
            self._stop_selected_objects()
        self.sim.forward()
        self.sim.step()

        # activate weld
        self._activate_weld(body1, body2)

        # release cursor
        if self._agent_type == "Cursor":
            self._cursor_selected[1] = None

        self._num_connected += 1
        self._connected = True
        self._connected_body1 = body1
        self._connected_body1_pos = self._get_qpos(body1)[:3]
        self._connected_body1_quat = self._get_qpos(body1)[3:]

        # set next subtask
        self._get_next_subtask()

        # reset robot arm
        if self._config.reset_robot_after_attach:
            self._initialize_robot_pos()
            if self._control_type in ["ik", "ik_quaternion"]:
                # set up ik controller
                self._controller.sync_state()

    def _try_connect(self, part1=None, part2=None):
        """
        Attempts to connect 2 parts. If they are correctly aligned,
        then we interpolate the 2 parts towards the target position and orientation
        for smoother visual connection.
        """
        if part1 is not None:
            body1_ids = [
                self.sim.model.body_name2id(obj_name)
                for obj_name in self._object_names
                if self._find_group(obj_name) == self._find_group(part1)
            ]
        else:
            body1_ids = [
                self.sim.model.body_name2id(obj_name) for obj_name in self._object_names
            ]

        if part2 is not None:
            body2_ids = [
                self.sim.model.body_name2id(obj_name)
                for obj_name in self._object_names
                if self._find_group(obj_name) == self._find_group(part2)
            ]
        else:
            body2_ids = [
                self.sim.model.body_name2id(obj_name) for obj_name in self._object_names
            ]

        sites1 = []
        sites2 = []
        for j, site in enumerate(self.sim.model.site_names):
            if "conn_site" in site:
                if self.sim.model.site_bodyid[j] in body1_ids:
                    sites1.append((j, site))
                if self.sim.model.site_bodyid[j] in body2_ids:
                    sites2.append((j, site))

        if len(sites1) == 0 or len(sites2) == 0:
            return False

        for i, (id1, id2) in enumerate(
            zip(self.sim.model.eq_obj1id, self.sim.model.eq_obj2id)
        ):
            if id1 in (body1_ids + body2_ids) and id2 in (body1_ids + body2_ids):
                break
        else:
            return False

        # site bookkeeping
        site_bodyid = self.sim.model.site_bodyid
        body_names = self.sim.model.body_names

        for site1_id, site1_name in sites1:
            site1_pairs = site1_name.split(",")[0].split("-")
            for site2_id, site2_name in sites2:
                site2_pairs = site2_name.split(",")[0].split("-")
                # first check if already connected
                if (
                    site1_id in self._connected_sites
                    or site2_id in self._connected_sites
                ):
                    continue
                if site1_pairs == site2_pairs[::-1]:
                    if self._is_aligned(site1_name, site2_name):
                        logger.debug(
                            f"connect {site1_name} and {site2_name}, {self._connect_step}/{self._num_connect_steps}"
                        )
                        if self._connect_step < self._num_connect_steps:
                            # set target as site2 pos
                            site1_pos_quat = self._site_xpos_xquat(site1_name)
                            site1_quat = self._target_connector_xquat
                            target_pos = site1_pos_quat[:3]
                            body2id = site_bodyid[site2_id]
                            part2 = body_names[body2id]
                            part2_qpos = self._get_qpos(part2).copy()
                            site2_pos_quat = self._site_xpos_xquat(site2_name)
                            site2_pos = site2_pos_quat[:3]
                            body_pos, body_rot = T.transform_to_target_quat(
                                site2_pos_quat, part2_qpos, site1_quat
                            )
                            body_pos += target_pos - site2_pos
                            if self._connect_step == 0:
                                # generate rotation interpolations
                                self.next_rot = []
                                for f in range(self._num_connect_steps):
                                    step = (f + 1) * 1 / (self._num_connect_steps)
                                    q = T.quat_slerp(part2_qpos[3:], body_rot, step)
                                    self.next_rot.append(q)

                                # generate pos interpolation
                                x = [0, 1]
                                y = [part2_qpos[:3], body_pos]
                                f = interp1d(x, y, axis=0)
                                xnew = np.linspace(
                                    1 / self._num_connect_steps,
                                    0.9,
                                    self._num_connect_steps,
                                )
                                self.next_pos = f(xnew)

                            next_pos, next_rotation = (
                                self.next_pos[self._connect_step],
                                self.next_rot[self._connect_step],
                            )
                            self._move_objects_target(
                                part2, next_pos, list(next_rotation)
                            )
                            self._connect_step += 1
                            return False
                        else:
                            self._connect(site1_id, site2_id, self._auto_align)
                            self._connect_step = 0
                            self.next_pos = self.next_rot = None
                            return True

        self._connect_step = 0
        return False

    def _site_xpos_xquat(self, site):
        """
        Gets the site's position and quaternion
        """
        site_id = self.sim.model.site_name2id(site)
        site_xpos = self.sim.data.get_site_xpos(site).copy()
        site_quat = self.sim.model.site_quat[site_id].copy()
        body_id = self.sim.model.site_bodyid[site_id]
        body_quat = self.sim.data.body_xquat[body_id].copy()

        site_xquat = list(Quaternion(body_quat) * Quaternion(site_quat))
        return np.hstack([site_xpos, site_xquat])

    def _is_aligned(self, connector1, connector2):
        """
        Checks if two sites are connected or not, given the site names, and
        returns possible rotations
        """
        site1_xpos = self._site_xpos_xquat(connector1)
        site2_xpos = self._site_xpos_xquat(connector2)

        allowed_angles = [x for x in connector1.split(",")[1:-1] if x]
        for i in range(len(allowed_angles)):
            allowed_angles[i] = float(allowed_angles[i])

        up1 = self._get_up_vector(connector1)
        up2 = self._get_up_vector(connector2)
        forward1 = self._get_forward_vector(connector1)
        forward2 = self._get_forward_vector(connector2)
        pos_dist = T.l2_dist(site1_xpos[:3], site2_xpos[:3])
        rot_dist_up = T.cos_siml(up1, up2)
        rot_dist_forward = T.cos_siml(forward1, forward2)

        project1_2 = np.dot(up1, T.unit_vector(site2_xpos[:3] - site1_xpos[:3]))
        project2_1 = np.dot(up2, T.unit_vector(site1_xpos[:3] - site2_xpos[:3]))

        logger.debug(
            f"pos_dist: {pos_dist}  "
            + f"rot_dist_up: {rot_dist_up}  "
            + f"rot_dist_forward: {rot_dist_forward}  "
            + f"project: {project1_2}, {project2_1}  "
        )

        max_rot_dist_forward = rot_dist_forward
        if len(allowed_angles) == 0:
            is_rot_forward_aligned = True
            cos = T.cos_siml(forward1, forward2)
            forward1_rotated_pos = T.rotate_vector_cos_siml(forward1, up1, cos, 1)
            forward1_rotated_neg = T.rotate_vector_cos_siml(forward1, up1, cos, -1)
            rot_dist_forward_pos = T.cos_siml(forward1_rotated_pos, forward2)
            rot_dist_forward_neg = T.cos_siml(forward1_rotated_neg, forward2)
            if rot_dist_forward_pos > rot_dist_forward_neg:
                forward1_rotated = forward1_rotated_pos
            else:
                forward1_rotated = forward1_rotated_neg
            max_rot_dist_forward = max(rot_dist_forward_pos, rot_dist_forward_neg)
            self._target_connector_xquat = T.convert_quat(
                T.lookat_to_quat(up1, forward1_rotated), "wxyz"
            )
        else:
            is_rot_forward_aligned = False
            for angle in allowed_angles:
                forward1_rotated = T.rotate_vector(forward1, up1, angle)
                rot_dist_forward = T.cos_siml(forward1_rotated, forward2)
                max_rot_dist_forward = max(max_rot_dist_forward, rot_dist_forward)
                if rot_dist_forward > self._config.alignment_rot_dist_forward:
                    is_rot_forward_aligned = True
                    self._target_connector_xquat = T.convert_quat(
                        T.lookat_to_quat(up1, forward1_rotated), "wxyz"
                    )
                    break

        if (
            pos_dist < self._config.alignment_pos_dist
            and rot_dist_up > self._config.alignment_rot_dist_up
            and is_rot_forward_aligned
            and abs(project1_2) > self._config.alignment_project_dist
            and abs(project2_1) > self._config.alignment_project_dist
        ):
            return True

        # connect two parts if they are very close to each other
        if (
            pos_dist < self._config.alignment_pos_dist / 2
            and rot_dist_up > self._config.alignment_rot_dist_up
            and is_rot_forward_aligned
        ):
            return True

        if pos_dist >= self._config.alignment_pos_dist:
            logger.debug(
                "(connect) two parts are too far ({} >= {})".format(
                    pos_dist, self._config.alignment_pos_dist
                )
            )
        elif rot_dist_up <= self._config.alignment_rot_dist_up:
            logger.debug(
                "(connect) misaligned ({} <= {})".format(
                    rot_dist_up, self._config.alignment_rot_dist_up
                )
            )
        elif not is_rot_forward_aligned:
            logger.debug(
                "(connect) aligned, but rotate a connector ({} <= {})".format(
                    max_rot_dist_forward, self._config.alignment_rot_dist_forward
                )
            )
        else:
            logger.debug("(connect) misaligned. move connectors to align the axis")
        return False

    def _move_objects_target(self, obj, target_pos, target_quat, gravity=1):
        """
        Moves objects toward target position and quaternion
        """
        qpos_base = self._get_qpos(obj)
        translation = target_pos - qpos_base[:3]
        self._move_objects_translation_quat(obj, translation, target_quat, gravity)

    def _move_objects_translation_quat(self, obj, translation, target_quat, gravity=1):
        """
        Moves objects with translation and target quaternion
        """
        obj_id = self._object_name2id[obj]
        qpos_base = self._get_qpos(obj)
        for i, obj_name in enumerate(self._object_names):
            if self._find_group(i) == self._find_group(obj_id):
                new_pos, new_rot = T.transform_to_target_quat(
                    qpos_base, self._get_qpos(obj_name), target_quat
                )
                new_pos = new_pos + translation
                self._set_qpos(obj_name, new_pos, new_rot)
                self._stop_object(obj_name, gravity=gravity)

    def _project_connector_forward(self, connector1, connector2, angle=None):
        """
        Returns @connector2's forward vector when aligned with @connector1 with @angle
        """
        up1 = self._get_up_vector(connector1)
        forward1 = self._get_forward_vector(connector1)
        forward2 = self._get_forward_vector(connector2)

        if angle is None:
            cos = T.cos_siml(forward1, forward2)
            forward1_rotated_pos = T.rotate_vector_cos_siml(forward1, up1, cos, 1)
            forward1_rotated_neg = T.rotate_vector_cos_siml(forward1, up1, cos, -1)
            rot_dist_forward_pos = T.cos_siml(forward1_rotated_pos, forward2)
            rot_dist_forward_neg = T.cos_siml(forward1_rotated_neg, forward2)
            if rot_dist_forward_pos > rot_dist_forward_neg:
                forward1_rotated = forward1_rotated_pos
            else:
                forward1_rotated = forward1_rotated_neg
        else:
            forward1_rotated = T.rotate_vector(forward1, up1, angle)

        return forward1_rotated

    def _project_connector_quat(self, connector1, connector2, angle=None):
        """
        Returns @connector2's xquat when aligned with @connector1 with @angle
        """
        up1 = self._get_up_vector(connector1)
        forward1 = self._get_forward_vector(connector1)
        forward2 = self._get_forward_vector(connector2)

        if angle is None:
            cos = T.cos_siml(forward1, forward2)
            forward1_rotated_pos = T.rotate_vector_cos_siml(forward1, up1, cos, 1)
            forward1_rotated_neg = T.rotate_vector_cos_siml(forward1, up1, cos, -1)
            rot_dist_forward_pos = T.cos_siml(forward1_rotated_pos, forward2)
            rot_dist_forward_neg = T.cos_siml(forward1_rotated_neg, forward2)
            if rot_dist_forward_pos > rot_dist_forward_neg:
                forward1_rotated = forward1_rotated_pos
            else:
                forward1_rotated = forward1_rotated_neg
        else:
            forward1_rotated = T.rotate_vector(forward1, up1, angle)

        return T.convert_quat(T.lookat_to_quat(up1, forward1_rotated), "wxyz")

    def _align_connectors(self, connector1, connector2, gravity=1):
        """
        Moves connector2 to connector 1
        """
        site1_xpos = self._site_xpos_xquat(connector1)
        site1_xpos[3:] = self._target_connector_xquat
        self._move_site_to_target(connector2, site1_xpos, gravity)

    def _move_site_to_target(self, site, target_qpos, gravity=1):
        """
        Moves target site towards target quaternion / position
        """
        qpos_base = self._site_xpos_xquat(site)
        target_quat = target_qpos[3:]

        site_id = self.sim.model.site_name2id(site)
        body_id = self.sim.model.site_bodyid[site_id]
        body_name = self.sim.model.body_names[body_id]
        body_qpos = self._get_qpos(body_name)
        new_pos, new_quat = T.transform_to_target_quat(
            qpos_base, body_qpos, target_quat
        )
        new_site_pos, new_site_quat = T.transform_to_target_quat(
            body_qpos, qpos_base, new_quat
        )
        translation = target_qpos[:3] - new_site_pos
        self._move_objects_translation_quat(body_name, translation, new_quat, gravity)

    def _bounded_d_pos(self, d_pos, pos):
        """
        Clips d_pos to the gripper limits
        """
        min_action = self._min_gripper_pos - pos
        max_action = self._max_gripper_pos - pos
        return np.clip(d_pos, min_action, max_action)

    def _step_continuous(self, action):
        """
        Step function for continuous control
        """
        connect = action[-1]
        if self._control_type in ["ik", "ik_quaternion"]:
            self._do_ik_step(action)

        elif self._control_type == "torque":
            self._do_simulation(action[:-1])
            if self._record_demo:
                self._demo.add(
                    low_level_ob=self._get_obs(include_qpos=True),
                    low_level_action=action[:-1],
                    connect_action=connect,
                )

        elif self._control_type == "impedance":
            a = self._setup_action(action[:-1])
            self._do_simulation(a)
            if self._record_demo:
                self._demo.add(
                    low_level_ob=self._get_obs(include_qpos=True),
                    low_level_action=action[:-1],
                    connect_action=connect,
                )

        elif self._control_type in NEW_CONTROLLERS:
            self._do_controller_step(action)

        if connect > 0:
            for arm in self._arms:
                touch_left_finger = {}
                touch_right_finger = {}
                for body_id in self._object_body_ids:
                    touch_left_finger[body_id] = False
                    touch_right_finger[body_id] = False

                for j in range(self.sim.data.ncon):
                    c = self.sim.data.contact[j]
                    body1 = self.sim.model.geom_bodyid[c.geom1]
                    body2 = self.sim.model.geom_bodyid[c.geom2]
                    if (
                        c.geom1 in self.l_finger_geom_ids[arm]
                        and body2 in self._object_body_ids
                    ):
                        touch_left_finger[body2] = True
                    if (
                        body1 in self._object_body_ids
                        and c.geom2 in self.l_finger_geom_ids[arm]
                    ):
                        touch_left_finger[body1] = True

                    if (
                        c.geom1 in self.r_finger_geom_ids[arm]
                        and body2 in self._object_body_ids
                    ):
                        touch_right_finger[body2] = True
                    if (
                        body1 in self._object_body_ids
                        and c.geom2 in self.r_finger_geom_ids[arm]
                    ):
                        touch_right_finger[body1] = True

                for body_id in self._object_body_ids:
                    if touch_left_finger[body_id] and touch_right_finger[body_id]:
                        logger.debug("try connect")
                        result = self._try_connect(self.sim.model.body_id2name(body_id))
                        if result:
                            return
                        break

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat.
        """
        return {
            "dpos": action[:3],
            # IK controller takes an absolute orientation in robot base frame
            "rotation": T.quat2mat(T.quat_multiply(old_quat, action[3:7])),
        }

    def _get_obs(self, include_qpos=False):
        """
        Returns observation dictionary
        """
        state = OrderedDict()

        # visual obs
        if self._visual_ob:
            camera_obs, depth_obs = self.render("rgbd_array")
            state["camera_ob"] = camera_obs
            if depth_obs is not None:
                state["depth_ob"] = depth_obs

        if self._segmentation_ob:
            segmentation_obs = self.render("segmentation")
            state["segmentation_ob"] = segmentation_obs

        # object states
        if self._object_ob:
            obj_states = OrderedDict()
            for i, obj_name in enumerate(self._object_names):
                if self._object_ob_all or i in [
                    self._subtask_part1,
                    self._subtask_part2,
                ]:
                    obj_pos = self._get_pos(obj_name)
                    obj_quat = self._get_quat(obj_name)
                    obj_states["{}_pos".format(obj_name)] = obj_pos
                    obj_states["{}_quat".format(obj_name)] = obj_quat

            if not self._object_ob_all and self._subtask_part1 == -1:
                obj_states["dummy"] = np.zeros(14)

            state["object_ob"] = np.concatenate(
                [x.ravel() for _, x in obj_states.items()]
            )

        # part ids
        if self._subtask_ob:
            state["subtask_ob"] = np.array(
                [self._subtask_part1 + 1, self._subtask_part2 + 1]
            )

        return state

    def _place_objects(self):
        """
        Returns the randomly distributed initial positions and rotations of furniture parts.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        if self._config.fix_init_parts and len(self.fixed_parts) == 0 and self.init_pos:
            mjcf_obj = next(iter(self.mujoco_objects.values()))
            for part in self._config.fix_init_parts:
                pos = self.init_pos[part]
                quat = Quaternion(self.init_quat[part])
                rad = mjcf_obj.get_horizontal_radius(part)
                self.fixed_parts.append((part, rad, Qpos(pos[0], pos[1], pos[2], quat)))
        return self.mujoco_model.place_objects(fixed_parts=self.fixed_parts)

    def _reset(self, furniture_id=None, background=None):
        """
        Internal reset function that resets the furniture and agent
        Randomly resets furniture by disabling robot collision, spreading
        parts around, and then reenabling collision.
        """
        if self._config.furniture_name == "Random":
            furniture_id = self._rng.randint(len(furniture_xmls))
        if (
            self._furniture_id is None
            or (self._furniture_id != furniture_id and furniture_id is not None)
            or (self._manual_resize is not None)
        ):
            # construct mujoco xml for furniture_id
            self._furniture_id = furniture_id or self._config.furniture_id
            self._reset_internal()
            self.file_prefix = (
                self._agent_type + "_" + furniture_names[self._furniture_id] + "_"
            )
            if self.vid_rec:
                self.vid_rec.set_outfile(self.file_prefix)

        if self._config.furn_size_rand != 0:
            rand = self._init_random(1, "resize")[0]
            resize_factor = 1 + rand
            self.mujoco_model.resize_objects(resize_factor)

        if self._load_init_states and np.random.rand() > 0.2:
            self.set_init_qpos(np.random.choice(self._load_init_states))

        # reset simulation data and clear buffers
        self.sim.reset()

        # store robot's contype, conaffinity (search MuJoCo XML API for details)
        # disable robot collision
        robot_col = {}
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name not in self._object_names and self.mujoco_robot.is_robot_part(
                geom_name
            ):
                robot_col[geom_name] = (
                    self.sim.model.geom_contype[geom_id],
                    self.sim.model.geom_conaffinity[geom_id],
                )
                self.sim.model.geom_contype[geom_id] = 0
                self.sim.model.geom_conaffinity[geom_id] = 0

        # initialize collision for non-visual geoms
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            body_name = self.sim.model.body_names[body_id]
            geom_name = self.sim.model.geom_id2name(geom_id)
            if body_name in self._object_names and "collision" in geom_name:
                self.sim.model.geom_contype[geom_id] = 1
                self.sim.model.geom_conaffinity[geom_id] = 1

        # initialize group
        self._object_group = list(range(len(self._object_names)))

        # initialize member variables
        self._connect_step = 0
        self._connected = False
        self._connected_sites = set()
        self._connected_body1 = None
        self._connected_body1_pos = None
        self._connected_body1_quat = None
        self._num_connected = 0
        self._prev_num_connected = 0
        if self._agent_type == "Cursor":
            self._cursor_selected = [None, None]
        if self._num_connects is not None:
            self._success_num_conn = self._num_connects
            self._success_num_conn += len(self._preassembled)
        else:
            self._success_num_conn = len(self._object_names) - 1

        self._touched = {}
        self._picked = {}
        for body_id in self._object_body_ids:
            self._touched[body_id] = False
            self._picked[body_id] = False

        # initialize weld constraints
        eq_obj1id = self.sim.model.eq_obj1id
        eq_obj2id = self.sim.model.eq_obj2id
        p = self._preassembled  # list of weld equality ids to activate
        if len(p) > 0 and not self._recipe:
            for eq_id in p:
                self.sim.model.eq_active[eq_id] = 1
                object_body_id1 = eq_obj1id[eq_id]
                object_body_id2 = eq_obj2id[eq_id]
                object_name1 = self._object_body_id2name[object_body_id1]
                object_name2 = self._object_body_id2name[object_body_id2]
                self._merge_groups(object_name1, object_name2)
        elif eq_obj1id is not None:
            for i, (id1, id2) in enumerate(zip(eq_obj1id, eq_obj2id)):
                self.sim.model.eq_active[i] = 1 if self._config.assembled else 0

        if self._init_qpos:
            self.set_env_state(self._init_qpos)
            # enable robot collision
            for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
                body_name = self.sim.model.body_names[body_id]
                geom_name = self.sim.model.geom_id2name(geom_id)
                if (
                    body_name not in self._object_names
                    and self.mujoco_robot.is_robot_part(geom_name)
                ):
                    contype, conaffinity = robot_col[geom_name]
                    self.sim.model.geom_contype[geom_id] = contype
                    self.sim.model.geom_conaffinity[geom_id] = conaffinity
            self.sim.forward()
        else:
            if self.init_pos is None:
                self.init_pos, self.init_quat = self._place_objects()
            elif not self._config.fix_init:
                init_pos, init_quat = self._place_objects()
                self.init_pos.update(init_pos)
                self.init_quat.update(init_quat)
            # set furniture positions
            for i, body in enumerate(self._object_names):
                logger.debug(f"{body} {self.init_pos[body]} {self.init_quat[body]}")
                if self._config.assembled:
                    self._object_group[i] = 0
                else:
                    self._set_qpos(body, self.init_pos[body], self.init_quat[body])

            # stablize furniture pieces
            for _ in range(10):
                self._stop_objects(gravity=0)
                for i in range(10):
                    self.sim.forward()
                    self.sim.step()
                    self._slow_objects()

        if self._recipe:
            # preassemble furniture pieces
            for i in p:
                # move site1 to site2
                site1, site2 = self._recipe["site_recipe"][i][0:2]
                site1_id = self.sim.model.site_name2id(site1)
                site2_id = self.sim.model.site_name2id(site2)
                if len(self._recipe["site_recipe"][i]) == 3:
                    angle = self._recipe["site_recipe"][i][2]
                else:
                    angle = None
                self._target_connector_xquat = self._project_connector_quat(
                    site2, site1, angle
                )
                self._connect(site2_id, site1_id, auto_align=self._init_qpos is None)
                self._connected = False
                self._connected_body1 = None

            # stablize furniture pieces
            for _ in range(10):
                self._stop_objects(gravity=0)
                for i in range(10):
                    self.sim.forward()
                    self.sim.step()
                    self._slow_objects()

        if self._init_qpos:
            self.sim.forward()
        else:
            # gravity compensation
            if self._agent_type != "Cursor":
                self.sim.data.qfrc_applied[
                    self._ref_joint_vel_indexes_all
                ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes_all]
                self.sim.data.qfrc_applied[
                    self._ref_gripper_joint_vel_indexes_all
                ] = self.sim.data.qfrc_bias[self._ref_gripper_joint_vel_indexes_all]

            # set initial pose of an agent
            self._initialize_robot_pos()
            self.sim.forward()
            self.sim.step()

            # enable robot collision
            for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
                body_name = self.sim.model.body_names[body_id]
                geom_name = self.sim.model.geom_id2name(geom_id)
                if (
                    body_name not in self._object_names
                    and self.mujoco_robot.is_robot_part(geom_name)
                ):
                    contype, conaffinity = robot_col[geom_name]
                    self.sim.model.geom_contype[geom_id] = contype
                    self.sim.model.geom_conaffinity[geom_id] = conaffinity

            # gravity compensation
            if self._agent_type != "Cursor":
                self.sim.data.qfrc_applied[
                    self._ref_joint_vel_indexes_all
                ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes_all]
                self.sim.data.qfrc_applied[
                    self._ref_gripper_joint_vel_indexes_all
                ] = self.sim.data.qfrc_bias[self._ref_gripper_joint_vel_indexes_all]

            # stablize robot
            for _ in range(100):
                # set initial pose of an agent
                self._initialize_robot_pos()
                self.sim.forward()
                self.sim.step()

        # store qpos of furniture and robot
        if self._record_demo:
            self._store_state()

        if self._init_qpos:
            self.set_env_state(self._init_qpos)

        # sync mujoco sim state
        if self._agent_type != "Cursor":
            self.sim.data.ctrl[:] = 0
        self.sim.data.qfrc_applied[:] = 0
        self.sim.data.xfrc_applied[:] = 0
        self.sim.data.qacc[:] = 0
        self.sim.data.qacc_warmstart[:] = 0
        self.sim.data.time = 0
        self.sim.forward()

        # gravity compensation
        if self._agent_type != "Cursor":
            self.sim.data.qfrc_applied[
                self._ref_joint_vel_indexes_all
            ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes_all]
            self.sim.data.qfrc_applied[
                self._ref_gripper_joint_vel_indexes_all
            ] = self.sim.data.qfrc_bias[self._ref_gripper_joint_vel_indexes_all]

        for _ in range(100):
            self.sim.forward()
            self.sim.step()

        if self._agent_type not in ["Cursor"]:
            self._initial_right_hand_quat = self._right_hand_quat
            if self._agent_type == "Baxter":
                self._initial_left_hand_quat = self._left_hand_quat

            if self._control_type in ["ik", "ik_quaternion"]:
                # set up ik controller
                self._controller.sync_state()

        # set next subtask
        self._get_next_subtask()

        # set object positions in unity
        if self._unity:
            if background is None and self._background is None:
                background = self._config.background
            if self._config.background == "Random":
                background = self._rng.choice(background_names)
            if background and background != self._background:
                self._background = background
                self._unity.set_background(background)

    def _load_controller(self, controller_type, controller_file, kwargs):
        """
        Loads controller to be used for dynamic trajectories
        Controller_type is a specified controller, and controller_params is a config file containing the appropriate
        parameters for that controller
        Kwargs is kwargs passed from init call and represents individual params to override in controller config file
        """

        # Load the controller config file
        try:
            with open(controller_file) as f:
                params = hjson.load(f)
        except FileNotFoundError:
            logger.warn(
                "Controller config file '{}' not found. Please check filepath and try again.".format(
                    controller_file
                )
            )

        controller_params = params[controller_type]

        # Load additional arguments from kwargs and override the prior config-file loaded ones
        for key, value in kwargs.items():
            if key in controller_params:
                controller_params[key] = value

        self.controller = {}
        for arm in self._arms:
            if controller_type == ControllerType.POS:
                self.controller[arm] = PositionController(**controller_params)
            elif controller_type == ControllerType.POS_ORI:
                self.controller[arm] = PositionOrientationController(
                    **controller_params
                )
            elif controller_type == ControllerType.JOINT_IMP:
                self.controller[arm] = JointImpedanceController(**controller_params)
            elif controller_type == ControllerType.JOINT_TORQUE:
                self.controller[arm] = JointTorqueController(**controller_params)
            else:
                self.controller[arm] = JointVelocityController(**controller_params)

    def _pre_action(self, action, policy_step):
        """
        Overrides the superclass method to actuate the robot with the
        passed joint velocities and gripper control.
        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.mujoco_robot.dof dimensions should be the desired
                normalized joint velocities and if the robot has
                a gripper, the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
        """

        if self._control_type not in NEW_CONTROLLERS:
            return

        def apply_rescaled_action(indexes, input_action):
            ctrl_range = self.sim.model.actuator_ctrlrange[indexes]
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_action = bias + weight * input_action
            self.sim.data.ctrl[indexes] = applied_action

        # Split action into joint control and peripheral (i.e.: gripper) control (as specified by individual gripper)
        # Gripper action
        last = 0
        for arm in self._arms:
            last += self.controller[arm].control_dim
        for arm in self._arms:
            gripper_action_in = action[last : last + self.gripper[arm].dof]
            last = last + self.gripper[arm].dof
            gripper_action_actual = self.gripper[arm].format_action(gripper_action_in)
            apply_rescaled_action(
                self._ref_gripper_joint_vel_indexes[arm], gripper_action_actual
            )

        # Arm action
        last = 0
        for arm in self._arms:
            arm_action = action[last : last + self.controller[arm].control_dim]
            last += self.controller[arm].control_dim
            # First, get joint space action
            self.controller[arm].update_model(
                self.sim,
                id_name=arm + "_hand",
                joint_index=self._ref_joint_pos_indexes[arm],
            )
            torques = self.controller[arm].action_to_torques(
                arm_action, policy_step
            )  # this scales and clips the actions correctly

            # Now, control both gripper and joints
            self.sim.data.ctrl[self._ref_joint_vel_indexes[arm]] = (
                self.sim.data.qfrc_bias[self._ref_joint_vel_indexes[arm]] + torques
            )

    def _initialize_robot_pos(self):
        """
        Initializes robot posision with random noise perturbation.
        """
        if self._agent_type not in ["Cursor"]:
            noise = self._init_random(self.mujoco_robot.init_qpos.shape, "agent")
            self.sim.data.qpos[self._ref_joint_pos_indexes_all] = (
                self.mujoco_robot.init_qpos + noise
            )
            for arm in self._arms:
                self.sim.data.qpos[
                    self._ref_gripper_joint_pos_indexes[arm]
                ] = self.gripper[
                    arm
                ].init_qpos  # open

        elif self._agent_type == "Cursor":
            self._set_pos("cursor0", [-0.2, 0.0, self._move_speed / 2])
            self._set_pos("cursor1", [0.2, 0.0, self._move_speed / 2])

    def get_env_state(self):
        """
        Returns current qpos and qvel.
        """
        state = {
            "qpos": self.sim.data.qpos.copy(),
            "qvel": self.sim.data.qvel.copy(),
        }
        if self._agent_type == "Cursor":
            state["cursor0"] = self._get_pos("cursor0")
            state["cursor1"] = self._get_pos("cursor1")
        return state

    def set_env_state(self, given_state):
        self._stop_objects(gravity=0)
        self.sim.data.qpos[:] = given_state["qpos"]
        self.sim.data.qvel[:] = given_state["qvel"]
        self.sim.data.ctrl[:] = 0

        if "cursor0" in given_state:
            self._set_pos("cursor0", given_state["cursor0"])
        if "cursor1" in given_state:
            self._set_pos("cursor1", given_state["cursor1"])

    def _store_state(self):
        """
        Stores current qpos, qvel for demonstration.
        """
        state = self.get_env_state()
        self._demo.add(state=state)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # instantiate simulation from MJCF model
        self._load_model_robot()
        self._load_model_arena()
        self._load_model_object()
        self._load_model()

        # read recipe
        self._load_recipe()

        # write xml for unity viewer
        if self._unity:
            self._unity.change_model(
                xml=self.mujoco_model.get_xml(),
                camera_id=self._camera_ids[0],
                screen_width=self._screen_width,
                screen_height=self._screen_height,
            )

        logger.debug(self.mujoco_model.get_xml())

        # construct mujoco model from xml
        self.mjpy_model = self.mujoco_model.get_model(mode="mujoco_py")
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        self.initialize_time()

        self._is_render = self._visual_ob or self._render_mode != "no"
        if self._is_render:
            self._destroy_viewer()
            if self._camera_ids[0] == 0:
                # front view
                self._set_camera_position(self._camera_ids[0], [0.0, -0.7, 0.5])
                self._set_camera_rotation(self._camera_ids[0], [0.0, 0.0, 0.0])
            elif self._camera_ids[0] == 1:
                # side view
                self._set_camera_position(self._camera_ids[0], [-2.5, 0.0, 0.5])
                self._set_camera_rotation(self._camera_ids[0], [0.0, 0.0, 0.0])

        # additional housekeeping
        self._sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0

        # necessary to refresh MjData
        self.sim.forward()

        # setup mocap for ik control
        if (
            self._control_type in ["ik", "ik_quaternion"]
            and self._agent_type != "Cursor"
        ):
            from .models import assets_root

            if self._agent_type == "Sawyer":
                from .controllers import SawyerIKController as IKController
            elif self._agent_type == "Baxter":
                from .controllers import BaxterIKController as IKController
            elif self._agent_type == "Panda":
                from .controllers import PandaIKController as IKController
            elif self._agent_type == "Jaco":
                from .controllers import JacoIKController as IKController
            elif self._agent_type == "Fetch":
                from .controllers import FetchIKController as IKController
            else:
                raise ValueError

            self._controller = IKController(
                bullet_data_path=os.path.join(assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        elif self._control_type in NEW_CONTROLLERS:
            for arm in self._arms:
                self.controller[arm].reset()

    def _load_model_robot(self):
        """
        Loads sawyer, baxter, or cursor
        """
        use_torque = self._control_type in ["torque"] + NEW_CONTROLLERS
        if self._agent_type == "Sawyer":
            from .models.robots import Sawyer

            self.mujoco_robot = Sawyer(use_torque=use_torque)
            self.gripper = {"right": gripper_factory("TwoFingerGripper")}
            self.gripper["right"].hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper["right"])
            self.mujoco_robot.set_base_xpos([0, 0.65, -0.7])
            self.mujoco_robot.set_base_xquat([1, 0, 0, -1])

        elif self._agent_type == "Panda":
            from .models.robots import Panda

            self.mujoco_robot = Panda(use_torque=use_torque)
            self.gripper = {"right": gripper_factory("PandaGripper")}
            self.gripper["right"].hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper["right"])
            self.mujoco_robot.set_base_xpos([0, 0.65, -0.7])
            self.mujoco_robot.set_base_xquat([1, 0, 0, -1])

        elif self._agent_type == "Jaco":
            from .models.robots import Jaco

            self.mujoco_robot = Jaco(use_torque=use_torque)
            self.gripper = {"right": gripper_factory("JacoGripper")}
            self.gripper["right"].hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper["right"])
            self.mujoco_robot.set_base_xpos([0, 0.65, -0.7])
            self.mujoco_robot.set_base_xquat([1, 0, 0, -1])

        elif self._agent_type == "Baxter":
            from .models.robots import Baxter

            self.mujoco_robot = Baxter(use_torque=use_torque)
            self.gripper = {
                "right": gripper_factory("TwoFingerGripper"),
                "left": gripper_factory("LeftTwoFingerGripper"),
            }
            self.gripper["right"].hide_visualization()
            self.gripper["left"].hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper["right"])
            self.mujoco_robot.add_gripper("left_hand", self.gripper["left"])
            self.mujoco_robot.set_base_xpos([0, 0.65, -0.7])
            self.mujoco_robot.set_base_xquat([1, 0, 0, -1])

        elif self._agent_type == "Fetch":
            from .models.robots import Fetch

            self.mujoco_robot = Fetch(use_torque=use_torque)
            self.gripper = {"right": gripper_factory("FetchGripper")}
            self.gripper["right"].hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper["right"])
            self.mujoco_robot.set_base_xpos([0, 0.65, -0.7])
            self.mujoco_robot.set_base_xquat([1, 0, 0, -1])

        elif self._agent_type == "Cursor":
            from .models.robots import Cursor

            self.mujoco_robot = Cursor()
            self.mujoco_robot.set_size(self._move_speed / 2)
            self.mujoco_robot.set_xpos([0, 0, self._move_speed / 2])

        # hide an agent
        if not self._config.render_agent:
            for x in self.mujoco_robot.worldbody.findall(".//geom"):
                x.set("rgba", "0 0 0 0")

        # no collision with an agent
        if self._config.no_collision:
            for x in self.mujoco_robot.worldbody.findall(".//geom"):
                x.set("conaffinity", "0")
                x.set("contype", "0")

    def _load_model_arena(self):
        """
        Loads the arena XML
        """
        floor_full_size = (1.5, 1.0)
        floor_friction = (2.0, 0.005, 0.0001)
        from .models.arenas import FloorArena

        self.mujoco_arena = FloorArena(
            floor_full_size=floor_full_size, floor_friction=floor_friction
        )

    def _load_model_object(self):
        """
        Loads the object XMLs
        """
        # load pre_trained_models for objects
        path = xml_path_completion(furniture_xmls[self._furniture_id])
        logger.debug("load furniture %s" % path)
        resize_factor = None
        if self._manual_resize is not None:
            resize_factor = 1 + self._manual_resize
        elif self._config.furn_size_rand != 0:
            rand = self._init_random(1, "resize")[0]
            resize_factor = 1 + rand
        self._objects = MujocoXMLObject(path, debug=self._debug, resize=resize_factor)
        self._objects.hide_visualization()
        part_names = self._objects.get_children_names()

        # furniture pieces
        lst = []
        for part_name in part_names:
            lst.append((part_name, self._objects))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)
        self.mujoco_equality = self._objects.equality

    def _load_model(self):
        """
        loads initial furniture qpos from xml, if any then
        Loads the Task, which is composed of arena, robot, objects, equality
        """
        # task includes arena, robot, and objects of interest
        from .models.tasks import FloorTask

        init_qpos = next(iter(self.mujoco_objects.values())).get_init_qpos(
            list(self.mujoco_objects.keys())
        )
        if init_qpos:
            self.init_pos = {}
            self.init_quat = {}
            for key, qpos in init_qpos.items():
                self.init_pos[key] = [qpos.x, qpos.y, qpos.z]
                self.init_quat[key] = qpos.quat
        self.mujoco_model = FloorTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.mujoco_equality,
            self._config.furn_xyz_rand,
            self._config.furn_rot_rand,
            self._rng,
            init_qpos,
        )

    def _load_recipe(self):
        furniture_name = furniture_names[self._furniture_id]
        recipe_path = os.path.join(
            os.path.dirname(__file__),
            f"models/assets/recipes/{furniture_name}.yaml",
        )
        if os.path.exists(recipe_path):
            with open(recipe_path, "r") as stream:
                self._recipe = yaml.load(stream, Loader=PrettySafeLoader)
                self._site_recipe = self._recipe["site_recipe"]
        else:
            self._recipe = None

    def key_callback(self, window, key, scancode, action, mods):
        """
        Key listener for MuJoCo viewer
        """
        import glfw

        if action != glfw.RELEASE:
            return
        elif key == glfw.KEY_SPACE:
            action = "sel"
        elif key == glfw.KEY_ENTER:
            action = "des"
        elif key == glfw.KEY_W:
            action = "m_f"
        elif key == glfw.KEY_S:
            action = "m_b"
        elif key == glfw.KEY_E:
            action = "m_u"
        elif key == glfw.KEY_Q:
            action = "m_d"
        elif key == glfw.KEY_A:
            action = "m_l"
        elif key == glfw.KEY_D:
            action = "m_r"
        elif key == glfw.KEY_I:
            action = "r_f"
        elif key == glfw.KEY_K:
            action = "r_b"
        elif key == glfw.KEY_O:
            action = "r_u"
        elif key == glfw.KEY_U:
            action = "r_d"
        elif key == glfw.KEY_J:
            action = "r_l"
        elif key == glfw.KEY_L:
            action = "r_r"
        elif key == glfw.KEY_C:
            action = "connect"
        elif key == glfw.KEY_1:
            action = "switch1"
        elif key == glfw.KEY_2:
            action = "switch2"
        elif key == glfw.KEY_T:
            action = "screenshot"
        elif key == glfw.KEY_Y:
            action = "save"
        elif key == glfw.KEY_ESCAPE:
            action = "reset"
        else:
            return

        logger.info("Input action: %s" % action)
        self.action = action
        self._action_on = True

    def key_input_unity(self):
        """
        Key input for unity If adding new keys,
        make sure to add keys to whitelist in MJTCPInterace.cs
        """
        key = self._unity.get_input()
        if key == "None":
            return
        elif key == "Space":
            action = "sel"
        elif key == "Return":
            action = "des"
        elif key == "W":
            action = "m_f"
        elif key == "S":
            action = "m_b"
        elif key == "E":
            action = "m_u"
        elif key == "Q":
            action = "m_d"
        elif key == "A":
            action = "m_l"
        elif key == "D":
            action = "m_r"
        elif key == "I":
            action = "r_f"
        elif key == "K":
            action = "r_b"
        elif key == "O":
            action = "r_u"
        elif key == "U":
            action = "r_d"
        elif key == "J":
            action = "r_l"
        elif key == "L":
            action = "r_r"
        elif key == "C":
            action = "connect"
        elif key == "Alpha1":
            action = "switch1"
        elif key == "Alpha2":
            action = "switch2"
        elif key == "T":
            action = "screenshot"
        elif key == "Y":
            action = "save"
        elif key == "Escape":
            action = "reset"
        else:
            return

        logger.info("Input action: %s" % action)
        self.action = action
        self._action_on = True

    def resize_key_input_unity(self):
        """
        Key input for unity If adding new keys,
        make sure to add keys to whitelist in MJTCPInterace.cs
        """
        key = self._unity.get_input()
        if key == "None":
            return
        elif key == "Q":
            action = "smaller"
        elif key == "W":
            action = "fine_smaller"
        elif key == "E":
            action = "fine_larger"
        elif key == "R":
            action = "larger"
        elif key == "Y":
            action = "save"
        elif key == "Escape":
            action = "reset"
        else:
            return

        logger.info("Input action: %s" % action)
        self.action = action
        self._action_on = True

    def run_demo(self, config=None):
        """
        Since we save all states, just play back states
        """
        if config is None:
            config = self._config
        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        self.reset(config.furniture_id, config.background)
        if self._record_vid:
            self.vid_rec.capture_frame(self.render("rgb_array")[0])
        else:
            self.render("rgb_array")[0]
        with open(config.load_demo, "rb") as f:
            demo = pickle.load(f)
            all_states = demo["state"]
            if config.debug:
                for i, (obs, action) in enumerate(zip(demo["obs"], demo["actions"])):
                    logger.debug("action", i, action)
        try:
            for state in all_states:
                self.set_env_state(state)
                self.sim.forward()
                if self._unity:
                    self._update_unity()
                if self._record_vid:
                    self.vid_rec.capture_frame(self.render("rgb_array")[0])
                else:
                    self.render("rgb_array")[0]

        finally:
            if self._record_vid:
                self.vid_rec.close()

    def get_vr_input(self, controller):
        c = self.vr.devices[controller]
        if controller not in self.vr.devices:
            logger.warn("Lost track of ", controller)
            return None, None
        # pose = c.get_pose_euler()
        pose = c.get_pose_quaternion()
        # match rotation in sim and vr controller
        if self._control_type == "ik":
            pose[3:] = T.euler_to_quat([0, 0, -90], pose[3:])
        else:
            pose[3:] = T.euler_to_quat([0, 0, 180], pose[3:])
        state = c.get_controller_inputs()
        if pose is None or state is None or np.linalg.norm(pose[:3]) < 0.001:
            logger.warn("Lost track of pose ", controller)
            return None, None
        return np.asarray(pose), state

    def run_vr(self, config=None):
        """
        Runs the environment with HTC Vive support
        """
        from ..util.triad_openvr import triad_openvr

        self.vr = triad_openvr.triad_openvr()
        self.vr.print_discovered_objects()

        if config is None:
            config = self._config
        assert config.render, "Set --render True to see the viewer"

        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        self.reset(config.furniture_id, config.background)

        if self._record_vid:
            self.vid_rec.capture_frame(self.render("rgb_array")[0])
        if config.render:
            self.render()

        # set initial pose of controller as origin
        origin_vr_pos = {}
        origin_vr_quat = {}
        origin_sim_pos = {}
        origin_sim_quat = {}
        flag = {}

        def init_origin():
            for i, arm in enumerate(self._arms):
                logger.warn("Initialize %s VR controller", arm)
                while True:
                    origin_pose, origin_state = self.get_vr_input(
                        "controller_%d" % (i + 1)
                    )
                    if origin_pose is None or origin_state is None:
                        time.sleep(0.1)
                    else:
                        break
                origin_vr_pos[arm] = origin_pose[:3].copy()
                origin_vr_quat[arm] = origin_pose[3:].copy()
                origin_sim_pos[arm] = self.sim.data.get_body_xpos(
                    "%s_hand" % arm
                ).copy()
                origin_sim_quat[arm] = self.sim.data.get_body_xquat(
                    "%s_hand" % arm
                ).copy()
                flag[arm] = -1

        def rel_pos(a, b):
            return a - b

        def rel_quat(a, b):
            return np.array(list(Quaternion(a).inverse * Quaternion(b)))

        def quat_to_rot(quat):
            rot = np.array(
                T.quaternion_to_euler(*T.convert_quat(np.array(quat), to="xyzw"))
            )
            # swap rotation axes
            rot[1] = -rot[1]
            rot[0], rot[2] = -rot[2], rot[0]

            if abs(rot[0]) < 1:
                rot[0] = 0
            if abs(rot[1]) < 1:
                rot[1] = 0
            if abs(rot[2]) < 1:
                rot[2] = 0
            return rot

        def get_action(pose, arm):
            rel_vr_pos = rel_pos(origin_vr_pos[arm], pose[:3])
            # relative movement speed between VR and simulation
            rel_vr_pos *= 2.5
            # swap y, z axes
            rel_vr_pos[1], rel_vr_pos[2] = -rel_vr_pos[2], rel_vr_pos[1]
            rel_vr_quat = rel_quat(origin_vr_quat[arm], pose[3:])  # wxyz

            sim_pos = self.sim.data.get_body_xpos("%s_hand" % arm).copy()
            sim_quat = self.sim.data.get_body_xquat("%s_hand" % arm).copy()  # wxyz
            rel_sim_pos = rel_pos(origin_sim_pos[arm], sim_pos)
            rel_sim_quat = rel_quat(origin_sim_quat[arm], sim_quat)  # wxyz

            action_pos = rel_pos(rel_sim_pos, rel_vr_pos)
            action_quat = rel_quat(rel_sim_quat, rel_vr_quat)  # wxyz
            action_rot = quat_to_rot(action_quat)

            if self._control_type == "ik":
                return action_pos, action_rot
            elif self._control_type == "ik_quaternion":
                return action_pos, action_quat

        t = 0
        connect = -1
        init_origin()
        while True:
            pose = {}
            state = {}
            action_pos = {}
            action_rot = {}
            for i, arm in enumerate(self._arms):
                while True:
                    # get pose of the vr
                    pose[arm], state[arm] = self.get_vr_input("controller_%d" % (i + 1))
                    # check if controller is connected
                    if pose[arm] is None or state[arm] is None:
                        time.sleep(0.1)
                    else:
                        break
                action_pos[arm], action_rot[arm] = get_action(pose[arm], arm)

            if config.render:
                self.render()

            reset = False
            connect = -1
            for arm in self._arms:
                s = state[arm]
                # select
                if s["trigger"] > 0.01:
                    flag[arm] = 1
                else:
                    flag[arm] = -1

                # connect
                if s["trackpad_pressed"] != 0:
                    connect = 1

                # reset
                if s["grip_button"] != 0:
                    reset = True

            if reset:
                t = 0
                connect = -1
                if self._record_demo:
                    self._demo.save(self.file_prefix)
                self.reset(config.furniture_id, config.background)
                init_origin()
                continue

            action_items = []
            for arm in self._arms:
                action_items.append(action_pos[arm])
                if self._control_type == "ik":
                    action_items.append(action_rot[arm] / self._rotate_speed)
                else:
                    action_items.append(action_rot[arm])
            for arm in self._arms:
                action_items.append([flag[arm]])
            action_items.append([connect])

            action = np.hstack(action_items)
            action = np.clip(action, -1.0, 1.0)

            logger.info(str(t) + " Take action: " + str(action))
            ob, reward, done, info = self.step(action)

            if self._record_vid:
                self.vid_rec.capture_frame(self.render("rgb_array")[0])
            t += 1
            if done:
                if self._record_demo:
                    self._demo.save(self.file_prefix)
                self.reset(config.furniture_id, config.background)
                if self._record_vid:
                    self.vid_rec.capture_frame(self.render("rgb_array")[0])
                t = 0
                connect = -1
                init_origin()

            time.sleep(0.05)

    def run_manual(self, config=None):
        """
        Run the environment under manual (keyboard) control
        """
        if config is None:
            config = self._config
        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        self.reset(config.furniture_id, config.background)

        if self._record_vid:
            self.vid_rec.capture_frame(self.render("rgb_array")[0])
        else:
            self.render()

        if not config.unity:
            # override keyboard callback function of viewer
            import glfw

            assert self._config.render, "Set --render True for manual control"
            glfw.set_key_callback(self._get_viewer().window, self.key_callback)

        cursor_idx = 0
        flag = [-1, -1]
        t = 0
        try:
            while True:
                if config.unity:
                    self.key_input_unity()

                if not self._action_on:
                    self.render()
                    time.sleep(0.01)
                    continue

                action = np.zeros((8,))
                if self.action == "reset":
                    self.reset()
                    if self._config.record_vid:
                        self.vid_rec.capture_frame(self.render("rgb_array")[0])
                    else:
                        self.render()
                    self._action_on = False
                    continue

                if self.action == "switch1":
                    cursor_idx = 0
                    self._action_on = False
                    continue
                if self.action == "switch2":
                    cursor_idx = 1
                    self._action_on = False
                    continue

                # pick
                if self.action == "sel":
                    flag[cursor_idx] = 1
                if self.action == "des":
                    flag[cursor_idx] = -1

                # connect
                if self.action == "connect":
                    action[7] = 1

                # move
                if self.action == "m_f":
                    action[1] = 1
                if self.action == "m_b":
                    action[1] = -1
                if self.action == "m_u":
                    action[2] = 1
                if self.action == "m_d":
                    action[2] = -1
                if self.action == "m_l":
                    action[0] = -1
                if self.action == "m_r":
                    action[0] = 1
                # rotate
                if self.action == "r_f":
                    action[4] = 1
                if self.action == "r_b":
                    action[4] = -1
                if self.action == "r_u":
                    action[5] = 1
                if self.action == "r_d":
                    action[5] = -1
                if self.action == "r_l":
                    action[3] = -1
                if self.action == "r_r":
                    action[3] = 1

                if self._agent_type == "Cursor":
                    if cursor_idx:
                        action = np.hstack(
                            [
                                np.zeros_like(action[:6]),
                                [flag[0]],
                                action[:6],
                                [flag[1], action[7]],
                            ]
                        )
                    else:
                        action = np.hstack(
                            [
                                action[:6],
                                [flag[0]],
                                np.zeros_like(action[:6]),
                                [flag[1], action[7]],
                            ]
                        )
                elif self._control_type in ["ik", "position_orientation"]:
                    if self._agent_type in ["Sawyer", "Panda", "Jaco", "Fetch"]:
                        action = action[:8]
                        action[6] = flag[0]
                    elif self._agent_type == "Baxter":
                        if cursor_idx:
                            action = np.hstack(
                                [np.zeros(6), action[:6], [flag[0], flag[1], action[7]]]
                            )
                        else:
                            action = np.hstack(
                                [action[:6], np.zeros(6), [flag[0], flag[1], action[7]]]
                            )

                logger.info(f"Action: {action}")
                ob, reward, done, info = self.step(action)

                if self._record_vid:
                    self.vid_rec.capture_frame(self.render("rgb_array")[0])
                else:
                    self.render()
                if self.action == "screenshot":
                    import imageio

                    img, depth = self.render("rgbd_array")

                    if len(img.shape) == 4:
                        img = np.concatenate(img)
                        if depth is not None:
                            depth = np.concatenate(depth)

                    imageio.imwrite(config.furniture_name + ".png", img)
                    if self._segmentation_ob:
                        seg = self.render("segmentation")
                        if len(seg.shape) == 4:
                            seg = np.concatenate(seg)
                        color_seg = color_segmentation(seg)
                        imageio.imwrite("segmentation_ob.png", color_seg)

                    if self._depth_ob:
                        imageio.imwrite("depth_ob.png", depth)

                if self.action == "save" and self._record_demo:
                    self._demo.save(self.file_prefix)

                self._action_on = False
                t += 1
                if done:
                    t = 0
                    flag = [-1, -1]
                    if self._record_demo:
                        self._demo.save(self.file_prefix)
                    self.reset(config.furniture_id, config.background)
                    if self._record_vid:
                        self.vid_rec.capture_frame(self.render("rgb_array")[0])
                    else:
                        self.render()
        finally:
            if self._record_vid:
                self.vid_rec.close()

    def run_demo_actions(self, config=None):
        """
        Play the stored actions in demonstration
        """
        if config is None:
            config = self._config
        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        self.reset(config.furniture_id, config.background)
        if self._record_vid:
            self.vid_rec.capture_frame(self.render("rgb_array")[0])
        elif self._config.render:
            self.render()

        # Load demo
        with open(config.load_demo, "rb") as f:
            demo = pickle.load(f)
            actions = demo["actions"]
            low_level_actions = demo["low_level_actions"]

        try:
            i = 0
            if self._control_type == "impedance":
                for action in low_level_actions:
                    logger.info("Action: %s", str(action))
                    ob, _, _, _ = self.step(action)
                    if self._record_vid:
                        self.vid_rec.capture_frame(self.render("rgb_array")[0])
                    elif self._config.render:
                        self.render()
                        time.sleep(0.03)
            else:
                for action in actions:
                    # logger.info("Action: %s", str(action))
                    ob, _, _, _ = self.step(action)
                    if self._record_vid:
                        self.vid_rec.capture_frame(self.render("rgb_array")[0])
                    elif self._config.render:
                        self.render()
                        time.sleep(0.03)

        finally:
            if self._record_vid:
                self.vid_rec.close()

    def run_resizer(self, config=None):
        """
        Run a resizing program in unity for adjusting furniture size in xml
        """
        self._manual_resize = 0
        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        ob = self.reset(config.furniture_id, config.background)
        self.render()
        cursor_idx = 0
        flag = [-1, -1]
        t = 0
        while True:
            if config.unity:
                self.resize_key_input_unity()

            if not self._action_on:
                time.sleep(0.1)
                continue

            if self.action == "reset":
                self.reset()
                self.render()
                self._action_on = False
                continue
            # move
            if self.action == "smaller":
                self._manual_resize -= 0.1
            if self.action == "fine_smaller":
                self._manual_resize -= 0.02
            if self.action == "fine_larger":
                self._manual_resize += 0.02
            if self.action == "larger":
                self._manual_resize += 0.1
            if self.action == "save":
                path = xml_path_completion(furniture_xmls[self._furniture_id])
                next(iter(self.mujoco_objects.values())).save_model(path)
                return
            self.render("rgb_array")
            action = np.zeros((15,))
            ob, reward, done, info = self.step(action)
            self.render("rgb_array")
            logger.info("current_scale: " + str(1 + self._manual_resize))
            self.reset(config.furniture_id, config.background)
            self._action_on = False

    def run_img(self, config=None):
        """
        Run a resizing program in unity for adjusting furniture size in xml
        """
        flag = [-1, -1]
        n_img = 5
        grid = np.zeros((n_img, 3, self._screen_height, self._screen_width))
        blended = np.zeros((3, self._screen_height, self._screen_width))
        for i in range(n_img):
            self.reset()
            grid[i] = np.transpose((self.render("rgb_array")[0]), (2, 0, 1))
            blended += grid[i]
        blended = blended / n_img
        path = "randomness_distribution"
        blended_img_path = os.path.join(
            path, furniture_names[self._furniture_id] + "_blended" + str(n_img) + ".jpg"
        )
        grid_img_path = os.path.join(
            path, furniture_names[self._furniture_id] + "_grid" + str(n_img) + ".jpg"
        )
        from ..util.pytorch import save_distribution_imgs

        save_distribution_imgs(grid, blended, grid_img_path, blended_img_path)

    def _get_reference(self):
        """
        Store ids / keys of objects, connector sites, and collision data in the scene
        """
        self._object_body_id = {}
        self._object_body_id2name = {}
        for obj_str in self.mujoco_objects.keys():
            self._object_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
            self._object_body_id2name[self.sim.model.body_name2id(obj_str)] = obj_str

        # for checking distance to / contact with objects we want to pick up
        self._object_body_ids = list(map(int, self._object_body_id.values()))

        # information of objects
        self._object_names = list(self.mujoco_objects.keys())
        self._object_name2id = {k: i for i, k in enumerate(self._object_names)}
        self._object_group = list(range(len(self._object_names)))
        self._object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self._object_names
        ]

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

    def _get_next_subtask(self):
        eq_obj1id = self.sim.model.eq_obj1id
        if eq_obj1id is not None:
            for i, (id1, id2) in enumerate(
                zip(self.sim.model.eq_obj1id, self.sim.model.eq_obj2id)
            ):
                object_name1 = self._object_body_id2name[id1]
                object_name2 = self._object_body_id2name[id2]
                if self._find_group(object_name1) != self._find_group(object_name2):
                    self._subtask_part1 = self._object_name2id[object_name1]
                    self._subtask_part2 = self._object_name2id[object_name2]
                    return
        self._subtask_part1 = -1
        self._subtask_part2 = -1

    def _find_group(self, idx):
        """
        Finds the group of the object
        """
        if isinstance(idx, str):
            idx = self._object_name2id[idx]
        if self._object_group[idx] == idx:
            return idx
        self._object_group[idx] = self._find_group(self._object_group[idx])
        return self._object_group[idx]

    def _merge_groups(self, idx1, idx2):
        """
        Merges two groups into one
        """
        if isinstance(idx1, str):
            idx1 = self._object_name2id[idx1]
        if isinstance(idx2, str):
            idx2 = self._object_name2id[idx2]
        p_idx1 = self._find_group(idx1)
        p_idx2 = self._find_group(idx2)
        self._object_group[p_idx1] = p_idx2

    def _activate_weld(self, part1, part2):
        """
        Turn on weld constraint between two parts
        """
        for i, (id1, id2) in enumerate(
            zip(self.sim.model.eq_obj1id, self.sim.model.eq_obj2id)
        ):
            p1 = self.sim.model.body_id2name(id1)
            p2 = self.sim.model.body_id2name(id2)
            if p1 in [part1, part2] and p2 in [part1, part2]:
                # setup eq_data
                self.sim.model.eq_data[i] = T.rel_pose(
                    self._get_qpos(p1), self._get_qpos(p2)
                )
                self.sim.model.eq_active[i] = 1
                self._merge_groups(part1, part2)

    def _stop_object(self, obj_name, gravity=1):
        """
        Stops object by removing force and velocity. If gravity=1, then
        it compensates for gravity.
        """
        body_id = self.sim.model.body_name2id(obj_name)
        self.sim.data.xfrc_applied[body_id] = [
            0,
            0,
            -gravity
            * self.sim.model.opt.gravity[-1]
            * self.sim.model.body_mass[body_id],
            0,
            0,
            0,
        ]
        qvel_addr = self.sim.model.get_joint_qvel_addr(obj_name)
        self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = [0] * (
            qvel_addr[1] - qvel_addr[0]
        )
        self.sim.data.qfrc_applied[qvel_addr[0] : qvel_addr[1]] = [0] * (
            qvel_addr[1] - qvel_addr[0]
        )

    def _stop_objects(self, gravity=1):
        """
        Stop all objects
        """
        for obj_name in self._object_names:
            self._stop_object(obj_name, gravity)

    def _stop_selected_objects(self, gravity=1):
        """
        Stops all objects selected by cursor
        """
        selected_idx = []
        for obj_name in self._cursor_selected:
            if obj_name is not None:
                selected_idx.append(self._find_group(obj_name))
        for obj_name in self._object_names:
            if self._find_group(obj_name) in selected_idx:
                self._stop_object(obj_name, gravity)

    def _slow_object(self, obj_name):
        """
        Slows object by clipping qvelocity
        """
        body_id = self.sim.model.body_name2id(obj_name)
        self.sim.data.xfrc_applied[body_id] = [
            0,
            0,
            -self.sim.model.opt.gravity[-1] * self.sim.model.body_mass[body_id],
            0,
            0,
            0,
        ]
        qvel_addr = self.sim.model.get_joint_qvel_addr(obj_name)
        self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = np.clip(
            self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]], -0.2, 0.2
        )
        self.sim.data.qfrc_applied[qvel_addr[0] : qvel_addr[1]] = [0] * (
            qvel_addr[1] - qvel_addr[0]
        )

    def _slow_objects(self):
        """
        Slow all objects
        """
        for obj_name in self._object_names:
            self._slow_object(obj_name)

    def initialize_time(self):
        """
        Initializes the time constants used for simulation.
        """
        self._cur_time = 0
        self._model_timestep = self.sim.model.opt.timestep
        self._control_timestep = 1.0 / self._control_freq

    def _do_simulation(self, a):
        """
        Take multiple physics simulation steps, bounded by self._control_timestep
        """
        try:
            if self.sim.data.ctrl is not None:
                self.sim.data.ctrl[:] = 0 if a is None else a

            if self._agent_type == "Cursor":
                # gravity compensation
                selected_idx = []
                for obj_name in self._cursor_selected:
                    if obj_name is not None:
                        selected_idx.append(self._find_group(obj_name))
                for obj_name in self._object_names:
                    if self._find_group(obj_name) in selected_idx:
                        self._stop_object(obj_name, gravity=1)
                    else:
                        self._stop_object(obj_name, gravity=0)

            self.sim.forward()
            for _ in range(int(self._control_timestep / self._model_timestep)):
                self.sim.step()

            self._cur_time += self._control_timestep

            if self._agent_type == "Cursor":
                # gravity compensation
                for obj_name in self._object_names:
                    if self._find_group(obj_name) in selected_idx:
                        self._stop_object(obj_name, gravity=1)

        except Exception as e:
            logger.warn(
                "[!] Warning: Simulation is unstable. The episode is terminated."
            )
            logger.warn(e)
            logger.warn(type(e))
            self.set_init_qpos(None)
            self.reset()
            self._fail = True

    def _do_ik_step(self, action):
        """
        Take multiple physics simulation steps, bounded by self._control_timestep
        """
        # stop moving arm for IK control, critical for BC with IK
        # for arm in self._arms:
        #     for qvel_addr in self._ref_joint_vel_indexes[arm]:
        #         self.sim.data.qvel[qvel_addr] = 0.0
        # self.sim.forward()

        connect = action[-1]

        if self._control_type == "ik":
            if self._agent_type in ["Sawyer", "Panda", "Jaco", "Fetch"]:
                action[:3] = action[:3] * self._move_speed
                action[:3] = [-action[1], action[0], action[2]]
                gripper_pos = self.sim.data.get_body_xpos("right_hand")
                d_pos = self._bounded_d_pos(action[:3], gripper_pos)
                self._initial_right_hand_quat = T.euler_to_quat(
                    action[3:6] * self._rotate_speed, self._initial_right_hand_quat
                )
                d_quat = T.quat_multiply(
                    T.quat_inverse(self._right_hand_quat), self._initial_right_hand_quat
                )
                gripper_dis = action[-2]
                action = np.concatenate([d_pos, d_quat, [gripper_dis]])

            elif self._agent_type == "Baxter":
                action[:3] = action[:3] * self._move_speed
                action[:3] = [-action[1], action[0], action[2]]
                action[6:9] = action[6:9] * self._move_speed
                action[6:9] = [-action[7], action[6], action[8]]
                right_gripper_pos = self.sim.data.get_body_xpos("right_hand")
                right_d_pos = self._bounded_d_pos(action[:3], right_gripper_pos)
                self._initial_right_hand_quat = T.euler_to_quat(
                    action[3:6] * self._rotate_speed, self._initial_right_hand_quat
                )
                right_d_quat = T.quat_multiply(
                    T.quat_inverse(self._right_hand_quat), self._initial_right_hand_quat
                )

                right_gripper_dis = action[-3]
                left_gripper_pos = self.sim.data.get_body_xpos("left_hand")
                left_d_pos = self._bounded_d_pos(action[6:9], left_gripper_pos)
                self._initial_left_hand_quat = T.euler_to_quat(
                    action[9:12] * self._rotate_speed, self._initial_left_hand_quat
                )
                left_d_quat = T.quat_multiply(
                    T.quat_inverse(self._left_hand_quat), self._initial_left_hand_quat
                )
                left_gripper_dis = action[-2]
                action = np.concatenate(
                    [
                        right_d_pos,
                        right_d_quat,
                        left_d_pos,
                        left_d_quat,
                        [right_gripper_dis, left_gripper_dis],
                    ]
                )

            input_1 = self._make_input(action[:7], self._right_hand_quat)
            if self._agent_type in ["Sawyer", "Panda", "Fetch"]:
                velocities = self._controller.get_control(**input_1)
                low_action = np.concatenate([velocities, action[7:8]])
            elif self._agent_type == "Jaco":
                velocities = self._controller.get_control(**input_1)
                low_action = np.concatenate([velocities] + [action[7:]] * 3)
            elif self._agent_type == "Baxter":
                input_2 = self._make_input(action[7:14], self._left_hand_quat)
                velocities = self._controller.get_control(input_1, input_2)
                low_action = np.concatenate([velocities, action[14:16]])
            else:
                raise Exception(
                    "Only Sawyer, Panda, Jaco, Baxter robot environments are supported for IK "
                    "control currently."
                )

            # keep trying to reach the target in a closed-loop
            ctrl = self._setup_action(low_action)
            for i in range(self._action_repeat):
                self._do_simulation(ctrl)
                if self._record_demo:
                    self._demo.add(
                        low_level_ob=self._get_obs(include_qpos=True),
                        low_level_action=np.clip(low_action, -1, 1),
                        connect_action=connect if i == self._action_repeat - 1 else 0,
                    )

                if i + 1 < self._action_repeat:
                    velocities = self._controller.get_control()
                    if self._agent_type in ["Sawyer", "Panda", "Fetch"]:
                        low_action = np.concatenate([velocities, action[7:]])
                    elif self._agent_type == "Jaco":
                        low_action = np.concatenate([velocities] + [action[7:]] * 3)
                    elif self._agent_type == "Baxter":
                        low_action = np.concatenate([velocities, action[14:]])
                    ctrl = self._setup_action(low_action)

        elif self._control_type == "ik_quaternion":
            arm_pos = {}
            arm_quat = {}
            gripper_action = {}
            last = 0
            for arm in self._arms:
                d_pos = action[last : last + 3] * self._move_speed
                d_pos = [-d_pos[1], d_pos[0], d_pos[2]]
                gripper_pos = self.sim.data.get_body_xpos("%s_hand" % arm)
                d_pos = self._bounded_d_pos(d_pos, gripper_pos)
                arm_pos[arm] = d_pos
                arm_quat[arm] = T.convert_quat(action[last + 3 : last + 7])
                last += 7

            for arm in self._arms:
                gripper_action[arm] = [action[last]]
                last += 1

            action_items = []
            for arm in self._arms:
                action_items.append(arm_pos[arm])
                action_items.append(arm_quat[arm])
            action = np.hstack(action_items)

            action_items = []
            for arm in self._arms:
                action_items.append(gripper_action[arm])
            gripper_action = np.hstack(action_items)

            input_1 = self._make_input(action[:7], self._right_hand_quat)
            if self._agent_type in ["Sawyer", "Panda", "Fetch"]:
                velocities = self._controller.get_control(**input_1)
                low_action = np.concatenate([velocities, gripper_action])
            elif self._agent_type == "Jaco":
                velocities = self._controller.get_control(**input_1)
                low_action = np.concatenate([velocities] + [gripper_action] * 3)
            elif self._agent_type == "Baxter":
                input_2 = self._make_input(action[7:14], self._left_hand_quat)
                velocities = self._controller.get_control(input_1, input_2)
                low_action = np.concatenate([velocities, gripper_action])
            else:
                raise Exception(
                    "Only Sawyer, Panda, Jaco, Baxter robot environments are supported for IK "
                    "control currently."
                )

            # keep trying to reach the target in a closed-loop
            ctrl = self._setup_action(low_action)
            for i in range(self._action_repeat):
                self._do_simulation(ctrl)
                if self._record_demo:
                    self._demo.add(
                        low_level_ob=self._get_obs(include_qpos=True),
                        low_level_action=np.clip(low_action, -1, 1),
                        connect_action=connect if i == self._action_repeat - 1 else 0,
                    )

                if i + 1 < self._action_repeat:
                    velocities = self._controller.get_control()
                    if self._agent_type in ["Sawyer", "Panda", "Fetch"]:
                        low_action = np.concatenate([velocities, gripper_action])
                    elif self._agent_type == "Jaco":
                        low_action = np.concatenate([velocities] + [gripper_action] * 3)
                    elif self._agent_type == "Baxter":
                        low_action = np.concatenate([velocities, gripper_action])
                    ctrl = self._setup_action(low_action)

    def _do_controller_step(self, action):
        """
        Take multiple physics simulation steps, bounded by self._control_timestep
        """
        if self._agent_type in ["Sawyer", "Panda", "Jaco", "Fetch"]:
            action[:3] = action[:3] * self._move_speed
            action[:3] = [-action[1], action[0], action[2]]
        elif self._agent_type == "Baxter":
            action[:3] = action[:3] * self._move_speed
            action[:3] = [-action[1], action[0], action[2]]
            action[6:9] = action[6:9] * self._move_speed
            action[6:9] = [-action[7], action[6], action[8]]

        try:
            self.sim.forward()
            for i in range(int(self._control_timestep / self._model_timestep)):
                self._pre_action(action, policy_step=(i == 0))
                self.sim.step()

            self._cur_time += self._control_timestep

        except Exception as e:
            logger.warn(
                "[!] Warning: Simulation is unstable. The episode is terminated."
            )
            logger.warn(e)
            logger.warn(type(e))
            self.reset()
            self._fail = True

    def set_state(self, qpos, qvel):
        """
        Sets the qpos and qvel of the MuJoCo sim
        """
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    def _get_cursor_pos(self, name=None):
        """
        Returns the cursor positions
        """
        if self._agent_type in ["Sawyer", "Panda", "Jaco", "Baxter", "Fetch"]:
            return self.sim.data.site_xpos[self.eef_site_id["right"]].copy()
        elif self._agent_type == "Cursor":
            if name:
                return self._get_pos(name)
            else:
                return np.hstack([self._get_pos("cursor0"), self._get_pos("cursor1")])
        else:
            return None

    def _get_pos(self, name):
        """
        Get the position of a site, body, or geom
        """
        if name in self.sim.model.body_names:
            return self.sim.data.get_body_xpos(name).copy()
        if name in self.sim.model.geom_names:
            return self.sim.data.get_geom_xpos(name).copy()
        if name in self.sim.model.site_names:
            return self.sim.data.get_site_xpos(name).copy()
        raise ValueError

    def _set_pos(self, name, pos):
        """
        Set the position of a body or geom
        """
        if name in self.sim.model.body_names:
            body_idx = self.sim.model.body_name2id(name)
            self.sim.model.body_pos[body_idx] = pos[:].copy()
            return
        if name in self.sim.model.geom_names:
            geom_idx = self.sim.model.geom_name2id(name)
            self.sim.model.geom_pos[geom_idx][0:3] = pos[:].copy()
            return
        raise ValueError

    def _get_quat(self, name):
        """
        Get the quaternion of a body, geom, or site
        """
        if name in self.sim.model.body_names:
            body_idx = self.sim.model.body_name2id(name)
            return self.sim.data.body_xquat[body_idx].copy()
        if name in self.sim.model.geom_names:
            geom_idx = self.sim.model.geom_name2id(name)
            return self.sim.model.geom_quat[geom_idx].copy()
        if name in self.sim.model.site_names:
            site_idx = self.sim.model.site_name2id(name)
            return self.sim.model.site_quat[site_idx].copy()
        raise ValueError

    def _set_quat(self, name, quat):
        """
        Set the quaternion of a body
        """
        if name in self.sim.model.body_names:
            body_idx = self.sim.model.body_name2id(name)
            self.sim.model.body_quat[body_idx][0:4] = quat[:]
            return
        raise ValueError

    def _get_left_vector(self, name):
        """
        Get the left vector of a geom, or site
        """
        if name in self.sim.model.geom_names:
            return self.sim.data.get_geom_xmat(name)[:, 0].copy()
        if name in self.sim.model.site_names:
            return self.sim.data.get_site_xmat(name)[:, 0].copy()
        raise ValueError

    def _get_forward_vector(self, name):
        """
        Get the forward vector of a geom, or site
        """
        if name in self.sim.model.geom_names:
            return self.sim.data.get_geom_xmat(name)[:, 1].copy()
        if name in self.sim.model.site_names:
            return self.sim.data.get_site_xmat(name)[:, 1].copy()
        raise ValueError

    def _get_up_vector(self, name):
        """
        Get the up vector of a geom, or site
        """
        if name in self.sim.model.geom_names:
            return self.sim.data.get_geom_xmat(name)[:, 2].copy()
        if name in self.sim.model.site_names:
            return self.sim.data.get_site_xmat(name)[:, 2].copy()
        raise ValueError

    def _get_distance(self, name1, name2):
        """
        Get the distance vector of a body, geom, or site
        """
        pos1 = self._get_pos(name1)
        pos2 = self._get_pos(name2)
        return np.linalg.norm(pos1 - pos2)

    def _get_size(self, name):
        """
        Get the size of a body
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                return self.sim.model.geom_size[geom_idx, :].copy()
        raise ValueError

    def _set_size(self, name, size):
        """
        Set the size of a body
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.sim.model.geom_size[geom_idx, :] = size
                return
        raise ValueError

    def _get_geom_type(self, name):
        """
        Get the type of a geometry
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                return self.sim.model.geom_type[geom_idx].copy()

    def _set_geom_type(self, name, geom_type):
        """
        Set the type of a geometry
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.sim.model.geom_type[geom_idx] = geom_type

    def _get_qpos(self, name):
        """
        Get the qpos of a joint
        """
        object_qpos = self.sim.data.get_joint_qpos(name)
        return object_qpos.copy()

    def _set_qpos(self, name, pos, rot=[1, 0, 0, 0]):
        """
        Set the qpos of a joint
        """
        object_qpos = self.sim.data.get_joint_qpos(name)
        assert object_qpos.shape == (7,)
        object_qpos[:3] = pos
        object_qpos[3:] = rot
        self.sim.data.set_joint_qpos(name, object_qpos)

    def _set_qpos0(self, name, qpos):
        """
        Set the qpos0
        """
        qpos_addr = self.sim.model.get_joint_qpos_addr(name)
        self.sim.model.qpos0[qpos_addr[0] : qpos_addr[1]] = qpos

    def _set_color(self, name, color):
        """
        Set the color
        """
        body_idx1 = self.sim.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.sim.model.geom_rgba[geom_idx, 0 : len(color)] = color

    def _mass_center(self):
        """
        Get the mass center
        """
        mass = np.expand_dims(self.sim.model.body_mass, axis=1)
        xpos = self.sim.data.xipos
        return np.sum(mass * xpos, 0) / np.sum(mass)

    def on_collision(self, ref_name, body_name=None):
        """
        Checks if there is collision
        """
        mjcontacts = self.sim.data.contact
        ncon = self.sim.data.ncon
        for i in range(ncon):
            ct = mjcontacts[i]
            g1 = self.sim.model.geom_id2name(ct.geom1)
            g2 = self.sim.model.geom_id2name(ct.geom2)
            if g1 is None or g2 is None:
                continue  # geom_name can be None
            if body_name is not None:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0) and (
                    g1.find(body_name) >= 0 or g2.find(body_name) >= 0
                ):
                    return True
            else:
                if g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0:
                    return True
        return False

    # inverse kinematics
    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Sawyer robots have 7 joints and positions are in rotation angles.
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes_all]

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes_all]

    def _robot_jpos_getter(self):
        return np.array(self._joint_positions)

    def _setup_action(self, action):
        if self._rescale_actions:
            action = np.clip(action, -1, 1)

        arm_action = action[: self.mujoco_robot.dof]
        if self._agent_type in ["Sawyer", "Panda", "Jaco", "Fetch"]:
            gripper_action_in = action[
                self.mujoco_robot.dof : self.mujoco_robot.dof
                + self.gripper["right"].dof
            ]
            gripper_action_actual = self.gripper["right"].format_action(
                gripper_action_in
            )
            action = np.concatenate([arm_action, gripper_action_actual])

        elif self._agent_type == "Baxter":
            last = self.mujoco_robot.dof  # Degrees of freedom in arm, i.e. 14
            gripper_right_action_in = action[last : last + self.gripper["right"].dof]
            last = last + self.gripper["right"].dof
            gripper_left_action_in = action[last : last + self.gripper["left"].dof]
            gripper_right_action_actual = self.gripper["right"].format_action(
                gripper_right_action_in
            )
            gripper_left_action_actual = self.gripper["left"].format_action(
                gripper_left_action_in
            )
            action = np.concatenate(
                [arm_action, gripper_right_action_actual, gripper_left_action_actual]
            )

        if self._rescale_actions:
            # rescale normalized action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_action = bias + weight * action
        else:
            applied_action = action

        # gravity compensation
        self.sim.data.qfrc_applied[
            self._ref_joint_vel_indexes_all
        ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes_all]
        self.sim.data.qfrc_applied[
            self._ref_gripper_joint_vel_indexes_all
        ] = self.sim.data.qfrc_bias[self._ref_gripper_joint_vel_indexes_all]

        return applied_action

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_quat(self):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _left_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("left_hand")

    @property
    def _left_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._left_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _left_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._left_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _left_hand_quat(self):
        """
        Returns eef orientation of left hand in base from of robot.
        """
        return T.mat2quat(self._left_hand_orn)
