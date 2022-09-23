import argparse
import configparser
import math
import os
import random
import time
from typing import Dict

from hydra.utils import to_absolute_path
from hydra import compose, initialize
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
# from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from omegaconf import OmegaConf


# from tasks.dual_franka import *
# from utils.utils import HDF5DatasetWriter_multi
# from utils.utils import set_np_formatting, set_seed


# self define functions
def myparser(args, cfg_path):
    temp = OmegaConf.load(cfg_path)
    args = vars(args)
    res = list()
    for key, value in args.items():
        if key not in temp.keys():
            pass
        else:
            if key == 'physics_engine':
                res.append('physics_engine=physx')
            else:
                res.append(str(key) + '=' + str(value))
    res.append('pipeline=None')  # force pipeline=None
    res.append('task=DualFranka')
    return res


def get_cfg(path):
    # override
    res = myparser(args, path)
    initialize(config_path="cfg")
    cfg = compose(config_name="config", overrides=res)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # add all other args that do not override
    cfg = dict(cfg)
    cfg.update(vars(args))
    cfg = argparse.Namespace(**cfg)

    return cfg


def save(path, name, data):
    # save pose to do ik
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)
    if not os.path.exists(file_path):
        os.system(r"touch {}".format(path))
    with open(file_path, encoding="utf-8", mode="a") as file:
        file.write(str(data) + '\n')
    print('save success to', name)


def load_target_ee(filepath, dlen=9):
    # read target ee
    with open(filepath, "r") as f:
        txtdata = f.read()
    import re
    x = txtdata.split("]")
    res = []
    for i in x:
        tmp = re.findall(r"-?\d+\.?\d*", i)
        if len(tmp) == dlen:
            res.append(tmp)
        elif len(tmp) == 0:
            pass
        else:
            raise Exception('data format error')
    if len(res) % 2:
        raise Exception('data format error')

    target = np.array(res).astype(float).reshape(-1, 2, 9)

    return torch.from_numpy(target).float()


def load_franka_dof(doftxt):
    doftensor = load_target_ee(doftxt, 9).flatten()
    env.load_franka_dof(doftensor)
    print('load dof target', doftensor)


def parse_reward_detail(dictobj: Dict):
    for k, v in dictobj.items():
        for k1, v1 in v.items():
            if isinstance(v1, tuple):
                new_list = list()
                for obj in v1:
                    if isinstance(obj, torch.Tensor):
                        new_list.append(obj.tolist()[0] if obj.shape == 1 else obj.tolist())
                    else:
                        new_list.append(obj)
                dictobj[k][k1] = tuple(new_list)
            elif isinstance(v1, torch.Tensor):
                dictobj[k][k1] = v1.tolist()[0] if v1.shape == 1 else v1.tolist()

    return dictobj


def print_detail_clearly(dictobj):
    obj = parse_reward_detail(dictobj)
    for k, v in obj.items():
        print_highlight(k)
        if isinstance(v, Dict):
            findoprint(v)
        else:
            print(str(k) + ": " + str(v))


def findoprint(dictobj):
    for k, v in dictobj.items():
        if isinstance(v, Dict):
            findoprint(v)
        else:
            print(str(k) + ": " + str(v))


def print_highlight(*args):
    msg = ''
    for k in args:
        msg += str(k) + ' '
    print("\033[1;32m" + msg + "\033[0m")


total_print_mode = 3


def print_state(if_all=False):
    if print_mode >= 1 or if_all == True:
        print('actions', pos_action.numpy())
        print('franka_dof', franka_dof)
        print('ee_pose&gripper', torch.cat((ee_pose, gripper_dof), dim=1))
        print('obs-', env.compute_observations())
        print('rew-', env.compute_reward())

    if print_mode >= 2 or if_all == True:
        print_detail_clearly(env.reward_dict)

    # print reset env_ids
    if print_mode >= 1 or if_all == True:
        check_reset()


def check_reset():
    env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if env_ids.shape[0] != 0:
        print_highlight('trigger reset:', env_ids.numpy())
        return True
    return False


# camera pos
cam_switch = 0
cam_pos = [
    [[0.66, 0.75, 1.20], [-2.80, -0.83, -3.14]],  # from RF
    [[0.66, 0.75, -1.20], [-2.80, -0.83, 3.14]],  # from LF
    [[-1.77, 0.81, -0.17], [3.14, -0.80, -0.7]],  # from behind
]


# DualFranka class for test
class DualFrankaTest(DualFranka):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, sim_params):
        self.sim_params = sim_params
        super().__init__(cfg, sim_device, graphics_device_id, headless)

    def set_viewer(self):
        """Create the viewer."""
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a vienv.viewerewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "reset")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_P, "change_print_state")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "save")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_T, "print_once")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_C, "switch_cam_view")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_0, "if_target_track")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_N, "force_next_stage")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_D, "change_debug")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "check_task_stage")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_9, "pause_tracking")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_L, "load_franka_dof")
            # ik ee drive
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT_SHIFT, "switch_franka")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_UP, "drive_xminus")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_DOWN, "drive_xplus")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT, "drive_zplus")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT, "drive_zminus")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT_BRACKET, "drive_yplus")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT_BRACKET, "drive_yminus")

            # set camera view
            self.cam_view_switch(cam_pos[cam_switch])

    def compute_reward(self, action=None):
        # no action penalty in test
        if action is None:
            self.actions = torch.zeros((self.num_Envs, self.cfg["env"]["numActions"])).to(self.device)
        else:
            self.actions = action
        super().compute_reward()
        return self.rew_buf

    def cam_view_switch(self, vec):
        # Point camera at middle env
        num_per_row = int(math.sqrt(self.num_envs))
        cam_pos = gymapi.Vec3(*vec[0])
        cam_target = gymapi.Vec3(*vec[1])
        middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    def load_franka_dof(self, doftensor):
        self.franka_dof_targets[:, :18] = doftensor
        self.compute_reward(action=None)
        self.pre_physics_step(self.actions)

    def judge_now_stage(self, debug=False):
        # pre compute
        self.compute_observations()
        # object base link pose
        table_pose = self.rigid_body_states[:, self.table_handle][:, 0:3]
        cup_pos = self.rigid_body_states[:, self.cup_handle][:, 0:3]
        cup_rot = self.rigid_body_states[:, self.cup_handle][:, 3:7]
        spoon_pos = self.rigid_body_states[:, self.spoon_handle][:, 0:3]
        spoon_rot = self.rigid_body_states[:, self.spoon_handle][:, 3:7]
        # grasp pose
        cup_grasp_pos = self.cup_grasp_pos
        cup_grasp_rot = self.cup_grasp_rot
        spoon_grasp_pos = self.spoon_grasp_pos
        spoon_grasp_rot = self.spoon_grasp_rot
        # franka grasp point
        right_franka_grasp_pos = self.franka_grasp_pos
        right_franka_grasp_rot = self.franka_grasp_rot
        left_franka_grasp_pos = self.franka_grasp_pos_1
        left_franka_grasp_rot = self.franka_grasp_rot_1

        axis0 = quat_rotate_inverse(cup_rot, left_franka_grasp_pos - cup_grasp_pos)
        pre_stage_1 = [torch.abs(spoon_grasp_pos - right_franka_grasp_pos)[:, 1] < 0.01,  # y in spoon thickness
                       torch.gt(torch.tensor([0.025, 0.05, 0.025]), axis0).all(),
                       # in cup volume(cupsize 0.05*0.05*0.1)
                       # torch.sqrt((left_franka_grasp_pos[:, 0]-cup_grasp_pos[:, 0])**2 \
                       #     + (left_franka_grasp_pos[:, 2]-cup_grasp_pos[:, 2])**2) < 0.025, # < cup_width/2
                       ]

        stage_1 = [spoon_pos[:, 1] - 0.4 > 0.15,  # spoon_y - table_height > x  (shelf height ignored)
                   torch.norm(spoon_grasp_pos - right_franka_grasp_pos) < 0.05,  # keep in hand
                   cup_pos[:, 1] - 0.4 > 0.1,
                   torch.norm(cup_grasp_pos - left_franka_grasp_pos) < 0.05,
                   ]

        cup_up_axis = torch.tensor([0.0, 1.0, 0.0])  # cup stand: cup-y
        spoon_stand_axis = torch.tensor([1.0, 0.0, 0.0])  # spoon ready for stir: spoon-x
        axis1 = tf_vector(cup_rot, cup_up_axis)
        axis2 = tf_vector(spoon_rot, spoon_stand_axis)
        dot1 = torch.bmm(axis1.view(env.num_envs, 1, 3), axis2.view(env.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        axis3 = quat_rotate_inverse(cup_rot, spoon_pos - cup_pos)  # relative spoon pos in cup

        stage_2 = [torch.acos(dot1) / 3.1415 * 180 < 30,  # spoon-x should align with cup-y(<30deg)
                   # torch.gt(torch.tensor([0.025, 0.025]),axis3[:, [0,2]]).all() ,
                   #     axis3[:, 1]-0.15/2 -0.1 > 0, # spoon tip higher than cup height(spoon_base_y-half_spoon_len-cup_height>0)
                   ]

        spoon_tip_pos = quat_rotate_inverse(spoon_rot, spoon_pos) - 0.5 * torch.tensor([0.15, 0, 0])
        spoon_tip_pos = quat_rotate(spoon_rot, spoon_tip_pos)
        v1_s3 = quat_rotate_inverse(cup_rot, spoon_tip_pos - cup_pos)  # relative spoon pos in cup
        stage_3 = [
            torch.acos(dot1) / 3.1415 * 180 < 30,
            torch.gt(torch.tensor([0.025, 0.025]), axis3[:, [0, 2]]).all(),
            axis3[:, 1] - 0.15 / 2 - 0.1 < 0,  # spoon tip in cup
        ]

        prestage_s3 = [torch.gt(torch.tensor([0.025, 0.025]), v1_s3[:, [0, 2]]),
                       torch.lt(torch.tensor([-0.025, -0.025]), v1_s3[:, [0, 2]]),  # x,z in cup
                       v1_s3[:, 1] - 0.1 < 0 and v1_s3[:, 1] > 0]

        if debug:
            print("pre_stage_1", pre_stage_1)
            print("stage_1", stage_1)
            print("stage_2", stage_2)
            print("stage_3", stage_3)
            print("prestage_3", prestage_s3)
        return [all(pre_stage_1), all(stage_1), all(stage_2), all(stage_3), ]


# calculation
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(dpose, jacobian):
    # solve damped least squares
    j_eef_T = torch.transpose(jacobian, 1, 2)
    lmbda = torch.eye(6, device=env.device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(jacobian @ j_eef_T + lmbda) @ dpose).view(env.num_envs, 7)
    return u


def ee_position_drive(franka, dist: list):
    global manual_drive, now_target
    manual_drive |= 0b10
    env.curi_dof_targets[:, 3:21] = franka_dof.flatten()
    now_target[:, :, :7] = ee_pose.view(-1, 2, 7)[..., :7]
    now_target[:, :, -2:] = gripper_dof
    now_target[:, franka, 0:3] += torch.tensor(dist, dtype=torch.float, device=env.device)
    # env.gym.set_rigid_body_state_tensor(env.sim, gymtorch.unwrap_tensor(now_target))


def get_franka():
    franka_dof = env.dof_state[:, 0][3:].view(2, -1)
    gripper_dof = franka_dof[:, -2:]
    ee_pose = torch.cat(
        (env.rigid_body_states[:, env.hand_handle][:, 0:7], env.rigid_body_states[:, env.hand_handle_1][:, 0:7]))
    return franka_dof, gripper_dof, ee_pose


def reset_env():
    # need to disable pose override in viewer
    print('Reset env')
    # env.reset_idx_replay_buffer(torch.arange(env.num_envs, device=env.device))
    env.reset_idx(torch.arange(env.num_envs, device=env.device))


def ready_to_track():
    global print_mode, now_stage, target_pose, total_stage, writer, step, reset_flag, track_time, prev_err, relative_err, task_stage, prev_task_stage
    # global values
    prev_err = torch.ones((1,), dtype=torch.float, device=env.device)
    relative_err = prev_err.clone()
    reset_flag = False
    step = 0
    # reset track values
    task_stage = 0
    prev_task_stage = 0
    print_mode = 0
    now_stage = 0
    if read_from_origindata:
        target_pose = load_target_ee(target_data_path, 9).to(env.device)
    else:
        cup_npy = np.load('test_save/cup.npy')
        spoon_npy = np.load('test_save/spoon.npy')
        cup_pos = cup_npy[:, :7]
        spoon_pos = spoon_npy[:, :7]
        # cup_interpos = np.zeros([500, 7], dtype=float)
        # spoon_interpos = np.zeros([500, 7], dtype=float)
        # j = 0
        # for i in range(int(cup_pos.shape[0] / 5)):
        #     cup_interpos[i, :] = cup_pos[j, :]
        #     spoon_interpos[i, :] = spoon_pos[j, :]
        #     j += 5
        # cup_interpos[80:,:]=cup_pos[400:,:]
        # spoon_interpos[80:,:]=spoon_pos[400:,:]
        spoon_gripper = np.zeros([cup_pos.shape[0], 2], dtype=float)
        total_stage = np.zeros([cup_pos.shape[0], 2, 9], dtype=float)
        spoon_gripper[170:, :] = [0.0035, 0.0035]
        spoon_gripper[:170, :] = [0.04, 0.04]

        cup_gripper = np.zeros([cup_pos.shape[0], 2], dtype=float)
        cup_gripper[44:, :] = [0.024, 0.024]
        cup_gripper[:44, :] = [0.04, 0.04]

        franka1_pos = np.hstack((cup_pos, spoon_gripper))
        franka_pos = np.hstack((spoon_pos, cup_gripper))
        total_stage[:, 0, :] = franka1_pos
        total_stage[:, 1, :] = franka_pos
        target_pose = torch.from_numpy(total_stage).float()
    total_stage = target_pose.shape[0]
    print("The num of total stage is: ", total_stage)
    track_time = time.time()
    print('Start tracking, stage 0')


if __name__ == "__main__":
    file_time = time.strftime("%m-%d-%H_%M_%S", time.localtime())
    test_config = configparser.ConfigParser()
    test_config.read('test_config.ini')
    franka_cfg_path = test_config['PRESET'].get('franka_cfg_path', './cfg/config.yaml')
    print_mode = test_config['PRESET'].getint('print_mode', 0)
    debug_mode = test_config['PRESET'].getint('debug_mode', 0)
    target_data_path = test_config["SIM"].get('target_data_path', None)
    auto_track_pose = test_config["DEFAULT"].getboolean('auto_track_pose', False)
    read_from_origindata = test_config["DEFAULT"].getboolean('read_from_origindata', False)
    left_control_k = test_config["SIM"].getfloat('left_control_k', 0.6)
    right_control_k = test_config["SIM"].getfloat('right_control_k', 0.6)
    gripper_control_k = test_config["SIM"].getfloat('gripper_control_k', 0.6)
    damping = test_config["SIM"].getfloat('damping', 0.05)
    norm_err = test_config["SIM"].getfloat('norm_err', 1e-2)
    gripper_err = test_config["SIM"].getfloat('gripper_err', 1e-2)
    output_path = test_config['SIM'].get('output_path', './test_save')
    output_name = file_time + '.txt'
    write_hdf5data = test_config["DEFAULT"].getboolean('write_hdf5data', False)
    MultiPPO = test_config["DEFAULT"].getboolean('MultiPPO', False)
    output_hdf5_path = os.path.join(output_path, 'hdf5')
    output_hdf5_name = file_time + '.hdf5'
    manual_drive_k = test_config["SIM"].getfloat('manual_drive_k', 0.1)
    auto_error = test_config["SIM"].getfloat('auto_error', 1e-2)
    dof_path = test_config['SIM'].get('load_dof_path', './test_save/dof.txt')
    if target_data_path is None:
        auto_track_pose = False
    if write_hdf5data:
        pass

    ## OmegaConf & Hydra Config
    # Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    # allows us to resolve default arguments which are copied in multiple places in the config. used primarily for num_ensv
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

    # global test args
    args = gymutil.parse_arguments(description="Franka Tensor OSC Example",
                                   custom_parameters=[
                                       {"name": "--num_envs", "type": int, "default": 1,
                                        "help": "Number of environments to create"},
                                       {"name": "--pos_control", "type": gymutil.parse_bool, "const": True,
                                        "default": True,
                                        "help": "Trace circular path in XZ plane"},
                                       {"name": "--orn_control", "type": gymutil.parse_bool, "const": True,
                                        "default": False, "help": "Send random orientation commands"}])

    # parse from default config
    cfg = get_cfg(franka_cfg_path)

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, -1, 0)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.up_axis = gymapi.UP_AXIS_Y
    else:
        raise Exception("This example can only be used with PhysX")

    sim_params.use_gpu_pipeline = False

    env = DualFrankaTest(cfg=omegaconf_to_dict(cfg.task),
                         sim_device=cfg.sim_device,
                         graphics_device_id=cfg.graphics_device_id,
                         headless=False,
                         sim_params=sim_params)

    # inital global values
    manual_drive = 0b00
    prev_err = torch.ones((1,), dtype=torch.float, device=env.device)
    relative_err = prev_err.clone()
    reset_flag = False
    step = 0
    now_target = torch.zeros((env.num_envs, 2, 9), dtype=torch.float, device=env.device)
    left_action = torch.zeros_like(env.franka_dof_state_1[..., 0]).squeeze(
        -1)  # only need [...,0]->position, 1 for velocity
    right_action = torch.zeros_like(env.franka_dof_state[..., 0]).squeeze(-1)
    pos_action = torch.zeros_like(torch.cat((right_action, left_action), dim=0))
    zero_action = torch.zeros_like(pos_action)
    if auto_track_pose:
        ready_to_track()
    if write_hdf5data:
        # init writer once
        os.makedirs(output_hdf5_path, exist_ok=True)
        path = os.path.join(output_hdf5_path, output_hdf5_name)
        writer = HDF5DatasetWriter_multi(outputPath=path, bufSize=1024)

    while not env.gym.query_viewer_has_closed(env.viewer):

        franka_dof, gripper_dof, ee_pose = get_franka()
        # Get input actions from the viewer and handle them appropriately
        for evt in env.gym.query_viewer_action_events(env.viewer):
            if evt.value > 0:
                if evt.action == "reset":
                    reset_env()
                elif evt.action == "save":
                    env.gym.refresh_actor_root_state_tensor(env.sim)
                    env.gym.refresh_dof_state_tensor(env.sim)
                    env.gym.refresh_net_contact_force_tensor(env.sim)
                    env.gym.refresh_rigid_body_state_tensor(env.sim)
                    env.gym.refresh_jacobian_tensors(env.sim)
                    franka_dof, gripper_dof, ee_pose = get_franka()
                    save_data = torch.cat((ee_pose, gripper_dof), dim=1)
                    # print dof,pose detail
                    print('franka_dof', franka_dof)
                    print('save: ee_pose&gripper', save_data)
                    # save to file
                    save(output_path, output_name, save_data)
                elif evt.action == "change_print_state":
                    print("Change print mode")
                    print_mode += 1
                    if print_mode >= total_print_mode:
                        print("Stop printing")
                        print_mode = 0
                elif evt.action == "print_once":
                    print("Print once", random.randint(0, 100))
                    print_state(if_all=True)
                elif evt.action == "switch_cam_view":
                    print("Switch view")
                    cam_switch += 1
                    if cam_switch >= len(cam_pos):
                        cam_switch = 0
                    env.cam_view_switch(cam_pos[cam_switch])
                elif evt.action == "if_target_track":
                    if auto_track_pose:
                        print('Disable target tracking')
                    else:
                        print('Enable asdf target tracking')
                        reset_env()
                        ready_to_track()
                    auto_track_pose = ~auto_track_pose
                elif evt.action == "force_next_stage":
                    if auto_track_pose and now_stage < total_stage - 1:
                        now_stage += 1
                        print('Force jump to next stage', now_stage)
                    else:
                        print('Empty key')
                elif evt.action == "change_debug":
                    if debug_mode == 0:
                        debug_mode = 1
                        print('Enable debug mode')
                    else:
                        debug_mode = 0
                        print('Disable debug mode')
                elif evt.action == "switch_franka":
                    if manual_drive & 1:
                        manual_drive &= 0b10
                        print('Drive right franka')
                    else:
                        manual_drive |= 0b01
                        print('Drive left franka')
                elif evt.action == "check_task_stage":
                    print_highlight('check task stage:')
                    env.judge_now_stage(debug=True)
                elif evt.action == "load_franka_dof":
                    load_franka_dof(dof_path)
                elif evt.action == "pause_tracking":
                    try:
                        if now_stage < total_stage:
                            if auto_track_pose:
                                print('Pause tracking')
                            else:
                                print('Resume tracking')
                            auto_track_pose = ~auto_track_pose
                        else:
                            print("Empty key")
                    except:
                        print("Empty key")
                elif evt.action == "drive_xminus":
                    ee_position_drive(manual_drive & 0b01, dist=[-manual_drive_k, 0, 0])
                elif evt.action == "drive_xplus":
                    ee_position_drive(manual_drive & 0b01, dist=[manual_drive_k, 0, 0])
                elif evt.action == "drive_yminus":
                    ee_position_drive(manual_drive & 0b01, dist=[0, -manual_drive_k, 0])
                elif evt.action == "drive_yplus":
                    ee_position_drive(manual_drive & 0b01, dist=[0, manual_drive_k, 0])
                elif evt.action == "drive_zminus":
                    ee_position_drive(manual_drive & 0b01, dist=[0, 0, -manual_drive_k])
                elif evt.action == "drive_zplus":
                    ee_position_drive(manual_drive & 0b01, dist=[0, 0, manual_drive_k])

        # Step the physics
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)

        # refresh tensors
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_jacobian_tensors(env.sim)

        if auto_track_pose or manual_drive & 0b10:

            if write_hdf5data and auto_track_pose:
                env.compute_observations()
                # save after get next_obs
                next_obs = env.obs_buf.clone().view(-1, 74).numpy()
                action = pos_action.clone().view(-1, 18).numpy()
                # TODO: here calculate done
                done = np.array([[0]], dtype='i8')

                if MultiPPO:
                    next_obs_left = env.obs_buf_left.clone().view(-1, 37).numpy()
                    next_action_left = env.franka_dof_pos.clone().view(-1, 9).numpy()
                    done_spoon = env.reset_buf_spoon.clone().view(-1, 1).numpy()
                    done_cup = env.reset_buf_cup.clone().view(-1, 1).numpy()
                    next_obs_right = env.obs_buf_right.clone().view(-1, 37).numpy()
                    next_action_right = env.franka_dof_pos_1.clone().view(-1, 9).numpy()
                    if step > 0:
                        # append last stage to writer
                        writer.add(obs_left, action_left, rew, next_obs_left, done_spoon, next_action_left,
                                   obs_right, action_right, rew, next_obs_right, done_cup, next_action_right)

                    # next round
                    action_left = next_action_left.copy()
                    action_right = next_action_right.copy()
                    obs_left = next_obs_left.copy()
                    obs_right = next_obs_right.copy()
                    rew = env.rew_buf.clone().view(-1, 1).numpy()
                else:
                    if step > 0:
                        # append last stage to writer
                        writer.add(obs, action, rew, next_obs, done)
                    # next round
                    obs = next_obs.copy()
                    rew = env.rew_buf.clone().view(-1, 1).numpy()
            ## Calculation here
            # get jacobian tensor
            # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
            jacobian_curi = env.gym.acquire_jacobian_tensor(env.sim, "curi")

            jacobian_left = gymtorch.wrap_tensor(jacobian_curi)[:, 3:12, :, 3:12]
            jacobian_right = gymtorch.wrap_tensor(jacobian_curi)[:, 12:, :, 12:]

            # get link index of panda hand, which we will use as end effector
            # franka_link_dict = env.gym.get_asset_rigid_body_dict(franka_asset)
            # franka_hand_index = franka_link_dict["panda_hand"]
            franka_hand_index = 8  # just set to 8 instead

            # jacobian entries corresponding to franka hand
            j_eef_left = jacobian_left[:, franka_hand_index - 1, :, :7]
            j_eef_right = jacobian_right[:, franka_hand_index - 1, :, :7]
            # print('j_eef_left: {}'.format(j_eef_left))
            # print('j_eef_right: {}'.format(j_eef_right))

            # decide goal(target)
            if auto_track_pose:
                now_target = target_pose[now_stage, ...].unsqueeze(0).repeat(env.num_envs, 1, 1)

            left_hand_pos = env.rigid_body_states[:, env.hand_handle_1][:, 0:3]
            left_goal_pos = now_target[:, 1, 0:3]
            left_hand_rot = env.rigid_body_states[:, env.hand_handle_1][:, 3:7]
            left_goal_rot = now_target[:, 1, 3:7]

            right_hand_pos = env.rigid_body_states[:, env.hand_handle][:, 0:3]
            right_goal_pos = now_target[:, 0, 0:3]
            right_hand_rot = env.rigid_body_states[:, env.hand_handle][:, 3:7]
            right_goal_rot = now_target[:, 0, 3:7]

            # compute position and orientation error
            left_pos_err = left_goal_pos - left_hand_pos
            left_orn_err = orientation_error(left_goal_rot, left_hand_rot)
            left_dpose = torch.cat([left_pos_err, left_orn_err], -1).unsqueeze(-1)
            right_pos_err = right_goal_pos - right_hand_pos
            right_orn_err = orientation_error(right_goal_rot, right_hand_rot)
            right_dpose = torch.cat([right_pos_err, right_orn_err], -1).unsqueeze(-1)
            left_grip_err = now_target[:, 1, 7:9] - gripper_dof[1, :]
            right_grip_err = now_target[:, 0, 7:9] - gripper_dof[0, :]

            # if goal then next target
            e = torch.norm(left_dpose) + torch.norm(right_dpose)
            relative_err = torch.norm(e - prev_err)
            prev_err = e.clone()
            e_gripper = torch.norm(left_grip_err) + torch.norm(right_grip_err)
            if (e < norm_err or relative_err < auto_error) and (e_gripper < gripper_err and auto_track_pose):
                now_stage += 1
                now_time = time.time()
                if now_stage >= total_stage:
                    print('complete all goals', now_time - track_time, 's')
                    auto_track_pose = False
                    print('Stop target tracking')
                    if write_hdf5data:
                        writer.flush()
                        print('now total steps saved:', writer.idx_left)
                else:
                    print_highlight('Stage', now_stage, 'Step', step, round(now_time - track_time, 4), 's')

            # body ik, relative control
            left_action[:, :7] = left_control_k * control_ik(left_dpose, j_eef_left)
            right_action[:, :7] = right_control_k * control_ik(right_dpose, j_eef_right)
            # gripper actions
            left_action[:, 7:9] = gripper_control_k * left_grip_err
            right_action[:, 7:9] = gripper_control_k * right_grip_err
            # merge two franka
            pos_action = torch.cat((right_action, left_action), dim=0)
            # print(pos_action)

            # Deploy actions
            env.pre_physics_step(pos_action.view(env.num_envs, -1))

            # calculation
            env.compute_observations()
            if auto_track_pose:
                env.compute_reward(action=pos_action.view(env.num_envs, -1))
            else:
                env.compute_reward(action=None)

            # check if trigger reset
            if not reset_flag:
                if check_reset():
                    reset_flag = True
                    print('now franka dof:', franka_dof)

            # check now stage if in auto tracking
            if auto_track_pose:
                task_stage = 0
                for index, s in enumerate(env.judge_now_stage(), start=1):
                    if s == True:
                        task_stage = index
                if task_stage != prev_task_stage:
                    # print_highlight("now task stage: {}".format(task_stage))
                    prev_task_stage = task_stage

            ## debug print
            if debug_mode and step % 70 == 0:
                torch.set_printoptions(precision=4, sci_mode=True)
                print('d err:', torch.norm(left_dpose), torch.norm(right_dpose))
                print('grip err:', left_grip_err, right_grip_err)
                print('e:', e, e_gripper, 'relative e:', relative_err)
                torch.set_printoptions(precision=4, sci_mode=False)

        # Step rendering
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, False)
        env.gym.sync_frame_time(env.sim)

        # print obs/reward
        step += 1
        if step % 70 == 0:
            if print_mode != 0:
                print_state()

        manual_drive &= 0b01

    print("Done")
    if write_hdf5data:
        writer.close()
    env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)
