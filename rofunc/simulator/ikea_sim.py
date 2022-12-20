import isaacgym
import os
from rofunc.simulator.franka_sim import FrankaSim
from rofunc.utils.logger.beauty_logger import beauty_print


class IkeaSim:
    def __init__(self, args, furniture_name, **kwargs):
        self.args = args
        self.furniture_name = furniture_name
        self.robot_sim = FrankaSim(args)
        self.asset_root = self.robot_sim.asset_root
        self._init_env_w_furniture()

    def _init_env_w_furniture(self):
        from isaacgym import gymtorch, gymapi

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005

        if self.furniture_name == "three_blocks":
            self.furniture_asset_folder = "urdf/objects/three_blocks/"
        elif self.furniture_name == "bed":
            self.furniture_asset_folder = "urdf/objects/bed/"

        beauty_print("Loading furniture asset {} from {}".format(self.furniture_asset_folder, self.asset_root),
                     type="info")

        furniture_assets = []
        furniture_poses = []

        for asset_file in os.listdir(os.path.join(self.asset_root, self.furniture_asset_folder)):
            if asset_file.endswith(".urdf"):
                asset_file = os.path.join(self.furniture_asset_folder, asset_file)
                furniture_asset = self.robot_sim.gym.load_asset(self.robot_sim.sim, self.asset_root, asset_file,
                                                                asset_options)
                furniture_pose = gymapi.Transform()
                furniture_pose.p = gymapi.Vec3(0, 1, 0)
                furniture_assets.append(furniture_asset)
                furniture_poses.append(furniture_pose)

        furniture_handles = []
        for i in range(self.robot_sim.num_envs):
            # add furniture
            for j in range(len(furniture_assets)):
                handle = self.robot_sim.gym.create_actor(self.robot_sim.envs[i], furniture_assets[j],
                                                         furniture_poses[j], "furniture", i, 1)
                self.robot_sim.gym.enable_actor_dof_force_sensors(self.robot_sim.envs[i], handle)
                furniture_handles.append(handle)

    def show(self):
        self.robot_sim.show()


if __name__ == '__main__':
    from isaacgym import gymutil

    args = gymutil.parse_arguments()
    args.use_gpu_pipeline = False

    furniture_name = "bed"
    ikea_sim = IkeaSim(args, furniture_name)
    ikea_sim.show()
