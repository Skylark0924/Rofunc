from abc import ABC
import cv2
from isaacgym import gymapi, gymtorch
from rofunc.learning.RofuncRL.tasks.isaacgymenv.grasp.utils.transform import fuse_rgbd_to_pointcloud, \
    subsample_valid, pointcloud_to_voxelgrid
import numpy as np
import os
import torch
from typing import *


class CameraTaskMixin(ABC):
    """Provides methods to configure camera setups from the environment's config
    file.

    Examples:
        The following block specifies a camera setup in the config file. The
        first three keywords define whether to save MP4 recordings of the
        rollouts from all cameras and environment instances, as well as whether
        RGBD images should be converted to further to point-cloud or voxel-grid
        representations.
        Every following block adds a new camera to the environment. Supported
        camera types are 'rgb' and 'rgbd'. The current images of all cameras are
        stored in a buffer of shape (num_envs, num_cameras, height, width,
        num_channels), hence requiring cameras to have the same type and image
        dimensions if multiple cameras are being used. This data will be added
        to the obs_dict under the 'image' key.

        cameras:
            save_recordings: ${...save_recordings}
            convert_to_pointcloud: True
            convert_to_voxelgrid: False
            camera0:
              type: rgbd
              pos: [ 0.0, -0.5, 1.3 ]
              lookat: [ 0,  0, 0.8 ]
              horizontal_fov: 70
              width: 128
              height: 128
            camera1:
              type: rgbd
              pos: [ 0.0, 0.5, 1.3 ]
              lookat: [ 0,  0, 0.8 ]
              horizontal_fov: 70
              width: 128
              height: 128
    """

    def _create_cameras(self, cfg: Dict[str, Dict]) -> None:
        """Creates camera sensors according to the provided config file.

        Args:
            cfg: Camera config.
        """
        self.camera_handles, self.camera_names = [], []
        for i in range(self.num_envs):
            self.camera_handles.append([])
            for camera_name, camera_cfg in cfg.items():
                if camera_name in ["save_recordings", "convert_to_pointcloud",
                                   "convert_to_voxelgrid"]:
                    continue
                self.camera_names.append(camera_name)
                camera_properties = gymapi.CameraProperties()
                camera_properties.width = camera_cfg["width"]
                camera_properties.height = camera_cfg["height"]
                camera_properties.horizontal_fov = camera_cfg["horizontal_fov"]
                if self.device.startswith("cuda"):
                    camera_properties.enable_tensors = True
                camera = self.gym.create_camera_sensor(self.envs[i],
                                                       camera_properties)
                pos = gymapi.Vec3(*camera_cfg["pos"])
                lookat = gymapi.Vec3(*camera_cfg["lookat"])
                self.gym.set_camera_location(camera, self.envs[i], pos, lookat)
                self.camera_handles[i].append(camera)

        self.num_cameras = len(self.camera_handles[0])
        self.camera_types = [t for t in [cfg[n]["type"]
                                         for n in self.camera_names]]
        self.camera_width = cfg[self.camera_names[0]]["width"]
        self.camera_height = cfg[self.camera_names[0]]["height"]
        self.convert_to_pointcloud = cfg["convert_to_pointcloud"]
        self.convert_to_voxelgrid = cfg["convert_to_voxelgrid"]

        self._infer_camera_matrices()

        if cfg["save_recordings"]:
            self._create_recordings()

    def _infer_camera_matrices(self) -> None:
        """Infers camera matrices, which are required to convert the image data
        to point-cloud or voxel-grid representations.
        """
        self.proj_mat = torch.from_numpy(
            self.gym.get_camera_proj_matrix(
                self.sim, self.envs[0], self.camera_handles[0][0])
        ).to(self.device)
        self.view_mat = []
        for cam_idx in range(self.num_cameras):
            view_mat = torch.from_numpy(
                self.gym.get_camera_view_matrix(self.sim, self.envs[0],
                                                self.camera_handles[0][
                                                    cam_idx])).to(self.device)
            self.view_mat.append(view_mat)
        self.view_mat = torch.stack(self.view_mat)

    def _create_recordings(self) -> None:
        """Initializes video writers for all cameras, image types, and
         environment instances.
         """
        train_dir = self.cfg.get("train_dir", "runs")
        experiment_name = self.cfg["experiment_name"]
        experiment_dir = os.path.join(train_dir, experiment_name)
        self.recording_dir = os.path.join(experiment_dir, "recordings")
        if not os.path.exists(self.recording_dir):
            os.mkdir(self.recording_dir)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.recording_episodes = [0 for _ in range(self.num_envs)]
        self.image_types = ["color", "depth"] if \
            self.camera_types[0] == "rgbd" else ["color"]

        self.recordings = [[[None
            for cam_id in range(self.num_cameras)]
            for image_type in self.image_types]
            for env_id in range(self.num_envs)]

        if self.convert_to_pointcloud:
            self.image_types.append("point_cloud")

            for env_id in range(self.num_envs):
                self.recordings[env_id].append([None for _ in range(self.num_cameras)])

            from pytorch3d.renderer import (
                look_at_view_transform,
                FoVOrthographicCameras,
                PointsRasterizationSettings,
                PointsRasterizer,
                PulsarPointsRenderer
            )
            R, T = look_at_view_transform(eye=((0.75, -0.4, 1.15),), up=((0, 0, 1),),
                                          at=((0., 0., 0.8),),)
            cameras = FoVOrthographicCameras(device=self.device, R=R, T=T,
                                             znear=0.01)
            raster_settings = PointsRasterizationSettings(
                image_size=(1080, 1920),
                radius=0.003,
                points_per_pixel=1
            )
            self.pc_renderer = PulsarPointsRenderer(
                rasterizer=PointsRasterizer(cameras=cameras,
                                            raster_settings=raster_settings),
                n_channels=3
            ).to(self.device)

        if self.convert_to_voxelgrid:
            self.image_types.append("voxel_grid")
            for env_id in range(self.num_envs):
                 self.recordings[env_id].append([None for _ in range(self.num_cameras)])

    def _set_camera(self, pos: List, lookat: List) -> None:
        if self.viewer is not None:
            pos = gymapi.Vec3(*pos)
            lookat = gymapi.Vec3(*lookat)
            self.gym.viewer_camera_look_at(self.viewer, None, pos, lookat)

    def _render_camera_sensors(self) -> None:
        """Renders all camera images and writes their content to the camera
        buffer of shape (num_envs, num_cameras, height, width, channel). This
        means that if multiple cameras are used they must have the same
        image dimensions."""
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
        if self.headless:
            self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        if self.device.startswith("cuda"):
            self.gym.start_access_image_tensors(self.sim)
        images = []
        for i in range(self.num_envs):
            images.append([])
            for j in range(self.num_cameras):
                images[i].append(self._get_image(
                    self.envs[i], self.camera_handles[i][j],
                    self.camera_types[j], self.device.startswith("cuda")))
            images[i] = torch.stack(images[i])
        images = torch.stack(images)

        if self.device.startswith("cuda"):
            self.gym.end_access_image_tensors(self.sim)

        self.image_buf[:] = images

    def _convert_rgbd_to_pointcloud(self,
                                    convert_rgb_to_camera_id: bool = False,
                                    visualize: bool = False
                                    ) -> None:
        """Fuses the RGBD views of all cameras into a colored point-cloud.

        Args:
            convert_rgb_to_camera_id: Whether to use color channels to identify
                which camera a point belongs to.
            visualize: Whether to visualize point-clouds in plotly.
        """
        xyz, features = fuse_rgbd_to_pointcloud(
            self.image_buf, self.proj_mat, self.view_mat,
            convert_rgb_to_camera_id)
        self.pointcloud_buf[..., 0:3] = xyz
        self.pointcloud_buf[..., 3:6] = features

        if visualize:
            from pytorch3d.structures import Pointclouds
            from pytorch3d.vis.plotly_vis import plot_scene
            plotly_figs = []
            for env_idx in range(self.num_envs):
                views = {}
                for cam_idx in range(self.num_cameras):
                    xyz_tmp, features_tmp = subsample_valid(
                        xyz[env_idx, cam_idx], features[env_idx, cam_idx])
                    views[self.camera_names[cam_idx]] = Pointclouds(
                        points=[xyz_tmp],
                        features=[features_tmp])
                plotly_figs.append(plot_scene({
                    f"Pointcloud in environment {env_idx}": views,
                }))
            for i in range(min(4, len(plotly_figs))):
                plotly_figs[i].show()
            import time
            time.sleep(10)

    def _convert_pointcloud_to_voxelgrid(self,
                                         visualize: bool = False
                                         ) -> None:
        """Converts the colored point-cloud into a voxel-grid.

        Args:
            visualize: Whether to visualize the resulting voxelgrid in
                matplotlib.
        """

        xyz = self.pointcloud_buf[..., 0:3]
        features = self.pointcloud_buf[..., 3:6]
        voxel_grid, voxel_features = pointcloud_to_voxelgrid(xyz, features)

        self.voxelgrid_buf[..., 0] = voxel_grid
        self.voxelgrid_buf[..., 1:4] = voxel_features

        if visualize:
            import matplotlib.pyplot as plt
            voxel_array = voxel_grid[0].cpu().numpy().astype(bool)
            colors = voxel_features[0].cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.voxels(voxel_array, facecolors=colors)
            ax.set_axis_off()
            plt.show()

    def _get_image(self, env_ptr, camera, image_type: str,
                   gpu_tensors: bool) -> torch.Tensor:
        """Called when rendering the camera sensors. Queries the camera images
        from Isaac Gym and puts them into the desired format.
        """
        if image_type == "rgb":
            image_types = [gymapi.IMAGE_COLOR]
        elif image_type == "rgbd":
            image_types = [gymapi.IMAGE_COLOR, gymapi.IMAGE_DEPTH]
        else:
            assert False

        image = []
        for t in image_types:
            if gpu_tensors:
                img_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera, t)
                img = gymtorch.wrap_tensor(img_tensor)
            else:
                img = self.gym.get_camera_image(self.sim, env_ptr, camera, t)
                img = torch.Tensor(img).view(img.shape[0], -1, 4)

            if t == gymapi.IMAGE_COLOR:
                img = img[..., 0:3] / 255.
            elif t == gymapi.IMAGE_DEPTH:
                img = img.unsqueeze(-1)

            image.append(img)
        image = torch.cat(image, dim=2)
        return image

    def _write_recordings(self) -> None:
        """Appends the latest camera images to the MP4 recordings.
        """
        for env_id in range(self.num_envs):
            for cam_id in range(self.num_cameras):
                for i, image_type in enumerate(self.image_types):
                    if image_type == "color":
                        image = self.image_buf[
                                    env_id, cam_id][..., 0:3].cpu().numpy()
                        image *= 255
                        image = image.astype(np.uint8)[..., ::-1]
                    elif image_type == "depth":
                        image = self.image_buf[
                                    env_id, cam_id][..., -1:].cpu().numpy()
                        max_dist = 2
                        image *= -255 / max_dist
                        image = np.clip(image, 0, 255)
                        image = 255 - image
                        image = image.astype(np.uint8)
                        image = np.repeat(image, 3, axis=2)
                    elif image_type == "point_cloud":
                        # Render merged point cloud from all camera views
                        if cam_id != 0:
                            continue
                        xyz = self.pointcloud_buf[
                              env_id, ..., 0:3].view(-1, 3)
                        features = self.pointcloud_buf[
                                   env_id, ..., 3:6].view(-1, 3)
                        xyz_valid, features_valid = subsample_valid(xyz,
                                                                    features)
                        image = self._render_pointcloud_to_image(
                            xyz_valid, features_valid).cpu().numpy()[0]
                        image *= 255
                        image = image.astype(np.uint8)[..., ::-1]

                    elif image_type == "voxel_grid":
                        # Render voxel grid
                        if cam_id != 0:
                            continue
                        voxel_array = self.voxelgrid_buf[env_id, ..., 0].cpu().numpy().astype(bool)
                        colors = self.voxelgrid_buf[env_id, ..., 1:4].cpu().numpy()

                        image = self._render_voxelgrid_to_image(voxel_array,
                                                                colors)
                    else:
                        assert False
                    self.recordings[env_id][i][cam_id].write(image)

    def reset_recordings_idx(self, env_ids):
        for env_id in env_ids:
            for cam_id in range(self.num_cameras):
                for i, image_type in enumerate(self.image_types):
                    if self.recordings[env_id][i][cam_id] is not None:
                        self.recordings[env_id][i][cam_id].release()
                    if i < 2:
                        self.recordings[env_id][i][cam_id] = \
                            cv2.VideoWriter(
                                os.path.join(self.recording_dir,
                                             f"recording_env_{env_id}_camera_{cam_id}"
                                             f"_episode_"
                                             f"{self.recording_episodes[env_id]}_"
                                             f"{image_type}.mp4"),
                                self.fourcc, 1 / (self.dt * self.control_freq_inv),
                                (self.camera_width, self.camera_height))
                    elif i == 2:
                        if cam_id == 0:
                            self.recordings[env_id][i][
                                cam_id] = cv2.VideoWriter(
                                os.path.join(self.recording_dir,
                                             f"recording_env_{env_id}_episode_"
                                             f"{self.recording_episodes[env_id]}_"
                                             f"{image_type}.mp4"),
                                self.fourcc, 1 / (self.dt * self.control_freq_inv),
                                (1920, 1080))
                    elif i == 3:
                        self.recordings[env_id][i][
                            cam_id] = cv2.VideoWriter(
                            os.path.join(self.recording_dir,
                                         f"recording_env_{env_id}_episode_"
                                         f"{self.recording_episodes[env_id]}_"
                                         f"{image_type}.mp4"),
                            self.fourcc, 1 / (self.dt * self.control_freq_inv),
                            (640, 480))
            self.recording_episodes[env_id] += 1

    def _render_pointcloud_to_image(self, xyz:  torch.Tensor,
                                    features: torch.Tensor) -> torch.Tensor:
        """Used to create recordings of the point-clouds.
        """
        from pytorch3d.structures import Pointclouds
        point_cloud = Pointclouds(points=xyz.unsqueeze(0), features=features.unsqueeze(0))
        image = self.pc_renderer(point_cloud, gamma=(1e-4,))
        return image

    def _render_voxelgrid_to_image(self, voxel_grid, colors) -> np.ndarray:
        import pyvista as pv

        grid = pv.UniformGrid()

        grid.dimensions = np.array(voxel_grid.shape) + 1
        grid.plot(show_edges=True)

        import time
        time.sleep(1000)
        data = 0
        return data
