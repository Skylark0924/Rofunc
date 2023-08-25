""" Data processing class for synchronisation and re-sampling of data.
/!\ This is very much a work in progress. /!\

/!\ Unstable when accessing the data in funkyway because all the data
holders are designed to keep as little data as possible in memory /!\

Known limitations:
- The re-sampling is only done by ignoring or copying samples. It would be nice to add interpolation methods
- The re-sampling is done in a for loop. A bit slow, can be improved.

Code structure:
- MultiModalDataHandler: class to handle multiple data streams. Interface with the data.
                         MultiModalDataHandler is 'fed' DataHolders.
                         DataHolders deal with the specificities of each data sources.
                         DataHolders must inherit from DataHolder.
- XSensorDataHolder: class to handle data from XSens sensor.
- OptitrackDataHolder: class to handle data from Optitrack.
- RGBDDatasetDataHolder: class to handle data from Zed2i.

Author: Donatien Delehelle (donatien.delehelle[at]iit.it)
"""
import os
import glob
import time
import locale
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from PIL import Image
from datetime import datetime
from typing import List, Tuple
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import rofunc as rf
from .utils import rreplace, pcd_concat


class DataHolder(object):
    """Class to hold data for each frame"""

    def __init__(self):
        self.time_serie = None
        self._sampled_data = None
        self.transform = [np.eye(3), np.zeros((1, 3)), 1]
        # R, t, s: pt' = s(R@pt + t)
        self.tstart = None
        self.tstop = None

    @property
    def sampled_data(self):
        # This has the effect on not letting the object reader modify the data. Good idea ??
        return self._sampled_data

    def _pre_load(self):
        rf.oslab.create_dir(f".data_tmp/{self.__class__.__name__}/")
        self.tmp_folder = f".data_tmp/{self.__class__.__name__}/"

    def to_pcd(self, frame_number):
        raise NotImplementedError

    def lin_trans(self, data, inv=False):
        if data.shape[-1] == 3:
            if inv:
                return (self.transform[0].T @ (data.T / self.transform[2] - self.transform[1].T)).T
            else:
                return self.transform[2] * (self.transform[0] @ data.T + self.transform[1].T).T
        elif data.shape[-1] == 7:
            pos = self.lin_trans(data[:, :3], inv=inv)
            if inv:
                rot = [Quaternion(matrix=(self.transform[0].T @ Quaternion(q).rotation_matrix.T).T).elements for q in
                       data[:, 3:]]
            else:
                rot = [Quaternion(matrix=(self.transform[0] @ Quaternion(q).rotation_matrix.T).T).elements for q in
                       data[:, 3:]]
            rot = np.stack(rot, axis=0)
            return np.hstack((pos, rot))

    def _sample_repeat(self, ticks):
        """Samples the data by repeating the last value of the data serie when the time serie is not aligned with the ticks.
        """
        self._sampled_data = [None] * len(ticks)
        j = 0
        for i, tick in enumerate(ticks):
            if i > 0:
                self._sampled_data[i] = self._sampled_data[i - 1]
            while j < len(self.time_serie) and self.time_serie[j] <= tick:
                self._sampled_data[i] = self[j]
                j += 1

    def __getitem__(self):
        raise NotImplementedError

    def __del__(self):
        for root, dirs, files in os.walk(self.tmp_folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))


class RGBDDataHolder(DataHolder):
    def __init__(self, root_path, show_color=True, stat_clean=True,
                 depth_trunc=3500, size=(1920, 1080), tshift=0):
        super().__init__()
        self.data_serie = []
        self.intrinsic = []
        self.world2cam = []
        self.cam2world = []
        self.img_size = size
        self.root_path = root_path
        self.depth_trunc = depth_trunc
        self.show_color = show_color
        self.stat_clean = stat_clean
        # TODO: put in parent
        self.tshift = tshift
        self._pre_load()

    def _pre_load(self):
        super()._pre_load()
        with open(osp.join(self.root_path, 'time_table.txt')) as f:
            lines = f.read().splitlines()
            # TODO replace by '' and keep root path as attribute
            fs = [l.split(',')[0].replace('/exchange/donatien',
                                          '/Users/donatien/data/CLOVER_captures') for l in lines[1:]]
            ts = [int(l.split(',')[1]) - self.tshift for l in lines[1:]]
            self.data_serie = [f for _, f in sorted(zip(ts, fs))]
            self.time_serie = sorted(ts)
        self.tstart = self.time_serie[0]
        self.tstop = self.time_serie[-1]
        self.intrinsic = np.load(osp.join(self.root_path, 'intrinsics.npy'))
        self.world2cam = np.load(osp.join(self.root_path, 'world2cam.npy'))
        self.cam2world = np.load(osp.join(self.root_path, 'cam2world.npy'))

    def __getitem__(self, item):
        return self.data_serie[item]

    def to_pcd(self, frame_number):
        if self._sampled_data is None:
            raise ValueError("Data must be sampled before converting to pcd.")
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.img_size[0], self.img_size[1],
                                                      0, 0, 0, 0)
        color = Image.open(self._sampled_data[frame_number])
        color = np.array(color)[:, :, :3]  # Remove alpha channel
        color_raw = o3d.geometry.Image(np.array(color).astype(np.uint8))
        depth_raw = o3d.io.read_image(rreplace(self._sampled_data[frame_number], 'left', 'depth', 1))
        intrinsic.intrinsic_matrix = self.intrinsic
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=1, depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsic)
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=1)
        cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=2.0)
        # display_inlier_outlier(voxel_down_pcd, ind)
        pcd = voxel_down_pcd.select_by_index(ind)
        pts = np.asarray(pcd.points)
        h_pts = np.ones((pts.shape[0], 4))
        h_pts[:, :3] = pts  # Homogeneous points
        pcd.points = o3d.utility.Vector3dVector(
            np.matmul(self.cam2world, h_pts.T).T[:, :3]
        )  # Transform points to optitrack frame
        if not self.show_color:
            pcd.paint_uniform_color([0, 1, 0])

        return pcd

    def orig_fn_to_sampled(self, frame_number):
        prev_fnb = 0
        for i, frame_name in enumerate(self._sampled_data):
            fnb = int(frame_name[-10:-4])
            if fnb > frame_number and prev_fnb <= frame_number:
                return i - 1
            else:
                prev_fnb = fnb


class XsensDataHolder(DataHolder):
    def __init__(self, root_path, transform=None):
        super().__init__()
        self.root_path = root_path
        if transform is None:
            transform = [np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64),
                         np.zeros((1, 3), dtype=np.float64), 1000]
        self.transform = transform
        self.labels = {}
        self.files = []
        self._pre_load()

    @property
    def sampled_data(self):
        # This has the effect on not letting the object reader modify the data. Good idea ??
        return np.array(self._sampled_data)

    def _pre_load(self):
        super()._pre_load()
        if 'ms.npy' not in os.listdir(self.root_path):
            raise ValueError('No ms.npy file found in {}'.format(self.root_path))
        self.time_serie = np.load(osp.join(self.root_path, 'ms.npy'), mmap_mode='r')
        if not np.all(np.diff(self.time_serie) > 0):
            raise ValueError("Time serie must be monotonically increasing.")
        self.tstart = self.time_serie[0]
        self.tstop = self.time_serie[-1]

        self.files = os.listdir(self.root_path)

        cursor = 0
        data_acc = []
        for file_name in self.files:
            if file_name.split('.')[-1] == 'npy':
                np_data = np.load(os.path.join(self.root_path, file_name), mmap_mode='r')
                if len(np_data.shape) == 1:
                    np_data = np_data.reshape((-1, 1))
                data_len = np_data.shape[1]
                data_acc.append(np_data)
                self.labels[f"{file_name.split('.')[0]}"] = (cursor, cursor + data_len)
                cursor += data_len
        data = np.concatenate(data_acc, axis=1)
        self.tmp_file = osp.join(self.tmp_folder, f'{len(os.listdir(self.tmp_folder))}.npy')
        np.save(self.tmp_file, data)

    def __getitem__(self, item):
        data_ptr = np.load(self.tmp_file, mmap_mode='r')
        return data_ptr[item].copy()

    def sampled_match(self, label, pos_array, fnb=0, bnds=None):
        if pos_array.shape[0] != 1 and \
                pos_array.shape[0] != len(self._sampled_data) and \
                bnds is None:
            print("Not matching : pos_array must be constant or span the whole demonstration")
            return
        if label not in self.labels:
            raise ValueError(f"Label {label} not found in data.")
        label_idx = self.labels[label][0]
        if pos_array.shape[-1] == 3:
            if pos_array.shape[0] == 1:
                diff = pos_array[0] - self.lin_trans(self.sampled_data[fnb, label_idx:label_idx + 3][None, :])
            else:
                print("Current behavior of sampled match: median matching")
                diff = np.median(pos_array[bnds[0]:bnds[1]] - \
                                 self.lin_trans(
                                     self.sampled_data[bnds[0]:bnds[1], label_idx:label_idx + 3]),
                                 axis=0)[None, :]
            self.transform[1] = diff / self.transform[2] + self.transform[1]
        # Experimental
        elif pos_array.shape[-1] == 7:
            pos_array[fnb, 3:] = Quaternion(matrix=np.eye(3)).elements
            if pos_array.shape[0] >= 0:
                qrot = Quaternion(pos_array[fnb, 3:7]) * Quaternion(
                    self.sampled_data[fnb, label_idx + 3:label_idx + 7]).inverse
                mrot = qrot.rotation_matrix
                trans = pos_array[fnb, :3] / self.transform[2] - mrot @ self.sampled_data[fnb, label_idx:label_idx + 3]
                self.transform[0] = mrot
                self.transform[1] = trans[None, :]

    def to_pcd(self, frame_number):
        if self._sampled_data is None:
            raise ValueError("Data must be sampled before converting to pcd.")
        pcds = []
        for label in self.labels:
            (imin, imax) = self.labels[label]
            if imax - imin < 3:
                continue
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10, resolution=5) \
                .translate(self.lin_trans(self._sampled_data[frame_number][imin:imin + 3][None, :]).T)
            # Xsens data is in mm for some reason
            sphere.compute_vertex_normals()
            sample = sphere.sample_points_uniformly(number_of_points=1000)
            pcds.append(sample)
            # TODO: set color as param
            pcds[-1].paint_uniform_color([1, 0, 0])

        return pcd_concat(pcds)


class OptitrackDataHolder(DataHolder):
    # TODO: Merge with XsensDataHolder ?
    def __init__(self, meta=None, labels=None, data=None):
        super().__init__()
        if data is None:
            raise ValueError("Must provide data.")
        self.meta = meta
        self.labels = None
        self.data = None
        self.objs = []
        self.files = []
        self._pre_load(data, labels)

    @property
    def sampled_data(self):
        # This has the effect on not letting the object reader modify the data. Good idea ??
        return np.array(self._sampled_data)

    def _pre_load(self, data, labels):
        super()._pre_load()
        ot_start = self.meta['Capture Start Time'].replace('下午', 'PM').replace('上午', 'AM')
        ot_start = int(datetime.strptime(ot_start, '%Y-%m-%d %I.%M.%S.%f %p').timestamp() * 1000)
        self.time_serie = (data[:, 1] * 1000).astype(int) + ot_start
        self.tstart = self.time_serie[0]
        self.tstop = self.time_serie[-1]
        # data = data[:, 2:]
        # self.labels = labels[2:]
        self.labels = labels
        # TODO: change stupid separator. Too common
        self.objs = list(set([l.split('.')[0] for l in self.labels]))
        self.tmp_file = osp.join(self.tmp_folder, f'{len(os.listdir(self.tmp_folder))}.npy')
        np.save(self.tmp_file, data)

    def __getitem__(self, item):
        data_ptr = np.load(self.tmp_file, mmap_mode='r')
        return data_ptr[item].copy()

    def to_pcd(self, frame_number):
        if self._sampled_data is None:
            raise ValueError("Data must be sampled before converting to pcd.")
        pcds = []
        for o in self.objs:
            if f"{o}.pose.x" not in self.labels:
                continue
            imin = self.labels.index(f"{o}.pose.x")
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10, resolution=5) \
                .translate(self.lin_trans(self._sampled_data[frame_number][imin:imin + 3][None, :]).T)
            sphere.compute_vertex_normals()
            sample = sphere.sample_points_uniformly(number_of_points=1000)
            pcds.append(sample)
            # TODO: set color as param
            pcds[-1].paint_uniform_color([0, 0, 1])

        return pcd_concat(pcds)

    def orig_fn_to_sampled(self, frame_number):
        fptr = self.labels.index('frame')
        prev_fnb = 0
        for i, data in enumerate(self._sampled_data):
            fnb = data[fptr]
            if fnb > frame_number and prev_fnb <= frame_number:
                return i - 1
            else:
                prev_fnb = fnb


class MultimodalDataHandler(object):
    """Class to handle multimodal data collection.

    Attributes:
        tstep (int): Time step between each data point.
        tstart (int): Start time of the shared data time serie.
        tstop (int): Stop time of the shared data time serie.
    """

    # TODO: put in DataHolders family ?
    def __init__(self, tstep: int = None, tstart: int = None, tstop: int = None,
                 data_holders: List[DataHolder] = None, interpolate: str = 'repeat'):
        self.tstep = tstep
        self.data_holders = data_holders
        self.interpolate = interpolate

        self.update_bounds()
        if tstart is not None:
            self.tstart = tstart
        if tstop is not None:
            self.tstop = tstop

    @property
    def sampled_data(self):
        return [dh.sampled_data for dh in self.data_holders]

    def update_bounds(self):
        """Update the ticks attribute of the handler.

        Args:
            force (bool, optional): Force the update of the ticks. Defaults to False.
        """
        self.tstart = max([dh.tstart for dh in self.data_holders])
        self.tstop = min([dh.tstop for dh in self.data_holders])

    def add_data(data_holders: List[DataHolder]):
        """Add new data series to the handler.
        Adding a data serie will update the time boundaries of the handler. You can set custom boundaries when loading the data.
        """
        if len(time_series) != len(data_series):
            raise ValueError(
                "Time series and data series must have the same length. Make sure each data serie is passed with corresponding time serie.")
        self.data_holders.append(data_holders)
        self.update_bounds()

    def __add__(self, other):
        """Adds the series of the other handler to the current handler and updates the time related variables.

        Raises:
            ValueError: If the time step of the two handlers are not the same.
        """
        if not isinstance(other, MultiModalDataHandler):
            raise ValueError("Can only add MultiModalDataHandler objects.")
        if self.tstep != other.tstep:
            raise ValueError("Can only add MultiModalDataHandler objects with same tstep.")
        self.data_holders += other.data_holders
        self.tstart = max(self.tstart, other.tstart)
        self.tstop = min(self.tstop, other.tstop)
        return self

    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.data_holders)} data holders. Tstep: {self.tstep}ms. Tstart: {self.tstart}ms. Tstop: {self.tstop}ms."

    def sample(self, bounds: Tuple[int, int] = None, tstep=None, interpolate: str = None):
        """Get the data from the data series.

        Args:
            bounds (Tuple[int, int], optional): Time bounds of the data to load. Defaults to None, which means all the data will be loaded.

        Returns:
        """
        if interpolate is None:
            interpolate = self.interpolate
        if bounds is None:
            bounds = (self.tstart, self.tstop)
        else:
            # Take the inner intersection of bounds and discrete time serie, so that all data returned by get is aligned on the same time serie.
            bounds = (max(self.tstard, bounds[0] - (bounds[0] + self.tstep - 1 - self.tstart) // self.tstep),
                      min(self.tstop, bounds[1] + (bounds[1] + self.tstep - 1 - self.tstart) % self.tstep))

        if tstep is None:
            tstep = self.tstep

        if tstep is None:
            raise ValueError("Time step must be set to sample data.")

        ticks = np.arange(bounds[0], bounds[1], tstep)
        for i, holder in enumerate(self.data_holders):
            # TODO add custom properties fn(i), For instance interpolation type
            data = None
            if interpolate == 'repeat':
                holder._sample_repeat(ticks)
            else:
                raise ValueError(f"Interpolation method {interpolate} not supported.")
        return ticks

    def get_slice(self, bounds):
        """Get a slice of the data handler.

        Args:
            bounds (Tuple[int, int]): Time bounds of the data to load.

        Returns:
            MultiModalDataHandler: A new data handler with the same time step and the same data holders.
        """
        out = MultiModalSlice(tstep=self.tstep, tstart=bounds[0], tstop=bounds[1], data_holders=self.data_holders)
        return out

    def to_pcd(self, frame_number, show_holder=None):
        """Generate a list of open3d point clouds from data holders

        Returns:
            List[open3d.geometry.PointCloud]: List of point clouds.
        """
        if show_holder is None:
            show_holder = [True] * len(self.data_holders)
        pcd = []
        for i, holder in enumerate(self.data_holders):
            if show_holder[i]:
                pcd.append(holder.to_pcd(frame_number))
        return pcd
