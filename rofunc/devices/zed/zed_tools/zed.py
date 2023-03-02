import pyzed.sl as sl

from common import load_omega_config


class DeviceBase(object):
    def __init__(self, config_name):
        super(DeviceBase, self).__init__()
        self.device_config = load_omega_config(config_name)
        self.device_id = self.device_config["id"]


class ZED(DeviceBase):
    def __init__(self, config_name):
        super(ZED, self).__init__(config_name)
        self.zed = sl.Camera()
        self._init_parameters = sl.InitParameters()
        
    def set_parameters(self):
        """
        The parameters are defined in the config file.
        Returns:

        """
        self._init_parameters.camera_fps = self.device_config["camera_fps"]
        self._init_parameters.camera_resolution = self.device_config["Resolution"]
        self._init_parameters.depth_mode = self.device_config["depth_mode"]
        self._init_parameters.sdk_gpu_id = self.device_config["sdk_gpu_id"]
        self._init_parameters.camera_image_flip = self.device_config[
            "camera_image_flip"
        ]
