import pyzed.sl as sl
import os
import shutil
import json
import sys


def parse_camera_parameters(zed):
    calibration_params = zed.get_camera_information().calibration_parameters
    settings_dict = {
        'left_fx': calibration_params.left_cam.fx,
        'left_fy': calibration_params.left_cam.fy,
        'left_cx': calibration_params.left_cam.cx,
        'left_cy': calibration_params.left_cam.cy,
        'left_disto': calibration_params.left_cam.disto.tolist(),
        # numpy array Distortion factor : [ k1, k2, p1, p2, k3 ]. Radial (k1,k2,k3) and Tangential (p1,p2) distortion.

        'right_fx': calibration_params.right_cam.fx,
        'right_fy': calibration_params.right_cam.fy,
        'right_cx': calibration_params.right_cam.cx,
        'right_cy': calibration_params.right_cam.cy,
        'right_disto': calibration_params.right_cam.disto.tolist(),

        'translation': calibration_params.T.tolist(),
        'rotation': calibration_params.R.tolist(),
        'stereo_transform': calibration_params.stereo_transform.m.tolist(),

        'resolution_width': zed.get_camera_information().camera_resolution.width,
        'resolution_height': zed.get_camera_information().camera_resolution.height,
        'fps': zed.get_camera_information().camera_fps,
        'num_frames': zed.get_svo_number_of_frames(),
        'timestamp': zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_milliseconds(),
        'depth_confidence_threshold': zed.get_runtime_parameters().confidence_threshold,
        'depth_min_range': zed.get_init_parameters().depth_minimum_distance,
        'depth_max_range': zed.get_init_parameters().depth_maximum_distance,
        'sdk_version': zed.get_sdk_version()

    }
    return settings_dict


def get_pose(zed, zed_pose, zed_sensors):
    # Get the pose of the left eye of the camera with reference to the world frame
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
    # zed_imu = zed_sensors.get_imu_data()

    # Display the translation and timestamp
    py_translation = sl.Translation()
    tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
    ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
    tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
    print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz,
                                                                           zed_pose.timestamp.get_milliseconds()))

    # Display the orientation quaternion
    py_orientation = sl.Orientation()
    ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
    oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
    print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

    pose_dict = {'Translation': {'Tx': tx, 'Ty': ty, 'Tz': tz},
                 'Orientation': {'Ox': ox, 'Oy': oy, 'Oz': oz, 'Ow': ow}}

    return pose_dict


def main():
    # if len(sys.argv) != 2:
    #     print("Please specify path to .svo file.")
    #     exit()

    # filepath = sys.argv[1]
    filepath = 'data/20220627_1218/31021548.svo'
    print("Reading SVO file: {0}".format(filepath))

    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime_parameters = sl.RuntimeParameters()

    # initialize images and point cloud
    i = 0
    image = sl.Mat()
    image_r = sl.Mat()
    pointcloud = sl.Mat()

    # Enable positional tracking with default parameters
    py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
    tracking_parameters = sl.PositionalTrackingParameters(init_pos=py_transform)
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    zed_pose = sl.Pose()
    zed_sensors = sl.SensorsData()

    # set up output directory and delete old output
    print("clear old output")
    dir_path = "output"
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))

    # number of frames in the recording
    nb_frames = zed.get_svo_number_of_frames()
    nb_frames = 5

    # main loop
    while True:  # change to True
        print("Doing {}".format(i))
        # path for images
        output_dir = os.path.join(dir_path, "frame_{}/images".format(i))
        pc_dir = os.path.join(dir_path, "frame_{}/pointcloud".format(i))
        pose_dir = os.path.join(dir_path, "frame_{}/pose".format(i))

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(pc_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)

        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            svo_position = zed.get_svo_position()

            # A new image is available if grab() returns SUCCESS
            print("Writing images")
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_image(image_r, sl.VIEW.RIGHT)
            image.write(os.path.join(output_dir, 'left_image.png'))
            image_r.write(os.path.join(output_dir, 'right_image.png'))

            # retrive and write point cloud
            print("Writing point cloud of resolution")
            zed.retrieve_measure(pointcloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            pointcloud.write(os.path.join(pc_dir, 'pointcloud.ply'))

            # retrieve and write pose
            pose_dict = get_pose(zed, zed_pose, zed_sensors)
            pose_filepath = os.path.join(pose_dir, 'pose.json')
            with open(pose_filepath, 'w') as outfile:
                json.dump(pose_dict, outfile)

            settings_dict = parse_camera_parameters(zed)
            settings_filepath = os.path.join(pose_dir, 'settings.json')
            with open(settings_filepath, 'w') as outfile:
                json.dump(settings_dict, outfile)

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

        i = i + 1

    # Close the camera
    zed.close()


if __name__ == "__main__":
    main()