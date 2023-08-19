import logging
import os
import signal
import threading
import time
import rofunc as rf

zed_list = []


def get_intrinsic_parameters(cam):
    """Get the intrinsic parameters of the camera.
    :param cam: the camera object
    :return: F, C, K, P, T : the focal length, the principal point, the radial distortion coefficients, the tangential
    distortion coefficients, the translation vector
    """
    calibration_params = cam.get_camera_information().camera_configuration.calibration_parameters
    # Focal length of the left eye in pixels
    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy

    # Principal point of the left eye in pixels
    cx = calibration_params.left_cam.cx
    cy = calibration_params.left_cam.cy

    # First radial distortion coefficient
    k1 = calibration_params.left_cam.disto[0]
    k2 = calibration_params.left_cam.disto[1]
    k3 = calibration_params.left_cam.disto[2]
    p1 = calibration_params.left_cam.disto[3]
    p2 = calibration_params.left_cam.disto[4]

    # Translation between left and right eye on z-axis
    T = calibration_params.T

    return (fx, fy), (cx, cy), (k1, k2, k3), (p1, p2), T


def signal_handler(signal, frame):
    global zed_list
    print('ZED', zed_list)
    for cam in zed_list:
        cam.disable_recording()
        cam.close()
    time.sleep(0.5)
    exit()


def grab_run(zed_list, recording_param_list, index):
    import pyzed.sl as sl

    err = zed_list[index].enable_recording(recording_param_list[index])
    if err != sl.ERROR_CODE.SUCCESS:
        print('Wrong', index)
        exit(1)

    runtime = sl.RuntimeParameters()
    print("Camera {} SVO is Recording, use Ctrl-C to stop.".format(index))
    frames_recorded = 0

    while True:
        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            frames_recorded += 1
            print("Camera: " + str(index) + "Frame count: " + str(frames_recorded), end="\r")


def record(root_dir, exp_name):
    import pyzed.sl as sl

    global zed_list

    if os.path.exists('{}/{}'.format(root_dir, exp_name)):
        raise Exception('There are already some files in {}, please rename the exp_name.'.format(
            '{}/{}'.format(root_dir, exp_name)))
    else:
        rf.oslab.create_dir('{}/{}'.format(root_dir, exp_name))
        rf.logger.beauty_print('Recording folder: {}/{}'.format(root_dir, exp_name), type='info')

    left_list = []
    depth_list = []
    timestamp_list = []
    thread_list = []
    recording_param_list = []

    # global path
    signal.signal(signal.SIGINT, signal_handler)
    logging.basicConfig(filename='{}/{}/parameter.log'.format(root_dir, exp_name), filemode='a', level=logging.INFO)

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 30  # The framerate is lowered to avoid any USB3 bandwidth issues
    # List and open cameras
    name_list = []
    last_ts_list = []
    cameras = sl.Camera.get_device_list()
    index = 0
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        if index >= 2:
            init.sdk_gpu_id = 1
        else:
            init.sdk_gpu_id = 0
        name_list.append("ZED {}".format(cam.serial_number))
        print("Opening {}".format(name_list[index]))

        zed_list.append(sl.Camera())
        left_list.append(sl.Mat())
        depth_list.append(sl.Mat())
        timestamp_list.append(0)
        last_ts_list.append(0)

        video_path = '{}/{}/{}.svo'.format(root_dir, exp_name, cam.serial_number)
        recording_param_list.append(sl.RecordingParameters(video_path, sl.SVO_COMPRESSION_MODE.H264))
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()

        # Log the intrinsic_parameters of each camera
        F, C, K, P, T = get_intrinsic_parameters(zed_list[index])
        logging.info('Serial_number: {}'.format(cam.serial_number))
        logging.info('focal_length: {}'.format(F))
        logging.info('principal_point: {}'.format(C))
        logging.info('radial_dist: {}'.format(K))
        logging.info('tangent_dist: {}'.format(P))
        logging.info('T: {}'.format(T))

        index = index + 1

    # Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(zed_list, recording_param_list, index,)))
            thread_list[index].start()

    # Stop the threads
    for index in range(0, len(thread_list)):
        thread_list[index].join()

    print("\nFINISH")
