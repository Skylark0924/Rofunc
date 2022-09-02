from signal import signal, SIGINT
import pyzed.sl as sl
import threading
import time
import signal
import logging
import os

exp_name = '20220630_1552_tactile'
if not os.path.exists('data/{}'.format(exp_name)):
    os.mkdir('data/{}'.format(exp_name))

zed_list = []
left_list = []
depth_list = []
timestamp_list = []
thread_list = []
stop_signal = False
recording_param_list = []

logging.basicConfig(filename='data/{}/{}.log'.format(exp_name, exp_name), filemode='a', level=logging.INFO)


def get_intrinsic_parameters(cam, serial_number):
    logging.info('Serial_number: {}'.format(serial_number))
    calibration_params = cam.get_camera_information().camera_configuration.calibration_parameters
    # Focal length of the left eye in pixels
    focal_left_x = calibration_params.left_cam.fx
    # First radial distortion coefficient
    k1 = calibration_params.left_cam.disto[0]
    # Translation between left and right eye on z-axis

    t = calibration_params.T
    # Horizontal field of view of the left eye in degrees
    h_fov = calibration_params.left_cam.h_fov

    logging.info('focal_left_x: {}'.format(focal_left_x))
    logging.info('k1: {}'.format(k1))
    logging.info('tz: {}'.format(t))
    logging.info('h_fov: {}'.format(h_fov))

    return focal_left_x, k1, t, h_fov


def signal_handler(signal, frame):
    global stop_signal
    global zed_list
    print('ZED', zed_list)
    for cam in zed_list:
        cam.disable_recording()
        cam.close()
    stop_signal = True
    time.sleep(0.5)
    exit()


def grab_run(index):
    global stop_signal
    global zed_list
    global timestamp_list
    global left_list
    global depth_list
    global recording_param_list

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
            # zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
            # zed_list[index].retrieve_measure(depth_list[index], sl.MEASURE.DEPTH)
            # timestamp_list[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
        # time.sleep(0.001)  # 1ms
    # zed_list[index].close()


def main():
    global stop_signal
    global zed_list
    global left_list
    global depth_list
    global timestamp_list
    global thread_list
    # global path
    global recording_param_list
    signal.signal(signal.SIGINT, signal_handler)

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

        video_path = 'data/{}/{}.svo'.format(exp_name, cam.serial_number)
        recording_param_list.append(sl.RecordingParameters(video_path, sl.SVO_COMPRESSION_MODE.H264))
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        get_intrinsic_parameters(zed_list[index], cam.serial_number)

        index = index + 1

    # Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()

    # # Display camera images
    # key = ''
    # while key != 113:  # for 'q' key
    #     for index in range(0, len(zed_list)):
    #         if zed_list[index].is_opened():
    #             if (timestamp_list[index] > last_ts_list[index]):
    #                 cv2.imshow(name_list[index], left_list[index].get_data())
    #                 x = round(depth_list[index].get_width() / 2)
    #                 y = round(depth_list[index].get_height() / 2)
    #                 err, depth_value = depth_list[index].get_value(x, y)
    #                 if np.isfinite(depth_value):
    #                     print("{} depth at center: {}MM".format(name_list[index], round(depth_value)))
    #                 last_ts_list[index] = timestamp_list[index]
    #     key = cv2.waitKey(10)
    # cv2.destroyAllWindows()

    # Stop the threads
    stop_signal = True
    for index in range(0, len(thread_list)):
        thread_list[index].join()

    print("\nFINISH")


# cam = sl.Camera()
#
#
# def handler(signal_received, frame):
#     cam.disable_recording()
#     cam.close()
#     sys.exit(0)
#
#
# signal(SIGINT, handler)
#
#
# def main(path):
#     # if not sys.argv or len(sys.argv) != 2:
#     #     print("Only the path of the output SVO file should be passed as argument.")
#     #     exit(1)
#
#     init = sl.InitParameters()
#     init.camera_resolution = sl.RESOLUTION.HD1080
#     init.depth_mode = sl.DEPTH_MODE.NONE
#
#     status = cam.open(init)
#     if status != sl.ERROR_CODE.SUCCESS:
#         print(repr(status))
#         exit(1)
#
#     path_output = path
#     recording_param = sl.RecordingParameters(path_output, sl.SVO_COMPRESSION_MODE.H264)
#     err = cam.enable_recording(recording_param)
#     if err != sl.ERROR_CODE.SUCCESS:
#         print(repr(status))
#         exit(1)
#
#     runtime = sl.RuntimeParameters()
#     print("SVO is Recording, use Ctrl-C to stop.")
#     frames_recorded = 0
#
#     while True:
#         if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
#             frames_recorded += 1
#             print("Frame count: " + str(frames_recorded), end="\r")


if __name__ == "__main__":
    main()
