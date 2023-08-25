import enum
import multiprocessing
import os
import sys

import cv2
import numpy as np
import rofunc as rf


class AppType(enum.Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3


def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def export(filepath, mode=1):
    """
    Export the svo file with specific mode.
    Args:
        filepath: SVO file path (input) : path/to/file.svo
        mode: Export mode:  0=Export LEFT+RIGHT AVI.
                            1=Export LEFT+DEPTH_VIEW AVI.
                            2=Export LEFT+RIGHT image sequence.
                            3=Export LEFT+DEPTH_VIEW image sequence.
                            4=Export LEFT+DEPTH_16Bit image sequence.

    Returns:

    """
    import pyzed.sl as sl

    # Get input parameters
    svo_input_path = filepath
    root_path = filepath.split('.svo')[0]
    output_dir = root_path + '_export'
    rf.oslab.create_dir(output_dir)

    output_as_video = True
    if mode == 0:
        app_type = AppType.LEFT_AND_RIGHT
        mode_name = 'left_and_right.avi'
    elif mode == 1:
        app_type = AppType.LEFT_AND_DEPTH
        mode_name = 'left_and_depth.avi'
    elif mode == 2:
        app_type = AppType.LEFT_AND_RIGHT
        mode_name = 'left_and_right_img_seq'
    elif mode == 3:
        app_type = AppType.LEFT_AND_DEPTH
        mode_name = 'left_and_depth_img_seq'
    elif mode == 4:
        app_type = AppType.LEFT_AND_DEPTH_16
        mode_name = 'left_and_depth_16_img_seq'
    else:
        raise Exception('Wrong mode index, should be one of [0, 1, 2, 3, 4]')

    output_path = os.path.join(output_dir, mode_name)

    # Check if exporting to AVI or SEQUENCE
    if mode != 0 and mode != 1:
        output_as_video = False

    if not output_as_video:
        rf.oslab.create_dir(output_path)

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    width = image_size.width
    height = image_size.height
    width_sbs = width * 2

    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat()

    video_writer = None
    if output_as_video:
        # Create video writer with MPEG-4 part 2 codec
        video_writer = cv2.VideoWriter(str(output_path),
                                       cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                       max(zed.get_camera_information().camera_fps, 25),
                                       (width_sbs, height))

        if not video_writer.isOpened():
            sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write "
                             "permissions.\n")
            zed.close()
            exit()

    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.FILL

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO to {}. Use Ctrl-C to interrupt conversion.\n".format(mode_name))

    nb_frames = zed.get_svo_number_of_frames()
    time_table = []

    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)  # Get the image timestamp

            if app_type == AppType.LEFT_AND_RIGHT:
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            elif app_type == AppType.LEFT_AND_DEPTH:
                zed.retrieve_image(right_image, sl.VIEW.DEPTH)
            elif app_type == AppType.LEFT_AND_DEPTH_16:
                zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            if output_as_video:
                # Copy the left image to the left side of SBS image
                svo_image_sbs_rgba[0:height, 0:width, :] = left_image.get_data()

                # Copy the right image to the right side of SBS image
                svo_image_sbs_rgba[0:, width:, :] = right_image.get_data()

                # Convert SVO image from RGBA to RGB
                ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba, cv2.COLOR_RGBA2RGB)

                # Write the RGB image in the video
                video_writer.write(ocv_image_sbs_rgb)
            else:
                # Generate file names
                filename1 = os.path.join(output_path, "left_{}.png".format(str(svo_position).zfill(6)))
                filename2 = os.path.join(output_path, "right_{}.png".format(
                    str(svo_position).zfill(6))) if app_type == AppType.LEFT_AND_RIGHT \
                    else os.path.join(output_path, "depth_{}.png".format(str(svo_position).zfill(6)))
                time_table.append((filename1, str(timestamp.get_milliseconds())))

                # Save Left images
                cv2.imwrite(str(filename1), left_image.get_data())

                if app_type != AppType.LEFT_AND_DEPTH_16:
                    # Save right images
                    cv2.imwrite(str(filename2), right_image.get_data())
                else:
                    # Save depth images (convert to uint16)
                    cv2.imwrite(str(filename2), depth_image.get_data().astype(np.uint16))

            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
        elif zed.grab(rt_param) == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break

    with open(os.path.join(output_dir, "time_table.txt"), "w") as f:
        f.write("filename,timestamp\n")
        for frame, timestamp in time_table:
            f.write("{},{}\n".format(frame, timestamp))

    if output_as_video:
        # Close the video writer
        video_writer.release()

    zed.close()
    return 0


def parallel(z):
    return export(z[0], z[1])


def export_batch(filedir, all_mode=True, mode_lst=None, core_num=10):
    files = os.listdir(filedir)
    pool = multiprocessing.Pool(core_num)

    if all_mode and mode_lst is None:
        filepaths = []
        for mode in range(5):
            for file in files:
                if file[-3:] == 'svo':
                    filepaths.append((os.path.join(filedir, file), mode))
        pool.map(parallel, filepaths)
    elif mode_lst is not None:
        filepaths = []
        for mode in mode_lst:
            for file in files:
                if file[-3:] == 'svo':
                    filepaths.append((os.path.join(filedir, file), mode))
            pool.map(parallel, filepaths)
    else:
        raise Exception('Wrong parameters')

    pool.close()
    pool.join()
