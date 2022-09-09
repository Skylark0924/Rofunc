import os.path

import pyzed.sl as sl
import cv2

saving_cnt = 0


def playback(filepath):
    global saving_cnt
    print("Reading SVO file: {0}".format(filepath))
    root_path = filepath.split('.')[0]

    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    key = ''
    print("  Save the current image:     s")
    print("  Quit the video reading:     q\n")
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat)
            cv2.imshow("ZED", mat.get_data())
            key = cv2.waitKey(1)
            saving_image(root_path, key, mat)
        else:
            key = cv2.waitKey(1)
    cv2.destroyAllWindows()

    print_camera_information(cam)
    cam.close()
    print("\nFINISH")


def print_camera_information(cam):
    while True:
        res = input("Do you want to display camera information? [y/n]: ")
        if res == "y":
            # print(cam.get_sensors_data())
            print("Distorsion factor of the right cam before calibration: {0}.".format(
                cam.get_camera_information().calibration_parameters_raw.right_cam.disto))
            print("Distorsion factor of the right cam after calibration: {0}.\n".format(
                cam.get_camera_information().calibration_parameters.right_cam.disto))

            print("Confidence threshold: {0}".format(cam.get_runtime_parameters().confidence_threshold))
            print("Depth min and max range values: {0}, {1}".format(cam.get_init_parameters().depth_minimum_distance,
                                                                    cam.get_init_parameters().depth_maximum_distance)
                  )
            print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2),
                                                 cam.get_camera_information().camera_resolution.height))
            print("Camera FPS: {0}".format(cam.get_camera_information().camera_fps))
            print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))
            break
        elif res == "n":
            print("Camera information not displayed.\n")
            break
        else:
            print("Error, please enter [y/n].\n")


def saving_image(root_path, key, mat):
    global saving_cnt
    if key == 115:
        img = sl.ERROR_CODE.FAILURE
        while img != sl.ERROR_CODE.SUCCESS:
            saving_img_dir = root_path + '_saving_img'
            if not os.path.exists(saving_img_dir):
                os.mkdir(saving_img_dir)
            filepath = os.path.join(saving_img_dir, '{}.png'.format(saving_cnt))
            img = mat.write(filepath)
            print("Saving image {}.png : {}".format(saving_cnt, repr(img)))
            if img == sl.ERROR_CODE.SUCCESS:
                saving_cnt += 1
                break
            else:
                print("Help: you must enter the filepath + filename + PNG extension.")


if __name__ == '__main__':
    import rofunc as rf

    rf.zed.playback('/home/ubuntu/Data/06_24/Video/20220624_1649/38709363.svo')
