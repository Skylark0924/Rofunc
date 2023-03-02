import time

import cv2 as cv
import numpy as np
import os.path as osp

from functools import wraps
from omegaconf import OmegaConf


def _preprocess_print(*args):
    """Preprocess the input for colorful printing.

    Args:
        args: Any/None One or more any type arguments to print.

    Returns:
        str Msg to print.
    """
    str_args = ""
    for a in args:
        if isinstance(a, np.ndarray):
            str_args += "\n" + np.array2string(a, separator=", ")
        else:
            str_args += " " + str(a)
    separate_with_newline = str_args.split("\n")
    extra_whitespaces_removed = []
    for b in separate_with_newline:
        extra_whitespaces_removed.append(" ".join(b.split()))
    return "\n".join(extra_whitespaces_removed)


def print_debug(*args):
    """Print information with green."""
    print("".join(["\033[1m\033[92m", _preprocess_print(*args), "\033[0m"]))


def print_info(*args):
    """Print information with sky blue."""
    print("".join(["\033[1m\033[94m", _preprocess_print(*args), "\033[0m"]))


def print_warning(*args):
    """Print a warning with yellow."""
    print("".join(["\033[1m\033[93m", _preprocess_print(*args), "\033[0m"]))


def print_error(*args):
    """Print error with red."""
    print("".join(["\033[1m\033[91m", _preprocess_print(*args), "\033[0m"]))


def omega_to_list(omega):
    return OmegaConf.to_object(omega)


def load_omega_config(config_name):
    """Load the configs listed in config_name.yaml.

    Args:
        config_name: str Name of the config file.

    Returns:
        dict A dict of configs.
    """
    return OmegaConf.load(
        osp.join(osp.dirname(__file__), "../../config/{}.yaml".format(config_name))
    )


def update_omega_config(config_name, key, value):
    """Update the config_name specified config file's one item.
    If the config has not been created, create the file and write key-value. If it has been created:
    1) If the item `key` is not created, create this key and assign its with `value`;
    2) If the `key` is already there, override its value with `value`.

    Args:
        config_name: str Name of the config file not including suffix such as .yaml
        key: Any Key of the item.
        value: Any Value of the item `key`.

    Returns:
        None
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    config_item = OmegaConf.create({key: value})
    config_file = osp.join(osp.dirname(__file__), f"../../config/{config_name}.yaml")
    if osp.exists(config_file):
        loaded = OmegaConf.load(config_file)
    else:
        loaded = None
    if loaded:
        if key in loaded:
            OmegaConf.update(loaded, key, value)
        else:
            loaded = OmegaConf.merge(config_item, loaded)
    else:
        loaded = config_item
    OmegaConf.save(loaded, f=config_file)


def pretty_print_configs(cfgs):
    """Print a dict of configurations in a visual friendly and organized way.

    Args:
        cfgs: dict A dict of configures. The items could be string, number, or a list/tuple.

    Returns:
        None
    """
    max_key_len = 0
    max_value_len = 0
    for key, value in cfgs.items():
        key_str = "{}".format(key)
        if len(key_str) > max_key_len:
            max_key_len = len(key_str)
        if isinstance(value, list) or isinstance(value, tuple):
            for i in value:
                i_str = "{}".format(i)
                if len(i_str) > max_value_len:
                    max_value_len = len(i_str)
        else:
            value_str = "{}".format(value)
            if len(value_str) > max_value_len:
                max_value_len = len(value_str)

    print_info(
        "\n{}{}{}".format(
            "=" * (max_key_len + 1), " ROPORT CONFIGS ", "=" * (max_value_len - 15)
        )
    )
    for key, value in cfgs.items():
        key_msg = "{message: <{width}}".format(message=key, width=max_key_len)
        empty_key_msg = "{message: <{width}}".format(message="", width=max_key_len)
        if isinstance(value, list) or isinstance(value, tuple):
            for i, i_v in enumerate(value):
                if i == 0:
                    print_info("{}: {}".format(key_msg, i_v))
                else:
                    print_info("{}: {}".format(empty_key_msg, i_v))
        else:
            print_info("{}: {}".format(key_msg, value))
    print_info(
        "{}{}{}\n".format(
            "=" * (max_key_len + 1), " END OF CONFIGS ", "=" * (max_value_len - 15)
        )
    )


def is_array_like(array):
    if isinstance(array, str):
        return False
    return hasattr(array, "__len__") and hasattr(array, "__iter__")


def expect_any_input(hint):
    if not isinstance(hint, str):
        hint = str(hint)
    if not hint.endswith(" "):
        hint += " "
    return input(hint)


def expect_yes_no_input(hint, is_yes_default=True):
    """Get user input for a yes/no choice.

    Args:
        hint: str Hint for the user to input.
        is_yes_default: bool If true, 'yes' will be considered as the default when empty input was given.
                        Otherwise, 'no' will be considered as the default choice.

    Returns:
        bool If the choice matches the default.
    """
    if is_yes_default:
        suffix = "(Y/n):"
        default = "yes"
    else:
        suffix = "(y/N):"
        default = "no"
    flag = input(" ".join((hint, suffix))).lower()

    expected_flags = ["", "y", "n"]
    while flag not in expected_flags:
        print_warning(
            f"Illegal input {flag}, valid inputs should be Y/y/N/n or ENTER for the default: {default}"
        )
        return expect_yes_no_input(hint, is_yes_default)

    if is_yes_default:
        return flag != "n"
    else:
        return flag != "y"


def is_float_compatible(string):
    """Check if the string can be converted to a float.

    Args:
        string: str Input string.

    Returns:
        True if the string can be converted to a float, false otherwise.
    """
    string = string.lstrip("-")
    s_dot = string.split(".")
    if len(s_dot) > 2:
        return False
    s_e_plus = string.split("e+")
    if len(s_e_plus) == 2:
        return is_float_compatible("".join(s_e_plus))
    s_e_minus = string.split("e-")
    if len(s_e_minus) == 2:
        return is_float_compatible("".join(s_e_minus))
    s_e = string.split("e")
    if len(s_e) == 2:
        return is_float_compatible("".join(s_e))

    for si in s_dot:
        if not si.isdigit():
            return False
    return True


def expect_float_input(hint):
    """Get user input for obtaining a float number.

    Args:
        hint: str Hint for the user to input.

    Returns:
        float
    """
    user_input = expect_any_input(hint)
    while not is_float_compatible(user_input):
        print_warning(
            f"Illegal input '{user_input}', valid input should be a float number"
        )
        return expect_float_input(hint)
    return float(user_input)


def get_image_hwc(image):
    """Get the height, width, and channel of a ndarray image.

    Args:
        image (ndarray): The image.

    Returns:
        [int, int, int] Height, width, channel.
    """
    assert isinstance(image, np.ndarray), print_error(
        f"Image type is not ndarray but {type(image)}"
    )
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
    elif len(image.shape) == 3:
        if image.shape[0] == 3:
            h, w = image.shape[1:]
        elif image.shape[-1] == 3:
            h, w = image.shape[:2]
        else:
            raise ValueError(f"Image of shape {image.shape} is not supported")
        c = 3
    else:
        raise ValueError(f"Image of shape {image.shape} is not supported")
    return h, w, c


def get_circle_centers_and_radii(img):
    """Get the center position and radius of circles in a numpy image.

    Args:
        img (ndarray): The image.

    Returns:
        centers: array (N, 2) The center positions in pixel coordinate system of N circles
        radii: array (N, ) The radii in pixel coordinate system of N circles
    """
    assert isinstance(img, np.ndarray), print_error(
        f"Image type is not ndarray but {type(img)}"
    )
    centers = []
    radii = []
    gray_img = cv.medianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 5)
    rows, columns, _ = get_image_hwc(gray_img)
    circles = cv.HoughCircles(
        gray_img,
        cv.HOUGH_GRADIENT,
        1,
        rows / 8,
        param1=100,
        param2=30,
        minRadius=80,
        maxRadius=300,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            centers.append([circle[0], circle[1]])
            radii.append([circle[2]])
        return centers, radii
    return None, None


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print_info(f"Function {func.__name__} took {total_time:.3f} seconds")
        return result

    return timeit_wrapper


def find_two_masks_overlaparea(MaskOne, MaskTwo):
    """

    Args:
       MaskOne: list[list[bool/int]] Mask of the bounding box.
       MaskTwo: list[list[bool/int]] Mask of the bounding box.

    Returns:
        mask_combined_area: int[pixel] The num of combined area pixels between two masks
        .
    """
    mask_combined = cv.bitwise_and(MaskOne, MaskTwo)
    mask_combined_area = cv.countNonZero(mask_combined)
    return mask_combined_area


def define_mask_rectangle_in_image(
    image, rectangle_width, rectangle_height, center_point, angle
):
    """
    In a retangle, after rotating counterclockwise degree θ from the center point, the points coordinates are:
    x′ = (x0 － xcenter) cosθ － (y0 － ycenter) sinθ ＋ xcenter;
    y′ = (x0 － xcenter) sinθ ＋ (y0 － ycenter) cosθ ＋ ycenter;

    In opencv coordinate, the functions are refactored as
    x′ = (x0 － xcenter) cos(pi/180*θ) － (y0 － ycenter) sin(pi/180*θ) ＋ xcenter;
    y0 = row -y0
    ycenter = row -ycenter
    y′ = (x0 － xcenter) sin(pi/180*θ) ＋ (y0 － ycenter) cos(pi/180*θ) ＋ ycenter;
    y' = row - y'
    Args:
        image (ndarray): The image.
        rectangle_width: The width of the targeted rectangle.
        rectangle_height: The height of the targeted rectangle.
        center_point: The center point of the targeted rectangle.
        angle: The Rotation angle
    Returns:
        mask_rectangle: Mask of the targeted rectangle.

    """
    image_height, image_width, _ = get_image_hwc(image)
    mask_rectangle = np.zeros((image_height, image_width), dtype="uint8")
    retangle_pts = np.array(
        [
            [
                center_point[0] - rectangle_height / 2,
                center_point[1] - rectangle_width / 2,
            ],
            [
                center_point[0] + rectangle_height / 2,
                center_point[1] - rectangle_width / 2,
            ],
            [
                center_point[0] + rectangle_height / 2,
                center_point[1] + rectangle_width / 2,
            ],
            [
                center_point[0] - rectangle_height / 2,
                center_point[1] + rectangle_width / 2,
            ],
        ],
    )
    center_point[1] = image_height - center_point[1]
    for i, points in enumerate(retangle_pts):
        points[1] = image_height - points[1]
        # Convert image coordinates to plane coordinates
        rotated_retangle_points_x = (
            (points[0] - center_point[0]) * np.cos(np.pi / 180.0 * angle)
            - (points[1] - center_point[1]) * np.sin(np.pi / 180.0 * angle)
            + center_point[0]
        )
        rotated_retangle_points_y = (
            (points[0] - center_point[0]) * np.sin(np.pi / 180.0 * angle)
            + (points[1] - center_point[1]) * np.cos(np.pi / 180.0 * angle)
            + center_point[1]
        )
        # Convert plane coordinates to image coordinates
        rotated_retangle_points_y = image_height - rotated_retangle_points_y
        retangle_pts[i] = [
            int(rotated_retangle_points_x),
            int(rotated_retangle_points_y),
        ]
    retangle_pts = np.array(retangle_pts, dtype=np.int32)
    cv.fillPoly(mask_rectangle, pts=[retangle_pts], color=(255, 255, 255))

    return mask_rectangle


def add_mask_from_detection_output(image, detection_output, *friend_masks):
    """

    Args:
        image:
        detection_output:
        friend_mask:

    Returns:

    """
    image_height, image_width, _ = get_image_hwc(image)
    sum_mask = np.zeros((image_height, image_width), dtype="uint8")
    for i, [label, _, mask_obj, _] in enumerate(detection_output):
        if friend_masks:
            for friend_mask in friend_masks:
                if cv.countNonZero(mask_obj.mask) != cv.countNonZero(friend_mask):
                    sum_mask = cv.bitwise_or(sum_mask, mask_obj.mask)
        else:
            sum_mask = cv.bitwise_or(sum_mask, mask_obj.mask)
    ret, sum_mask = cv.threshold(sum_mask, 0.9, 255, cv.THRESH_BINARY)
    return sum_mask
