from .transform_tensor import *


def convert_ori_format(ori, src_format: str, tar_format: str):
    """
    Convert orientation format from source to target format.

    :param ori: either quaternion, rotation matrix or euler angles
    :param src_format: source format
    :param tar_format: target format
    :return: converted orientation format
    """
    assert src_format in ['quat', 'mat', 'euler'], "Unsupported source format."
    assert tar_format in ['quat', 'mat', 'euler'], "Unsupported target format."
    if src_format == tar_format:
        return ori
    if src_format == 'quat':
        if tar_format == 'mat':
            return rot_matrix_from_quat_tensor(ori)
        elif tar_format == 'euler':
            return euler_from_quat_tensor(ori)
        else:
            raise ValueError("Unsupported target format.")
    elif src_format == 'mat':
        if tar_format == 'quat':
            return quat_from_rot_matrix_tensor(ori)
        elif tar_format == 'euler':
            return euler_from_rot_matrix_tensor(ori)
        else:
            raise ValueError("Unsupported target format.")
    elif src_format == 'euler':
        if tar_format == 'quat':
            return quat_from_euler_tensor(ori)
        elif tar_format == 'mat':
            return rot_matrix_from_euler_tensor(ori)
        else:
            raise ValueError("Unsupported target format.")


def convert_quat_order(quat, src_order, tar_order):
    """
    Convert quaternion order from source to target order.

    :param quat:
    :param src_order:
    :param tar_order:
    :return:
    """
    assert src_order in ['wxyz', 'xyzw'], "Unsupported source order."
    assert tar_order in ['wxyz', 'xyzw'], "Unsupported target order."
    quat = check_quat_tensor(quat)
    if src_order == tar_order:
        return quat
    if src_order == 'wxyz':
        if tar_order == 'xyzw':
            return quat[:, [1, 2, 3, 0]]
    elif src_order == 'xyzw':
        if tar_order == 'wxyz':
            return quat[:, [3, 0, 1, 2]]