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
