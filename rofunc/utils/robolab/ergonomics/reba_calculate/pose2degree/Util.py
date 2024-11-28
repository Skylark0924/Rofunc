import math
import numpy as np


 # multiplication of two quaternion(representing the orientation of joints)
def multiply_two_quaternion(q1, q2):
        a = q1[0]
        b = q1[1]
        c = q1[2]
        d = q1[3]
        e = q2[0]
        f = q2[1]
        g = q2[2]
        h = q2[3]

        m0 = round(a * e - b * f - c * g - d * h, 1)
        m1 = round(b * e + a * f + c * h - d * g, 1)
        m2 = round(a * g - b * h + c * e + d * f, 1)
        m3 = round(a * h + b * g - c * f + d * e, 1)
        return [m0, m1, m2, m3]


def get_angle_between_degs(v1, v2):
    len_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    len_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

    result = math.acos(round(np.dot(v1, v2) / (len_v1 * len_v2), 3)) * 180 / math.pi
    return result


def get_distance_between(p1, p2):
    result = [x + y for x, y in zip(p2, np.dot(p1, -1))]
    return math.sqrt(result[0] ** 2 + result[1] ** 2 + result[2] ** 2)


def normalization(vector):
    l = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if l == 0:
        l += 0.01
    normal_vector = [vector[0] / l, vector[1] / l, vector[2] / l]
    return normal_vector


def find_rotation_quaternion(outer_quaternion, inner_quaternion):
    conjucate = [outer_quaternion[0], -outer_quaternion[1], -outer_quaternion[2], -outer_quaternion[3]]
    length = math.sqrt(outer_quaternion[0] ** 2 + outer_quaternion[1] ** 2 +
                       outer_quaternion[2] ** 2 + outer_quaternion[3] ** 2)
    if length ==0:
        inverse =[0,0,0,-1]
    else:
        inverse = np.dot(conjucate, (1 / length))
    rotation = multiply_two_quaternion(inner_quaternion, inverse)
    return rotation