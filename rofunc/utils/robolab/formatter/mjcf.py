from typing import Union, Optional

import mujoco
import pytorch_kinematics.transforms as tf
from mujoco._structs import _MjModelBodyViews as MjModelBodyViews
from pytorch_kinematics import chain, frame

# Converts from MuJoCo joint types to pytorch_kinematics joint types
JOINT_TYPE_MAP = {
    mujoco.mjtJoint.mjJNT_HINGE: 'revolute',
    mujoco.mjtJoint.mjJNT_SLIDE: "prismatic"
}


def get_body_geoms(m: mujoco.MjModel, body: MjModelBodyViews, base: Optional[tf.Transform3d] = None):
    # Find all geoms which have body as parent
    base = base or tf.Transform3d()
    visuals = []
    for geom_id in range(m.ngeom):
        geom = m.geom(geom_id)
        if geom.bodyid == body.id:
            if geom.type == "capsule":
                param = (geom.size[0], geom.fromto)
            elif geom.type == "sphere":
                param = geom.size[0]
            else:
                param = geom.size
            visuals.append(frame.Visual(offset=tf.Transform3d(rot=geom.quat, pos=geom.pos), geom_type=geom.type,
                                        geom_param=param))
    return visuals


def body_to_link(body, base: Optional[tf.Transform3d] = None):
    base = base or tf.Transform3d()
    return frame.Link(body.name, offset=tf.Transform3d(rot=body.quat, pos=body.pos))


def joint_to_joint(joint, base: Optional[tf.Transform3d] = None):
    base = base or tf.Transform3d()
    return frame.Joint(
        joint.name,
        offset=tf.Transform3d(pos=joint.pos),
        joint_type=JOINT_TYPE_MAP[joint.type],
        axis=joint.axis,
    )


def add_composite_joint(root_frame, joints, base: Optional[tf.Transform3d] = None):
    base = base or tf.Transform3d()
    if len(joints) > 0:
        root_frame.children = root_frame.children + [
            frame.Frame(link=frame.Link(name=root_frame.link.name + "_child"), joint=joint_to_joint(joints[0], base))
        ]
        ret, offset = add_composite_joint(root_frame.children[-1], joints[1:])
        return ret, root_frame.joint.offset * offset
    else:
        return root_frame, root_frame.joint.offset


def _build_chain_recurse(m, parent_frame, parent_body):
    parent_frame.link.visuals = get_body_geoms(m, parent_body)
    # iterate through all bodies that are children of parent_body
    for body_id in range(m.nbody):
        body = m.body(body_id)
        if body.parentid == parent_body.id and body_id != parent_body.id:
            n_joints = body.jntnum
            if n_joints > 1:
                # Support for composite joints
                old_parent_frame = parent_frame
                for i in range(int(n_joints)):
                    joint = m.joint(body.jntadr[0] + i)
                    if i == 0:
                        joint_offset = tf.Transform3d(pos=joint.pos)
                        child_joint = frame.Joint(joint.name, offset=joint_offset, axis=joint.axis,
                                                  joint_type=JOINT_TYPE_MAP[joint.type[0]],
                                                  limits=(joint.range[0], joint.range[1]))
                    else:
                        child_joint = frame.Joint(joint.name, axis=joint.axis,
                                                  joint_type=JOINT_TYPE_MAP[joint.type[0]],
                                                  limits=(joint.range[0], joint.range[1]))
                    if i == 0:
                        child_link = frame.Link(body.name + "_" + str(i),
                                                offset=tf.Transform3d(rot=body.quat, pos=body.pos))
                    else:
                        child_link = frame.Link(body.name + "_" + str(i))
                    child_frame = frame.Frame(name=body.name + "_" + str(i), link=child_link, joint=child_joint)
                    parent_frame.children = parent_frame.children + [child_frame, ]
                    parent_frame = child_frame
                parent_frame = old_parent_frame
            elif n_joints == 1:
                # Find the joint for this body, again assuming there's only one joint per body.
                joint = m.joint(body.jntadr[0])
                joint_offset = tf.Transform3d(pos=joint.pos)
                child_joint = frame.Joint(joint.name, offset=joint_offset, axis=joint.axis,
                                          joint_type=JOINT_TYPE_MAP[joint.type[0]],
                                          limits=(joint.range[0], joint.range[1]))
                child_link = frame.Link(body.name, offset=tf.Transform3d(rot=body.quat, pos=body.pos))
                child_frame = frame.Frame(name=body.name, link=child_link, joint=child_joint)
                parent_frame.children = parent_frame.children + [child_frame, ]
            else:
                child_joint = frame.Joint(body.name + "_fixed_joint")
                child_link = frame.Link(body.name, offset=tf.Transform3d(rot=body.quat, pos=body.pos))
                child_frame = frame.Frame(name=body.name, link=child_link, joint=child_joint)
                parent_frame.children = parent_frame.children + [child_frame, ]
            _build_chain_recurse(m, child_frame, body)

    # # iterate through all sites that are children of parent_body
    # for site_id in range(m.nsite):
    #     site = m.site(site_id)
    #     if site.bodyid == parent_body.id:
    #         site_link = frame.Link(site.name, offset=tf.Transform3d(rot=site.quat, pos=site.pos))
    #         site_frame = frame.Frame(name=site.name, link=site_link)
    #         parent_frame.children = parent_frame.children + [site_frame, ]


# def _build_chain_recurse(m, root_frame, root_body):
#     base = root_frame.link.offset
#     cur_frame, cur_base = add_composite_joint(root_frame, root_body.joint, base)
#     jbase = cur_base.inverse() * base
#     if len(root_body.joint) > 0:
#         cur_frame.link.visuals = get_body_geoms(m, root_body.geom, jbase)
#     else:
#         cur_frame.link.visuals = get_body_geoms(m, root_body.geom)
#     for b in root_body.body:
#         cur_frame.children = cur_frame.children + [frame.Frame()]
#         next_frame = cur_frame.children[-1]
#         next_frame.name = b.name + "_frame"
#         next_frame.link = body_to_link(b, jbase)
#         _build_chain_recurse(m, next_frame, b)


def build_chain_from_mjcf(data, body: Union[None, str, int] = None):
    """
    Build a Chain object from MJCF data.

    :param data: MJCF string data
    :param body: the name or index of the body to use as the root of the chain. If None, body idx=0 is used.
    :return: Chain object created from MJCF
    """
    # import xml.etree.ElementTree as ET
    # root = ET.parse(path).getroot()
    #
    # ASSETS = dict()
    # mesh_dir = root.find("compiler").attrib["meshdir"]
    # for asset in root.findall("asset"):
    #     for mesh in asset.findall("mesh"):
    #         filename = mesh.attrib["file"]
    #         with open(os.path.join(os.path.dirname(path), mesh_dir, filename), 'rb') as f:
    #             ASSETS[filename] = f.read()

    m = mujoco.MjModel.from_xml_path(data)
    if body is None:
        root_body = m.body(0)
    else:
        root_body = m.body(body)
    root_frame = frame.Frame(root_body.name,
                             link=body_to_link(root_body),
                             joint=frame.Joint())
    _build_chain_recurse(m, root_frame, root_body)
    return chain.Chain(root_frame)


def build_serial_chain_from_mjcf(data, end_link_name, root_link_name=""):
    """
    Build a SerialChain object from MJCF data.

    :param data: MJCF string data
    :param end_link_name: the name of the link that is the end effector
    :param root_link_name: the name of the root link
    :return: SerialChain object created from MJCF
    """
    mjcf_chain = build_chain_from_mjcf(data)
    serial_chain = chain.SerialChain(mjcf_chain, end_link_name, "" if root_link_name == "" else root_link_name)
    return serial_chain
