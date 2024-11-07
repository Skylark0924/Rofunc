#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

import os

import numpy as np

from rofunc.devices.xsens.src.load_mvnx import load_mvnx
from rofunc.utils.oslab.path import get_rofunc_path
from rofunc.utils.logger.beauty_logger import beauty_print


def get_human_params(mvnx_file, human_mass, human_height):
    human_params = {}

    # -- Link base --
    # - Pelvis -
    # joints
    pelvis_data = mvnx_file.file_data['segments']['elements']['Pelvis']
    human_params['jL5S1'] = pelvis_data['points_mvn']['jL5S1']
    human_params['jLeftHip'] = pelvis_data['points_mvn']['jLeftHip']
    human_params['jRightHip'] = pelvis_data['points_mvn']['jRightHip']
    # box size
    pLeftASI = pelvis_data['points_mvn']['pLeftASI']
    pRightASI = pelvis_data['points_mvn']['pRightASI']
    pSacrum = pelvis_data['points_mvn']['pSacrum']
    pelvis_x = pLeftASI[0] - pSacrum[0]
    pelvis_y = pLeftASI[1] - pRightASI[1]
    pelvis_z = human_params['jL5S1'][2] - human_params['jRightHip'][2]
    human_params['pelvisBox'] = [pelvis_x, pelvis_y, pelvis_z]
    # box origin
    originWrtRightUpperCorner = 0.5 * np.array([-pelvis_x, pelvis_y, -pelvis_z])  # wrt pRightUpperCorner
    human_params['pelvisBoxOrigin'] = np.array(
        [pRightASI[0], pRightASI[1], human_params['jL5S1'][2]]) + originWrtRightUpperCorner  # wrt pHipOrigin
    # mass and inertia
    human_params['pelvisMass'] = 0.08 * human_mass
    human_params['pelvisIxx'] = human_params['pelvisMass'] / 12.0 * (pelvis_y ** 2 + pelvis_z ** 2)
    human_params['pelvisIyy'] = human_params['pelvisMass'] / 12.0 * (pelvis_x ** 2 + pelvis_z ** 2)
    human_params['pelvisIzz'] = human_params['pelvisMass'] / 12.0 * (pelvis_x ** 2 + pelvis_y ** 2)
    # markers
    human_params['pHipOrigin'] = pelvis_data['points_mvn']['pHipOrigin']
    human_params['pRightASI'] = pelvis_data['points_mvn']['pRightASI']
    human_params['pLeftASI'] = pelvis_data['points_mvn']['pLeftASI']
    human_params['pRightCSI'] = pelvis_data['points_mvn']['pRightCSI']
    human_params['pLeftCSI'] = pelvis_data['points_mvn']['pLeftCSI']
    human_params['pRightIschialTub'] = pelvis_data['points_mvn']['pRightIschialTub']
    human_params['pLeftIschialTub'] = pelvis_data['points_mvn']['pLeftIschialTub']
    human_params['pSacrum'] = pelvis_data['points_mvn']['pSacrum']

    # -- Chain links 2~7 --
    # - L5 -
    # joints
    l5_data = mvnx_file.file_data['segments']['elements']['L5']
    human_params['jL4L3'] = l5_data['points_mvn']['jL4L3']
    # box size
    L5_x = pelvis_x
    L5_y = human_params['jLeftHip'][1] - human_params['jRightHip'][1]
    L5_z = human_params['jL4L3'][2]
    human_params['L5Box'] = [L5_x, L5_y, L5_z]
    # box origin
    human_params['L5BoxOrigin'] = 0.5 * np.array([0, 0, L5_z])  # wrt jL5S1
    # mass and inertia
    human_params['L5Mass'] = 0.102 * human_mass
    human_params['L5Ixx'] = human_params['L5Mass'] / 12.0 * (L5_y ** 2 + L5_z ** 2)
    human_params['L5Iyy'] = human_params['L5Mass'] / 12.0 * (L5_x ** 2 + L5_z ** 2)
    human_params['L5Izz'] = human_params['L5Mass'] / 12.0 * (L5_x ** 2 + L5_y ** 2)
    # markers
    human_params['pL5SpinalProcess'] = l5_data['points_mvn']['pL5SpinalProcess']

    # - L3 -
    # joints
    l3_data = mvnx_file.file_data['segments']['elements']['L3']
    human_params['jL1T12'] = l3_data['points_mvn']['jL1T12']
    # box size
    L3_x = pelvis_x
    L3_y = L5_y
    L3_z = human_params['jL1T12'][2]
    human_params['L3Box'] = [L3_x, L3_y, L3_z]
    # box origin
    human_params['L3BoxOrigin'] = 0.5 * np.array([0, 0, L3_z])  # wrt jL4L3
    # mass and inertia
    human_params['L3Mass'] = 0.102 * human_mass
    human_params['L3Ixx'] = human_params['L3Mass'] / 12.0 * (L3_y ** 2 + L3_z ** 2)
    human_params['L3Iyy'] = human_params['L3Mass'] / 12.0 * (L3_x ** 2 + L3_z ** 2)
    human_params['L3Izz'] = human_params['L3Mass'] / 12.0 * (L3_x ** 2 + L3_y ** 2)
    # markers
    human_params['pL3SpinalProcess'] = l3_data['points_mvn']['pL3SpinalProcess']

    # - T12 -
    # joints
    t12_data = mvnx_file.file_data['segments']['elements']['T12']
    human_params['jT9T8'] = t12_data['points_mvn']['jT9T8']
    # box size
    T12_x = pelvis_x
    T12_y = L5_y
    T12_z = human_params['jT9T8'][2]
    human_params['T12Box'] = [T12_x, T12_y, T12_z]
    # box origin
    human_params['T12BoxOrigin'] = 0.5 * np.array([0, 0, T12_z])  # wrt jL1T12
    # mass and inertia
    human_params['T12Mass'] = 0.102 * human_mass
    human_params['T12Ixx'] = human_params['T12Mass'] / 12.0 * (T12_y ** 2 + T12_z ** 2)
    human_params['T12Iyy'] = human_params['T12Mass'] / 12.0 * (T12_x ** 2 + T12_z ** 2)
    human_params['T12Izz'] = human_params['T12Mass'] / 12.0 * (T12_x ** 2 + T12_y ** 2)
    # markers
    human_params['pT12SpinalProcess'] = t12_data['points_mvn']['pT12SpinalProcess']

    # - T8 -
    # joints
    t8_data = mvnx_file.file_data['segments']['elements']['T8']
    human_params['jT1C7'] = t8_data['points_mvn']['jT1C7']
    human_params['jRightT4Shoulder'] = t8_data['points_mvn']['jRightT4Shoulder']
    human_params['jLeftT4Shoulder'] = t8_data['points_mvn']['jLeftT4Shoulder']
    # box size
    neck_data = mvnx_file.file_data['segments']['elements']['Neck']
    pC7SpinalProcess = neck_data['points_mvn']['pC7SpinalProcess']
    pPX = t8_data['points_mvn']['pPX']
    T8_x = pPX[0] - pC7SpinalProcess[0]
    T8_y = human_params['jLeftT4Shoulder'][1] - human_params['jRightT4Shoulder'][1]
    T8_z = human_params['jT1C7'][2]
    human_params['T8Box'] = [T8_x, T8_y, T8_z]
    # box origin
    human_params['T8BoxOrigin'] = 0.5 * np.array([0, 0, T8_z])  # wrt jT9T8
    # mass and inertia
    human_params['T8Mass'] = 0.04 * human_mass
    human_params['T8Ixx'] = human_params['T8Mass'] / 12.0 * (T8_y ** 2 + T8_z ** 2)
    human_params['T8Iyy'] = human_params['T8Mass'] / 12.0 * (T8_x ** 2 + T8_z ** 2)
    human_params['T8Izz'] = human_params['T8Mass'] / 12.0 * (T8_x ** 2 + T8_y ** 2)
    # markers
    human_params['pPX'] = t8_data['points_mvn']['pPX']
    human_params['pIJ'] = t8_data['points_mvn']['pIJ']
    human_params['pC7SpinalProcess'] = neck_data['points_mvn']['pC7SpinalProcess']  # TODO: check
    human_params['pT8SpinalProcess'] = t8_data['points_mvn']['pT8SpinalProcess']
    human_params['pT4SpinalProcess'] = t8_data['points_mvn']['pT4SpinalProcess']

    # - Neck -
    # joints
    neck_data = mvnx_file.file_data['segments']['elements']['Neck']
    human_params['jC1Head'] = neck_data['points_mvn']['jC1Head']
    # box size
    pC7SpinalProcess = neck_data['points_mvn']['pC7SpinalProcess']
    human_params['neck_x'] = np.abs(pC7SpinalProcess[0])
    human_params['neck_z'] = human_params['jC1Head'][2]
    # box origin
    human_params['neckBoxOrigin'] = 0.5 * np.array([0, 0, human_params['neck_z']])  # wrt jT1C7
    # mass and inertia
    human_params['neckMass'] = 0.012 * human_mass
    human_params['neckIxx'] = human_params['neckMass'] / 12.0 * (
            3 * human_params['neck_x'] ** 2 + human_params['neck_z'] ** 2)
    human_params['neckIyy'] = human_params['neckMass'] / 12.0 * (
            3 * human_params['neck_x'] ** 2 + human_params['neck_z'] ** 2)
    human_params['neckIzz'] = human_params['neckMass'] / 2.0 * (human_params['neck_x'] ** 2)

    # - Head -
    # box size
    head_data = mvnx_file.file_data['segments']['elements']['Head']
    pTopHead = head_data['points_mvn']['pTopOfHead']
    human_params['head_z'] = np.abs(pTopHead[2])
    # box origin
    human_params['headBoxOrigin'] = 0.5 * np.array([0, 0, human_params['head_z']])  # wrt C1Head
    # mass and inertia
    human_params['headMass'] = 0.036 * human_mass
    human_params['headIxx'] = 2 * human_params['headMass'] / 5.0 * ((human_params['head_z'] / 2.) ** 2)
    human_params['headIyy'] = 2 * human_params['headMass'] / 5.0 * ((human_params['head_z'] / 2.) ** 2)
    human_params['headIzz'] = 2 * human_params['headMass'] / 5.0 * ((human_params['head_z'] / 2.) ** 2)
    # markers
    human_params['pTopOfHead'] = head_data['points_mvn']['pTopOfHead']
    human_params['pRightAuricularis'] = head_data['points_mvn']['pRightAuricularis']
    human_params['pLeftAuricularis'] = head_data['points_mvn']['pLeftAuricularis']
    human_params['pBackOfHead'] = head_data['points_mvn']['pBackOfHead']

    # -- Chain links 8~11 --
    # - Right shoulder -
    # joints
    right_shoulder_data = mvnx_file.file_data['segments']['elements']['RightShoulder']
    human_params['jRightShoulder'] = right_shoulder_data['points_mvn']['jRightShoulder']
    # box size
    pRightAcromion = right_shoulder_data['points_mvn']['pRightAcromion']
    human_params['rightSho_y'] = np.abs(human_params['jRightShoulder'][1])
    human_params['rightSho_z'] = np.abs(pRightAcromion[2])  # TODO: assumption
    # box origin
    human_params['rightShoulderBoxOrigin'] = 0.5 * np.array([0, -human_params['rightSho_y'], 0])  # wrt jRightT4Shoulder
    # mass and inertia
    human_params['rightShoulderMass'] = 0.031 * human_mass
    human_params['rightShoulderIxx'] = human_params['rightShoulderMass'] / 12. * (
            human_params['rightSho_y'] ** 2 + 3 * (human_params['rightSho_z'] / 2) ** 2)
    human_params['rightShoulderIyy'] = human_params['rightShoulderMass'] / 2. * ((human_params['rightSho_z'] / 2) ** 2)
    human_params['rightShoulderIzz'] = human_params['rightShoulderMass'] / 12. * (
            human_params['rightSho_y'] ** 2 + 3 * (human_params['rightSho_z'] / 2) ** 2)
    # markers
    human_params['pRightAcromion'] = right_shoulder_data['points_mvn']['pRightAcromion']

    # - Right upper arm -
    # joints
    right_upper_arm_data = mvnx_file.file_data['segments']['elements']['RightUpperArm']
    human_params['jRightElbow'] = right_upper_arm_data['points_mvn']['jRightElbow']
    # box size
    pRightArmLatEp = right_upper_arm_data['points_mvn']['pRightArmLatEpicondyle']
    pRightArmMedEp = right_upper_arm_data['points_mvn']['pRightArmMedEpicondyle']
    human_params['rightUpperArm_y'] = np.abs(human_params['jRightElbow'][1])
    human_params['rightUpperArm_z'] = pRightArmLatEp[2] - pRightArmMedEp[2]
    # box origin
    human_params['rightUpperArmBoxOrigin'] = 0.5 * np.array(
        [0, -human_params['rightUpperArm_y'], 0])  # wrt jRightShoulder
    # mass and inertia
    human_params['rightUpperArmMass'] = 0.030 * human_mass
    human_params['rightUpperArmIxx'] = human_params['rightUpperArmMass'] / 12. * (
            human_params['rightUpperArm_y'] ** 2 + 3 * (human_params['rightUpperArm_z'] / 2) ** 2)
    human_params['rightUpperArmIyy'] = human_params['rightUpperArmMass'] / 2. * (
            (human_params['rightUpperArm_z'] / 2) ** 2)
    human_params['rightUpperArmIzz'] = human_params['rightUpperArmMass'] / 12. * (
            human_params['rightUpperArm_y'] ** 2 + 3 * (human_params['rightUpperArm_z'] / 2) ** 2)
    # markers
    human_params['pRightArmLatEpicondyle'] = right_upper_arm_data['points_mvn']['pRightArmLatEpicondyle']
    human_params['pRightArmMedEpicondyle'] = right_upper_arm_data['points_mvn']['pRightArmMedEpicondyle']

    # - Right forearm -
    # joints
    right_forearm_data = mvnx_file.file_data['segments']['elements']['RightForeArm']
    human_params['jRightWrist'] = right_forearm_data['points_mvn']['jRightWrist']
    human_params['jRightWrist'][1] = human_params['jRightWrist'][1]
    # box size
    human_params['rightForeArm_y'] = np.abs(human_params['jRightWrist'][1])
    human_params['rightForeArm_z'] = 2 / 3 * human_params['rightUpperArm_z']  # assumption
    # box origin
    human_params['rightForeArmBoxOrigin'] = 0.5 * np.array([0, -human_params['rightForeArm_y'], 0])  # wrt jRightElbow
    # mass and inertia
    human_params['rightForeArmMass'] = 0.020 * human_mass
    human_params['rightForeArmIxx'] = human_params['rightForeArmMass'] / 12. * (
            human_params['rightForeArm_y'] ** 2 + 3 * (human_params['rightForeArm_z'] / 2) ** 2)
    human_params['rightForeArmIyy'] = human_params['rightForeArmMass'] / 2. * (
            (human_params['rightForeArm_z'] / 2) ** 2)
    human_params['rightForeArmIzz'] = human_params['rightForeArmMass'] / 12. * (
            human_params['rightForeArm_y'] ** 2 + 3 * (human_params['rightForeArm_z'] / 2) ** 2)
    # markers
    human_params['pRightUlnarStyloid'] = right_forearm_data['points_mvn']['pRightUlnarStyloid']
    human_params['pRightRadialStyloid'] = right_forearm_data['points_mvn']['pRightRadialStyloid']
    human_params['pRightOlecranon'] = right_forearm_data['points_mvn']['pRightOlecranon']

    # - Right hand -
    # box size
    right_hand_data = mvnx_file.file_data['segments']['elements']['RightHand']
    pTopHand = right_hand_data['points_mvn']['pRightTopOfHand']
    rightHand_y = np.abs(pTopHand[1])
    rightHand_x = 2 / 3 * rightHand_y  # assumption
    rightHand_z = human_params['rightForeArm_z']  # assumption
    human_params['rightHandBox'] = np.array([rightHand_x, rightHand_y, rightHand_z])
    # box origin
    human_params['rightHandBoxOrigin'] = 0.5 * np.array([0, -rightHand_y, 0])  # wrt jRightWrist
    # mass and inertia
    human_params['rightHandMass'] = 0.006 * human_mass
    human_params['rightHandIxx'] = human_params['rightHandMass'] / 12. * (rightHand_y ** 2 + rightHand_z ** 2)
    human_params['rightHandIyy'] = human_params['rightHandMass'] / 12. * (rightHand_x ** 2 + rightHand_z ** 2)
    human_params['rightHandIzz'] = human_params['rightHandMass'] / 12. * (rightHand_x ** 2 + rightHand_y ** 2)
    # markers
    human_params['pRightTopOfHand'] = right_hand_data['points_mvn']['pRightTopOfHand']
    human_params['pRightPinky'] = right_hand_data['points_mvn']['pRightPinky']
    human_params['pRightBallHand'] = right_hand_data['points_mvn']['pRightBallHand']

    # -- Chain links 12~15 --
    # - Left shoulder -
    # joints
    left_shoulder_data = mvnx_file.file_data['segments']['elements']['LeftShoulder']
    human_params['jLeftShoulder'] = left_shoulder_data['points_mvn']['jLeftShoulder']
    # box size
    pLeftAcromion = left_shoulder_data['points_mvn']['pLeftAcromion']
    human_params['leftSho_y'] = np.abs(human_params['jLeftShoulder'][1])
    human_params['leftSho_z'] = np.abs(pLeftAcromion[2])  # assumption
    # box origin
    human_params['leftShoulderBoxOrigin'] = 0.5 * np.array([0, human_params['leftSho_y'], 0])  # wrt jLeftT4Shoulder
    # mass and inertia
    human_params['leftShoulderMass'] = 0.031 * human_mass
    human_params['leftShoulderIxx'] = human_params['leftShoulderMass'] / 12. * (
            human_params['leftSho_y'] ** 2 + 3 * (human_params['leftSho_z'] / 2) ** 2)
    human_params['leftShoulderIyy'] = human_params['leftShoulderMass'] / 2. * ((human_params['leftSho_z'] / 2) ** 2)
    human_params['leftShoulderIzz'] = human_params['leftShoulderMass'] / 12. * (
            human_params['leftSho_y'] ** 2 + 3 * (human_params['leftSho_z'] / 2) ** 2)
    # markers
    human_params['pLeftAcromion'] = left_shoulder_data['points_mvn']['pLeftAcromion']

    # - Left upper arm -
    # joints
    left_upper_arm_data = mvnx_file.file_data['segments']['elements']['LeftUpperArm']
    human_params['jLeftElbow'] = left_upper_arm_data['points_mvn']['jLeftElbow']
    # box size
    pLeftArmLatEp = left_upper_arm_data['points_mvn']['pLeftArmLatEpicondyle']
    pLeftArmMedEp = left_upper_arm_data['points_mvn']['pLeftArmMedEpicondyle']
    human_params['leftUpperArm_y'] = np.abs(human_params['jLeftElbow'][1])
    human_params['leftUpperArm_z'] = pLeftArmLatEp[2] - pLeftArmMedEp[2]
    # box origin
    human_params['leftUpperArmBoxOrigin'] = 0.5 * np.array([0, human_params['leftUpperArm_y'], 0])  # wrt jLeftShoulder
    # mass and inertia
    human_params['leftUpperArmMass'] = 0.030 * human_mass
    human_params['leftUpperArmIxx'] = human_params['leftUpperArmMass'] / 12. * (
            human_params['leftUpperArm_y'] ** 2 + 3 * (human_params['leftUpperArm_z'] / 2) ** 2)
    human_params['leftUpperArmIyy'] = human_params['leftUpperArmMass'] / 2. * (
            (human_params['leftUpperArm_z'] / 2) ** 2)
    human_params['leftUpperArmIzz'] = human_params['leftUpperArmMass'] / 12. * (
            human_params['leftUpperArm_y'] ** 2 + 3 * (human_params['leftUpperArm_z'] / 2) ** 2)
    # markers
    human_params['pLeftArmLatEpicondyle'] = left_upper_arm_data['points_mvn']['pLeftArmLatEpicondyle']
    human_params['pLeftArmMedEpicondyle'] = left_upper_arm_data['points_mvn']['pLeftArmMedEpicondyle']

    # - Left foreArm -
    # joints
    left_forearm_data = mvnx_file.file_data['segments']['elements']['LeftForeArm']
    human_params['jLeftWrist'] = left_forearm_data['points_mvn']['jLeftWrist']
    # box size
    human_params['leftForeArm_y'] = np.abs(human_params['jLeftWrist'][1])
    human_params['leftForeArm_z'] = 2 / 3 * human_params['leftUpperArm_z']  # assumption
    # box origin
    human_params['leftForeArmBoxOrigin'] = 0.5 * np.array([0, human_params['leftForeArm_y'], 0])  # wrt jLeftElbow
    # mass and inertia
    human_params['leftForeArmMass'] = 0.020 * human_mass
    human_params['leftForeArmIxx'] = human_params['leftForeArmMass'] / 12. * (
            human_params['leftForeArm_y'] ** 2 + 3 * (human_params['leftForeArm_z'] / 2) ** 2)
    human_params['leftForeArmIyy'] = human_params['leftForeArmMass'] / 2. * ((human_params['leftForeArm_z'] / 2) ** 2)
    human_params['leftForeArmIzz'] = human_params['leftForeArmMass'] / 12. * (
            human_params['leftForeArm_y'] ** 2 + 3 * (human_params['leftForeArm_z'] / 2) ** 2)
    # markers
    human_params['pLeftUlnarStyloid'] = left_forearm_data['points_mvn']['pLeftUlnarStyloid']
    human_params['pLeftRadialStyloid'] = left_forearm_data['points_mvn']['pLeftRadialStyloid']
    human_params['pLeftOlecranon'] = left_forearm_data['points_mvn']['pLeftOlecranon']

    # - Left hand -
    # box size
    left_hand_data = mvnx_file.file_data['segments']['elements']['LeftHand']
    pTopHand = left_hand_data['points_mvn']['pLeftTopOfHand']
    leftHand_y = np.abs(pTopHand[1])
    leftHand_x = 2 / 3 * leftHand_y  # assumption
    leftHand_z = human_params['leftForeArm_z']  # assumption
    human_params['leftHandBox'] = np.array([leftHand_x, leftHand_y, leftHand_z])
    # box origin
    human_params['leftHandBoxOrigin'] = 0.5 * np.array([0, leftHand_y, 0])  # wrt jLeftWrist
    # mass and inertia
    human_params['leftHandMass'] = 0.006 * human_mass
    human_params['leftHandIxx'] = human_params['leftHandMass'] / 12. * (leftHand_y ** 2 + leftHand_z ** 2)
    human_params['leftHandIyy'] = human_params['leftHandMass'] / 12. * (leftHand_x ** 2 + leftHand_z ** 2)
    human_params['leftHandIzz'] = human_params['leftHandMass'] / 12. * (leftHand_x ** 2 + leftHand_y ** 2)
    # markers
    human_params['pLeftTopOfHand'] = left_hand_data['points_mvn']['pLeftTopOfHand']
    human_params['pLeftPinky'] = left_hand_data['points_mvn']['pLeftPinky']
    human_params['pLeftBallHand'] = left_hand_data['points_mvn']['pLeftBallHand']

    # -- Chain links 16~19 --
    # - Right Upper Leg -
    # joints
    right_upper_leg_data = mvnx_file.file_data['segments']['elements']['RightUpperLeg']
    human_params['jRightKnee'] = right_upper_leg_data['points_mvn']['jRightKnee']
    # box size
    pRightTro = right_upper_leg_data['points_mvn']['pRightGreaterTrochanter']
    pRightKneeUL = right_upper_leg_data['points_mvn']['pRightKneeMedEpicondyle']
    human_params['rightUpperLeg_x'] = pRightKneeUL[1] - pRightTro[1]
    human_params['rightUpperLeg_z'] = np.abs(human_params['jRightKnee'][2])
    # box origin
    human_params['rightUpperLegBoxOrigin'] = 0.5 * np.array([0, 0, -human_params['rightUpperLeg_z']])  # wrt jRightHip
    # mass and inertia
    human_params['rightUpperLegMass'] = 0.125 * human_mass
    human_params['rightUpperLegIxx'] = human_params['rightUpperLegMass'] / 12. * (
            3 * (human_params['rightUpperLeg_x'] / 2) ** 2 + human_params['rightUpperLeg_z'] ** 2)
    human_params['rightUpperLegIyy'] = human_params['rightUpperLegMass'] / 12. * (
            3 * (human_params['rightUpperLeg_x'] / 2) ** 2 + human_params['rightUpperLeg_z'] ** 2)
    human_params['rightUpperLegIzz'] = human_params['rightUpperLegMass'] / 2. * (
            (human_params['rightUpperLeg_x'] / 2) ** 2)
    # markers
    human_params['pRightGreaterTrochanter'] = right_upper_leg_data['points_mvn']['pRightGreaterTrochanter']
    human_params['pRightPatella'] = right_upper_leg_data['points_mvn']['pRightPatella']

    # - Right Lower Leg -
    # joints
    right_lower_leg_data = mvnx_file.file_data['segments']['elements']['RightLowerLeg']
    human_params['jRightAnkle'] = right_lower_leg_data['points_mvn']['jRightAnkle']
    # box size
    pRightKneeLatLL = right_upper_leg_data['points_mvn']['pRightKneeLatEpicondyle']  # TODO: check if correct
    pRightKneeMedLL = right_upper_leg_data['points_mvn']['pRightKneeMedEpicondyle']
    human_params['rightLowerLeg_x'] = pRightKneeMedLL[1] - pRightKneeLatLL[1]
    human_params['rightLowerLeg_z'] = np.abs(human_params['jRightAnkle'][2])
    # box origin
    human_params['rightLowerLegBoxOrigin'] = 0.5 * np.array([0, 0, -human_params['rightLowerLeg_z']])  # wrt jRightKnee
    # mass and inertia
    human_params['rightLowerLegMass'] = 0.0365 * human_mass
    human_params['rightLowerLegIxx'] = human_params['rightLowerLegMass'] / 12. * (
            3 * (human_params['rightLowerLeg_x'] / 2) ** 2 + human_params['rightLowerLeg_z'] ** 2)
    human_params['rightLowerLegIyy'] = human_params['rightLowerLegMass'] / 12. * (
            3 * (human_params['rightLowerLeg_x'] / 2) ** 2 + human_params['rightLowerLeg_z'] ** 2)
    human_params['rightLowerLegIzz'] = human_params['rightLowerLegMass'] / 2. * (
            (human_params['rightLowerLeg_x'] / 2) ** 2)
    # markers
    human_params['pRightKneeLatEpicondyle'] = right_upper_leg_data['points_mvn']['pRightKneeLatEpicondyle']
    human_params['pRightKneeMedEpicondyle'] = right_upper_leg_data['points_mvn']['pRightKneeMedEpicondyle']
    human_params['pRightLatMalleolus'] = right_lower_leg_data['points_mvn']['pRightLatMalleolus']
    human_params['pRightMedMalleolus'] = right_lower_leg_data['points_mvn']['pRightMedMalleolus']
    human_params['pRightTibialTub'] = right_lower_leg_data['points_mvn']['pRightTibialTub']

    # - Right Foot -
    # joints
    right_foot_data = mvnx_file.file_data['segments']['elements']['RightFoot']
    human_params['jRightBallFoot'] = right_foot_data['points_mvn']['jRightBallFoot']
    # box size
    pRightHeel = right_foot_data['points_mvn']['pRightHeelFoot']
    rightFoot_x = human_params['jRightBallFoot'][0] - pRightHeel[0]
    rightFoot_y = human_params['rightLowerLeg_x']
    rightFoot_z = np.abs(pRightHeel[2])
    human_params['rightFootBox'] = np.array([rightFoot_x, rightFoot_y, rightFoot_z])
    # box origin
    originWrtRightHeel = 0.5 * np.array([rightFoot_x, 0, rightFoot_z])  # wrt pRightHeelFoot
    human_params['rightFootBoxOrigin'] = pRightHeel + originWrtRightHeel  # wrt jRightAnkle TODO: check
    # mass and inertia
    human_params['rightFootMass'] = 0.0130 * human_mass
    human_params['rightFootIxx'] = human_params['rightFootMass'] / 12. * (rightFoot_y ** 2 + rightFoot_z ** 2)
    human_params['rightFootIyy'] = human_params['rightFootMass'] / 12. * (rightFoot_x ** 2 + rightFoot_z ** 2)
    human_params['rightFootIzz'] = human_params['rightFootMass'] / 12. * (rightFoot_x ** 2 + rightFoot_y ** 2)
    # markers
    human_params['pRightHeelFoot'] = right_foot_data['points_mvn']['pRightHeelFoot']
    human_params['pRightFirstMetatarsal'] = right_foot_data['points_mvn']['pRightFirstMetatarsal']
    human_params['pRightFifthMetatarsal'] = right_foot_data['points_mvn']['pRightFifthMetatarsal']
    human_params['pRightPivotFoot'] = right_foot_data['points_mvn']['pRightPivotFoot']
    human_params['pRightHeelCenter'] = right_foot_data['points_mvn']['pRightHeelCenter']

    # - Right Toe -
    # box size
    right_toe_data = mvnx_file.file_data['segments']['elements']['RightToe']
    pRightToe = right_toe_data['points_mvn']['pRightToe']
    rightToe_x = np.abs(pRightToe[0])
    rightToe_y = rightFoot_y
    rightToe_z = np.abs(pRightToe[2])
    human_params['rightToeBox'] = np.array([rightToe_x, rightToe_y, rightToe_z])
    # box origin
    human_params['rightToeBoxOrigin'] = 0.5 * np.array([rightToe_x, 0, 0])  # wrt jRightKnee
    # mass and inertia
    human_params['rightToeMass'] = 0.0015 * human_mass
    human_params['rightToeIxx'] = human_params['rightToeMass'] / 12. * (rightToe_y ** 2 + rightToe_z ** 2)
    human_params['rightToeIyy'] = human_params['rightToeMass'] / 12. * (rightToe_x ** 2 + rightToe_z ** 2)
    human_params['rightToeIzz'] = human_params['rightToeMass'] / 12. * (rightToe_x ** 2 + rightToe_y ** 2)
    # markers
    human_params['pRightToe'] = right_toe_data['points_mvn']['pRightToe']

    # -- Chain links 20~23 --
    # - Left Upper Leg -
    # joints
    left_upper_leg_data = mvnx_file.file_data['segments']['elements']['LeftUpperLeg']
    human_params['jLeftKnee'] = left_upper_leg_data['points_mvn']['jLeftKnee']
    # box size
    pLeftTro = left_upper_leg_data['points_mvn']['pLeftGreaterTrochanter']
    pLeftKneeUL = left_upper_leg_data['points_mvn']['pLeftKneeMedEpicondyle']
    human_params['leftUpperLeg_x'] = np.abs(pLeftKneeUL[1] - pLeftTro[1])
    human_params['leftUpperLeg_z'] = np.abs(human_params['jLeftKnee'][2])
    # box origin
    human_params['leftUpperLegBoxOrigin'] = 0.5 * np.array([0, 0, -human_params['leftUpperLeg_z']])  # wrt jLeftHip
    # mass and inertia
    human_params['leftUpperLegMass'] = 0.125 * human_mass
    human_params['leftUpperLegIxx'] = human_params['leftUpperLegMass'] / 12. * (
            3 * (human_params['leftUpperLeg_x'] / 2) ** 2 + human_params['leftUpperLeg_z'] ** 2)
    human_params['leftUpperLegIyy'] = human_params['leftUpperLegMass'] / 12. * (
            3 * (human_params['leftUpperLeg_x'] / 2) ** 2 + human_params['leftUpperLeg_z'] ** 2)
    human_params['leftUpperLegIzz'] = human_params['leftUpperLegMass'] / 2. * (
            (human_params['leftUpperLeg_x'] / 2) ** 2)
    # markers
    human_params['pLeftGreaterTrochanter'] = left_upper_leg_data['points_mvn']['pLeftGreaterTrochanter']
    human_params['pLeftPatella'] = left_upper_leg_data['points_mvn']['pLeftPatella']

    # - Left Lower Leg -
    # joints
    left_lower_leg_data = mvnx_file.file_data['segments']['elements']['LeftLowerLeg']
    human_params['jLeftAnkle'] = left_lower_leg_data['points_mvn']['jLeftAnkle']
    # box size
    pLeftKneeLatLL = left_upper_leg_data['points_mvn']['pLeftKneeLatEpicondyle']  # TODO: check if this is correct
    pLeftKneeMedLL = left_upper_leg_data['points_mvn']['pLeftKneeMedEpicondyle']
    human_params['leftLowerLeg_x'] = np.abs(pLeftKneeMedLL[1] - pLeftKneeLatLL[1])
    human_params['leftLowerLeg_z'] = np.abs(human_params['jLeftAnkle'][2])
    # box origin
    human_params['leftLowerLegBoxOrigin'] = 0.5 * np.array([0, 0, -human_params['leftLowerLeg_z']])  # wrt jLeftKnee
    # mass and inertia
    human_params['leftLowerLegMass'] = 0.0365 * human_mass
    human_params['leftLowerLegIxx'] = human_params['leftLowerLegMass'] / 12. * (
            3 * (human_params['leftLowerLeg_x'] / 2) ** 2 + human_params['leftLowerLeg_z'] ** 2)
    human_params['leftLowerLegIyy'] = human_params['leftLowerLegMass'] / 12. * (
            3 * (human_params['leftLowerLeg_x'] / 2) ** 2 + human_params['leftLowerLeg_z'] ** 2)
    human_params['leftLowerLegIzz'] = human_params['leftLowerLegMass'] / 2. * (
            (human_params['leftLowerLeg_x'] / 2) ** 2)
    # markers
    human_params['pLeftKneeLatEpicondyle'] = left_upper_leg_data['points_mvn']['pLeftKneeLatEpicondyle']  # TODO
    human_params['pLeftKneeMedEpicondyle'] = left_upper_leg_data['points_mvn']['pLeftKneeMedEpicondyle']
    human_params['pLeftLatMalleolus'] = left_lower_leg_data['points_mvn']['pLeftLatMalleolus']
    human_params['pLeftMedMalleolus'] = left_lower_leg_data['points_mvn']['pLeftMedMalleolus']
    human_params['pLeftTibialTub'] = left_lower_leg_data['points_mvn']['pLeftTibialTub']

    # - Left Foot -
    # joints
    left_foot_data = mvnx_file.file_data['segments']['elements']['LeftFoot']
    human_params['jLeftBallFoot'] = left_foot_data['points_mvn']['jLeftBallFoot']
    # box size
    pLeftHeel = left_foot_data['points_mvn']['pLeftHeelFoot']
    leftFoot_x = human_params['jLeftBallFoot'][0] - pLeftHeel[0]
    leftFoot_y = human_params['leftLowerLeg_x']
    leftFoot_z = np.abs(pLeftHeel[2])
    human_params['leftFootBox'] = np.array([leftFoot_x, leftFoot_y, leftFoot_z])
    # box origin
    originWrtLeftHeel = 0.5 * np.array([leftFoot_x, 0, leftFoot_z])  # wrt pLeftHeelFoot
    human_params['leftFootBoxOrigin'] = pLeftHeel + originWrtLeftHeel  # wrt jLeftAnkle TODO: check
    # mass and inertia
    human_params['leftFootMass'] = 0.0130 * human_mass
    human_params['leftFootIxx'] = human_params['leftFootMass'] / 12. * (leftFoot_y ** 2 + leftFoot_z ** 2)
    human_params['leftFootIyy'] = human_params['leftFootMass'] / 12. * (leftFoot_x ** 2 + leftFoot_z ** 2)
    human_params['leftFootIzz'] = human_params['leftFootMass'] / 12. * (leftFoot_x ** 2 + leftFoot_y ** 2)
    # markers
    human_params['pLeftHeelFoot'] = left_foot_data['points_mvn']['pLeftHeelFoot']
    human_params['pLeftFirstMetatarsal'] = left_foot_data['points_mvn']['pLeftFirstMetatarsal']
    human_params['pLeftFifthMetatarsal'] = left_foot_data['points_mvn']['pLeftFifthMetatarsal']
    human_params['pLeftPivotFoot'] = left_foot_data['points_mvn']['pLeftPivotFoot']
    human_params['pLeftHeelCenter'] = left_foot_data['points_mvn']['pLeftHeelCenter']

    # - Left Toe -
    # box size
    left_toe_data = mvnx_file.file_data['segments']['elements']['LeftToe']
    pLeftToe = left_toe_data['points_mvn']['pLeftToe']
    leftToe_x = np.abs(pLeftToe[0])
    leftToe_y = rightFoot_y
    leftToe_z = np.abs(pLeftToe[2])
    human_params['leftToeBox'] = np.array([leftToe_x, leftToe_y, leftToe_z])
    # box origin
    human_params['leftToeBoxOrigin'] = 0.5 * np.array([leftToe_x, 0, 0])  # wrt jLeftKnee
    # mass and inertia
    human_params['leftToeMass'] = 0.0015 * human_mass
    human_params['leftToeIxx'] = human_params['leftToeMass'] / 12. * (leftToe_y ** 2 + leftToe_z ** 2)
    human_params['leftToeIyy'] = human_params['leftToeMass'] / 12. * (leftToe_x ** 2 + leftToe_z ** 2)
    human_params['leftToeIzz'] = human_params['leftToeMass'] / 12. * (leftToe_x ** 2 + leftToe_y ** 2)
    # markers
    human_params['pLeftToe'] = left_toe_data['points_mvn']['pLeftToe']

    return human_params


def xsens2urdf(mvnx_path, save_dir=None, human_mass=70, human_height=183):
    if mvnx_path.endswith('.mvnx'):
        mvnx_file = load_mvnx(mvnx_path)
    else:
        raise Exception('Wrong file type, only support .mvnx')

    actor_name = mvnx_file.file_data['meta_data']['name']
    human_params = get_human_params(mvnx_file, human_mass, human_height)

    # Query Table
    query_table = {'PELVIS_BOX_ORIGIN': human_params['pelvisBoxOrigin'],
                   'PELVIS_COM_ORIGIN': human_params['pelvisBoxOrigin'],
                   'PELVIS_BOX_SIZE': human_params['pelvisBox'],
                   'PELVISMASS': human_params['pelvisMass'],
                   'PELVISINERTIAIXX': human_params['pelvisIxx'],
                   'PELVISINERTIAIYY': human_params['pelvisIyy'],
                   'PELVISINERTIAIZZ': human_params['pelvisIzz'],
                   'jL5S1_ORIGIN': human_params['jL5S1'],
                   'jLeftHip_ORIGIN': human_params['jLeftHip'],
                   'jRightHip_ORIGIN': human_params['jRightHip'],
                   'L5_BOX_ORIGIN': human_params['L5BoxOrigin'],
                   'L5_COM_ORIGIN': human_params['L5BoxOrigin'],
                   'L5_BOX_SIZE': human_params['L5Box'],
                   'L5MASS': human_params['L5Mass'],
                   'L5INERTIAIXX': human_params['L5Ixx'],
                   'L5INERTIAIYY': human_params['L5Iyy'],
                   'L5INERTIAIZZ': human_params['L5Izz'],
                   'jL4L3_ORIGIN': human_params['jL4L3'],
                   'L3_BOX_ORIGIN': human_params['L3BoxOrigin'],
                   'L3_COM_ORIGIN': human_params['L3BoxOrigin'],
                   'L3_BOX_SIZE': human_params['L3Box'],
                   'L3MASS': human_params['L3Mass'],
                   'L3INERTIAIXX': human_params['L3Ixx'],
                   'L3INERTIAIYY': human_params['L3Iyy'],
                   'L3INERTIAIZZ': human_params['L3Izz'],
                   'jL1T12_ORIGIN': human_params['jL1T12'],
                   'T12_BOX_ORIGIN': human_params['T12BoxOrigin'],
                   'T12_COM_ORIGIN': human_params['T12BoxOrigin'],
                   'T12_BOX_SIZE': human_params['T12Box'],
                   'T12MASS': human_params['T12Mass'],
                   'T12INERTIAIXX': human_params['T12Ixx'],
                   'T12INERTIAIYY': human_params['T12Iyy'],
                   'T12INERTIAIZZ': human_params['T12Izz'],
                   'jT9T8_ORIGIN': human_params['jT9T8'],
                   'T8_BOX_ORIGIN': human_params['T8BoxOrigin'],
                   'T8_COM_ORIGIN': human_params['T8BoxOrigin'],
                   'T8_BOX_SIZE': human_params['T8Box'],
                   'T8MASS': human_params['T8Mass'],
                   'T8INERTIAIXX': human_params['T8Ixx'],
                   'T8INERTIAIYY': human_params['T8Iyy'],
                   'T8INERTIAIZZ': human_params['T8Izz'],
                   'jT1C7_ORIGIN': human_params['jT1C7'],
                   'jRightC7Shoulder_ORIGIN': human_params['jRightT4Shoulder'],
                   'jLeftC7Shoulder_ORIGIN': human_params['jLeftT4Shoulder'],
                   'NECK_BOX_ORIGIN': human_params['neckBoxOrigin'],
                   'NECK_COM_ORIGIN': human_params['neckBoxOrigin'],
                   'NECKHEIGHT': human_params['neck_z'],
                   'NECKRADIUS': 0.5 * human_params['neck_x'],
                   'NECKMASS': human_params['neckMass'],
                   'NECKINERTIAIXX': human_params['neckIxx'],
                   'NECKINERTIAIYY': human_params['neckIyy'],
                   'NECKINERTIAIZZ': human_params['neckIzz'],
                   'jC1Head_ORIGIN': human_params['jC1Head'],
                   'HEAD_BOX_ORIGIN': human_params['headBoxOrigin'],
                   'HEAD_COM_ORIGIN': human_params['headBoxOrigin'],
                   'HEADRADIUS': 0.5 * human_params['head_z'],
                   'HEADMASS': human_params['headMass'],
                   'HEADINERTIAIXX': human_params['headIxx'],
                   'HEADINERTIAIYY': human_params['headIyy'],
                   'HEADINERTIAIZZ': human_params['headIzz'],

                   'RIGHTSHOULDER_BOX_ORIGIN': human_params['rightShoulderBoxOrigin'],
                   'RIGHTSHOULDER_COM_ORIGIN': human_params['rightShoulderBoxOrigin'],
                   'RIGHTSHOULDERHEIGHT': human_params['rightSho_y'],
                   'RIGHTSHOULDERRADIUS': 0.5 * human_params['rightSho_z'],
                   'RIGHTSHOULDERMASS': human_params['rightShoulderMass'],
                   'RIGHTSHOULDERINERTIAIXX': human_params['rightShoulderIxx'],
                   'RIGHTSHOULDERINERTIAIYY': human_params['rightShoulderIyy'],
                   'RIGHTSHOULDERINERTIAIZZ': human_params['rightShoulderIzz'],
                   'jRightShoulder_ORIGIN': human_params['jRightShoulder'],

                   'RIGHTUPPERARM_BOX_ORIGIN': human_params['rightUpperArmBoxOrigin'],
                   'RIGHTUPPERARM_COM_ORIGIN': human_params['rightUpperArmBoxOrigin'],
                   'RIGHTUPPERARMHEIGHT': human_params['rightUpperArm_y'],
                   'RIGHTUPPERARMRADIUS': 0.5 * human_params['rightUpperArm_z'],
                   'RIGHTUPPERARMMASS': human_params['rightUpperArmMass'],
                   'RIGHTUPPERARMINERTIAIXX': human_params['rightUpperArmIxx'],
                   'RIGHTUPPERARMINERTIAIYY': human_params['rightUpperArmIyy'],
                   'RIGHTUPPERARMINERTIAIZZ': human_params['rightUpperArmIzz'],
                   'jRightElbow_ORIGIN': human_params['jRightElbow'],

                   'RIGHTFOREARM_BOX_ORIGIN': human_params['rightForeArmBoxOrigin'],
                   'RIGHTFOREARM_COM_ORIGIN': human_params['rightForeArmBoxOrigin'],
                   'RIGHTFOREARMHEIGHT': human_params['rightForeArm_y'],
                   'RIGHTFOREARMRADIUS': 0.5 * human_params['rightForeArm_z'],
                   'RIGHTFOREARMMASS': human_params['rightForeArmMass'],
                   'RIGHTFOREARMINERTIAIXX': human_params['rightForeArmIxx'],
                   'RIGHTFOREARMINERTIAIYY': human_params['rightForeArmIyy'],
                   'RIGHTFOREARMINERTIAIZZ': human_params['rightForeArmIzz'],
                   'jRightWrist_ORIGIN': human_params['jRightWrist'],

                   'RIGHTHAND_BOX_ORIGIN': human_params['rightHandBoxOrigin'],
                   'RIGHTHAND_COM_ORIGIN': human_params['rightHandBoxOrigin'],
                   'RIGHTHAND_BOX_SIZE': human_params['rightHandBox'],
                   'RIGHTHANDMASS': human_params['rightHandMass'],
                   'RIGHTHANDINERTIAIXX': human_params['rightHandIxx'],
                   'RIGHTHANDINERTIAIYY': human_params['rightHandIyy'],
                   'RIGHTHANDINERTIAIZZ': human_params['rightHandIzz'],

                   'LEFTSHOULDER_BOX_ORIGIN': human_params['leftShoulderBoxOrigin'],
                   'LEFTSHOULDER_COM_ORIGIN': human_params['leftShoulderBoxOrigin'],
                   'LEFTSHOULDERHEIGHT': human_params['leftSho_y'],
                   'LEFTSHOULDERRADIUS': 0.5 * human_params['leftSho_z'],
                   'LEFTSHOULDERMASS': human_params['leftShoulderMass'],
                   'LEFTSHOULDERINERTIAIXX': human_params['leftShoulderIxx'],
                   'LEFTSHOULDERINERTIAIYY': human_params['leftShoulderIyy'],
                   'LEFTSHOULDERINERTIAIZZ': human_params['leftShoulderIzz'],
                   'jLeftShoulder_ORIGIN': human_params['jLeftShoulder'],

                   'LEFTUPPERARM_BOX_ORIGIN': human_params['leftUpperArmBoxOrigin'],
                   'LEFTUPPERARM_COM_ORIGIN': human_params['leftUpperArmBoxOrigin'],
                   'LEFTUPPERARMHEIGHT': human_params['leftUpperArm_y'],
                   'LEFTUPPERARMRADIUS': 0.5 * human_params['leftUpperArm_z'],
                   'LEFTUPPERARMMASS': human_params['leftUpperArmMass'],
                   'LEFTUPPERARMINERTIAIXX': human_params['leftUpperArmIxx'],
                   'LEFTUPPERARMINERTIAIYY': human_params['leftUpperArmIyy'],
                   'LEFTUPPERARMINERTIAIZZ': human_params['leftUpperArmIzz'],
                   'jLeftElbow_ORIGIN': human_params['jLeftElbow'],

                   'LEFTFOREARM_BOX_ORIGIN': human_params['leftForeArmBoxOrigin'],
                   'LEFTFOREARM_COM_ORIGIN': human_params['leftForeArmBoxOrigin'],
                   'LEFTFOREARMHEIGHT': human_params['leftForeArm_y'],
                   'LEFTFOREARMRADIUS': 0.5 * human_params['leftForeArm_z'],
                   'LEFTFOREARMMASS': human_params['leftForeArmMass'],
                   'LEFTFOREARMINERTIAIXX': human_params['leftForeArmIxx'],
                   'LEFTFOREARMINERTIAIYY': human_params['leftForeArmIyy'],
                   'LEFTFOREARMINERTIAIZZ': human_params['leftForeArmIzz'],
                   'jLeftWrist_ORIGIN': human_params['jLeftWrist'],

                   'LEFTHAND_BOX_ORIGIN': human_params['leftHandBoxOrigin'],
                   'LEFTHAND_COM_ORIGIN': human_params['leftHandBoxOrigin'],
                   'LEFTHAND_BOX_SIZE': human_params['leftHandBox'],
                   'LEFTHANDMASS': human_params['leftHandMass'],
                   'LEFTHANDINERTIAIXX': human_params['leftHandIxx'],
                   'LEFTHANDINERTIAIYY': human_params['leftHandIyy'],
                   'LEFTHANDINERTIAIZZ': human_params['leftHandIzz'],

                   'RIGHTUPPERLEG_BOX_ORIGIN': human_params['rightUpperLegBoxOrigin'],
                   'RIGHTUPPERLEG_COM_ORIGIN': human_params['rightUpperLegBoxOrigin'],
                   'RIGHTUPPERLEGHEIGHT': human_params['rightUpperLeg_z'],
                   'RIGHTUPPERLEGRADIUS': 0.5 * human_params['rightUpperLeg_x'],
                   'RIGHTUPPERLEGMASS': human_params['rightUpperLegMass'],
                   'RIGHTUPPERLEGINERTIAIXX': human_params['rightUpperLegIxx'],
                   'RIGHTUPPERLEGINERTIAIYY': human_params['rightUpperLegIyy'],
                   'RIGHTUPPERLEGINERTIAIZZ': human_params['rightUpperLegIzz'],
                   'jRightKnee_ORIGIN': human_params['jRightKnee'],

                   'RIGHTLOWERLEG_BOX_ORIGIN': human_params['rightLowerLegBoxOrigin'],
                   'RIGHTLOWERLEG_COM_ORIGIN': human_params['rightLowerLegBoxOrigin'],
                   'RIGHTLOWERLEGHEIGHT': human_params['rightLowerLeg_z'],
                   'RIGHTLOWERLEGRADIUS': 0.5 * human_params['rightLowerLeg_x'],
                   'RIGHTLOWERLEGMASS': human_params['rightLowerLegMass'],
                   'RIGHTLOWERLEGINERTIAIXX': human_params['rightLowerLegIxx'],
                   'RIGHTLOWERLEGINERTIAIYY': human_params['rightLowerLegIyy'],
                   'RIGHTLOWERLEGINERTIAIZZ': human_params['rightLowerLegIzz'],
                   'jRightAnkle_ORIGIN': human_params['jRightAnkle'],

                   'RIGHTFOOT_BOX_ORIGIN': human_params['rightFootBoxOrigin'],
                   'RIGHTFOOT_COM_ORIGIN': human_params['rightFootBoxOrigin'],
                   'RIGHTFOOT_BOX_SIZE': human_params['rightFootBox'],
                   'RIGHTFOOTMASS': human_params['rightFootMass'],
                   'RIGHTFOOTINERTIAIXX': human_params['rightFootIxx'],
                   'RIGHTFOOTINERTIAIYY': human_params['rightFootIyy'],
                   'RIGHTFOOTINERTIAIZZ': human_params['rightFootIzz'],
                   'jRightBallFoot_ORIGIN': human_params['jRightBallFoot'],

                   'RIGHTTOE_BOX_ORIGIN': human_params['rightToeBoxOrigin'],
                   'RIGHTTOE_COM_ORIGIN': human_params['rightToeBoxOrigin'],
                   'RIGHTTOE_BOX_SIZE': human_params['rightToeBox'],
                   'RIGHTTOEMASS': human_params['rightToeMass'],
                   'RIGHTTOEINERTIAIXX': human_params['rightToeIxx'],
                   'RIGHTTOEINERTIAIYY': human_params['rightToeIyy'],
                   'RIGHTTOEINERTIAIZZ': human_params['rightToeIzz'],

                   'LEFTUPPERLEG_BOX_ORIGIN': human_params['leftUpperLegBoxOrigin'],
                   'LEFTUPPERLEG_COM_ORIGIN': human_params['leftUpperLegBoxOrigin'],
                   'LEFTUPPERLEGHEIGHT': human_params['leftUpperLeg_z'],
                   'LEFTUPPERLEGRADIUS': 0.5 * human_params['leftUpperLeg_x'],
                   'LEFTUPPERLEGMASS': human_params['leftUpperLegMass'],
                   'LEFTUPPERLEGINERTIAIXX': human_params['leftUpperLegIxx'],
                   'LEFTUPPERLEGINERTIAIYY': human_params['leftUpperLegIyy'],
                   'LEFTUPPERLEGINERTIAIZZ': human_params['leftUpperLegIzz'],
                   'jLeftKnee_ORIGIN': human_params['jLeftKnee'],

                   'LEFTLOWERLEG_BOX_ORIGIN': human_params['leftLowerLegBoxOrigin'],
                   'LEFTLOWERLEG_COM_ORIGIN': human_params['leftLowerLegBoxOrigin'],
                   'LEFTLOWERLEGHEIGHT': human_params['leftLowerLeg_z'],
                   'LEFTLOWERLEGRADIUS': 0.5 * human_params['leftLowerLeg_x'],
                   'LEFTLOWERLEGMASS': human_params['leftLowerLegMass'],
                   'LEFTLOWERLEGINERTIAIXX': human_params['leftLowerLegIxx'],
                   'LEFTLOWERLEGINERTIAIYY': human_params['leftLowerLegIyy'],
                   'LEFTLOWERLEGINERTIAIZZ': human_params['leftLowerLegIzz'],
                   'jLeftAnkle_ORIGIN': human_params['jLeftAnkle'],

                   'LEFTFOOT_BOX_ORIGIN': human_params['leftFootBoxOrigin'],
                   'LEFTFOOT_COM_ORIGIN': human_params['leftFootBoxOrigin'],
                   'LEFTFOOT_BOX_SIZE': human_params['leftFootBox'],
                   'LEFTFOOTMASS': human_params['leftFootMass'],
                   'LEFTFOOTINERTIAIXX': human_params['leftFootIxx'],
                   'LEFTFOOTINERTIAIYY': human_params['leftFootIyy'],
                   'LEFTFOOTINERTIAIZZ': human_params['leftFootIzz'],
                   'jLeftBallFoot_ORIGIN': human_params['jLeftBallFoot'],

                   'LEFTTOE_BOX_ORIGIN': human_params['leftToeBoxOrigin'],
                   'LEFTTOE_COM_ORIGIN': human_params['leftToeBoxOrigin'],
                   'LEFTTOE_BOX_SIZE': human_params['leftToeBox'],
                   'LEFTTOEMASS': human_params['leftToeMass'],
                   'LEFTTOEINERTIAIXX': human_params['leftToeIxx'],
                   'LEFTTOEINERTIAIYY': human_params['leftToeIyy'],
                   'LEFTTOEINERTIAIZZ': human_params['leftToeIzz'],
                   }

    # Open template urdf
    rofunc_path = get_rofunc_path()
    human_urdf_dir = os.path.join(rofunc_path, 'simulator/assets/urdf/human')
    urdf_template_path = os.path.join(human_urdf_dir, 'human_xsenstemplate_48dof_zxy.urdf')
    with open(urdf_template_path, 'r') as f:
        urdf_template = f.read()

        for search_text, replace_text in query_table.items():
            replace_text = np.array(replace_text)
            if isinstance(replace_text, np.ndarray):
                replace_text = np.array2string(replace_text, separator=' ').replace('[', '').replace(']', '')
            else:
                replace_text = str(replace_text)
            urdf_template = urdf_template.replace(search_text, str(replace_text))

        fakemass = 0
        fakein = 0
        urdf_template = urdf_template.replace('FAKEMASS', str(fakemass))
        urdf_template = urdf_template.replace('FAKEIN', str(fakein))

    if save_dir is None:
        save_dir = human_urdf_dir

    with open(os.path.join(save_dir, '{}.urdf'.format(actor_name)), 'w') as f:
        f.write(urdf_template)

    beauty_print('Generated human urdf file for {} at {}'.format(actor_name, os.path.join(save_dir, '{}.urdf'.format(
        actor_name))))

    return mvnx_file
