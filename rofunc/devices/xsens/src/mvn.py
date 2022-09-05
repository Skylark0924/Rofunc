FRAMES_ALL = -1

AXIS_ALL = -1
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2
AXIS_W = 3  # The retrieval method will shift indices for w,x,y,z to 0,1,2,3

ANGLE_ALL = -1

ANGLE_ABDUCTION_ADDUCTION = 0
ANGLE_RADIAL_ULNAR_DEVIATION = 0
ANGLE_LATERAL_BENDING = 0

ANGLE_INTERNAL_EXTERNAL_ROTATION = 1
ANGLE_PRONATION_SUPINATION = 1
ANGLE_AXIAL_ROTATION = 1

ANGLE_FLEXION_EXTENSION = 2
ANGLE_DORSIFLEXION_PLANTARFLEXION = 2

SEGMENT_PELVIS = 0
SEGMENT_L5 = 1
SEGMENT_L3 = 2
SEGMENT_T12 = 3
SEGMENT_T8 = 4
SEGMENT_NECK = 5
SEGMENT_HEAD = 6
SEGMENT_RIGHT_SHOULDER = 7
SEGMENT_RIGHT_UPPER_ARM = 8
SEGMENT_RIGHT_FOREARM = 9
SEGMENT_RIGHT_HAND = 10
SEGMENT_LEFT_SHOULDER = 11
SEGMENT_LEFT_UPPER_ARM = 12
SEGMENT_LEFT_FOREARM = 13
SEGMENT_LEFT_HAND = 14
SEGMENT_RIGHT_UPPER_LEG = 15
SEGMENT_RIGHT_LOWER_LEG = 16
SEGMENT_RIGHT_FOOT = 17
SEGMENT_RIGHT_TOE = 18
SEGMENT_LEFT_UPPER_LEG = 19
SEGMENT_LEFT_LOWER_LEG = 20
SEGMENT_LEFT_FOOT = 21
SEGMENT_LEFT_TOE = 22

SEGMENT_CENTER_OF_MASS = 1000

SEGMENTS = {SEGMENT_PELVIS: 'Pelvis',
            SEGMENT_L5: 'L5',
            SEGMENT_L3: 'L3',
            SEGMENT_T12: 'T12',
            SEGMENT_T8: 'T8',
            SEGMENT_NECK: 'Neck',
            SEGMENT_HEAD: 'Head',
            SEGMENT_RIGHT_SHOULDER: 'RightShoulder',
            SEGMENT_RIGHT_UPPER_ARM: 'RightUpperArm',
            SEGMENT_RIGHT_FOREARM: 'RightForeArm',
            SEGMENT_RIGHT_HAND: 'RightHand',
            SEGMENT_LEFT_SHOULDER: 'LeftShoulder',
            SEGMENT_LEFT_UPPER_ARM: 'LeftUpperArm',
            SEGMENT_LEFT_FOREARM: 'LeftForeArm',
            SEGMENT_LEFT_HAND: 'LeftHand',
            SEGMENT_RIGHT_UPPER_LEG: 'RightUpperLeg',
            SEGMENT_RIGHT_LOWER_LEG: 'RightLowerLeg',
            SEGMENT_RIGHT_FOOT: 'RightFoot',
            SEGMENT_RIGHT_TOE: 'RightToe',
            SEGMENT_LEFT_UPPER_LEG: 'LeftUpperLeg',
            SEGMENT_LEFT_LOWER_LEG: 'LeftLowerLeg',
            SEGMENT_LEFT_FOOT: 'LeftFoot',
            SEGMENT_LEFT_TOE: 'LeftToe',
            SEGMENT_CENTER_OF_MASS: 'CoM'}


# regular joints

JOINT_L5_S1 = 0
JOINT_L4_L3 = 1
JOINT_L1_T12 = 2
JOINT_T9_T8 = 3
JOINT_T1_C7 = 4
JOINT_C1_HEAD = 5
JOINT_RIGHT_T4_SHOULDER = 6
JOINT_RIGHT_SHOULDER = 7
JOINT_RIGHT_ELBOW = 8
JOINT_RIGHT_WRIST = 9
JOINT_LEFT_T4_SHOULDER = 10
JOINT_LEFT_SHOULDER = 11
JOINT_LEFT_ELBOW = 12
JOINT_LEFT_WRIST = 13
JOINT_RIGHT_HIP = 14
JOINT_RIGHT_KNEE = 15
JOINT_RIGHT_ANKLE = 16
JOINT_RIGHT_BALL_FOOT = 17
JOINT_LEFT_HIP = 18
JOINT_LEFT_KNEE = 19
JOINT_LEFT_ANKLE = 20
JOINT_LEFT_BALL_FOOT = 21

JOINTS = {JOINT_L5_S1: 'jL5S1',
          JOINT_L4_L3: 'jL4L3',
          JOINT_L1_T12: 'jL1T12',
          JOINT_T9_T8: 'jT9T8',
          JOINT_T1_C7: 'jT1C7',
          JOINT_C1_HEAD: 'jC1Head',
          JOINT_RIGHT_T4_SHOULDER: 'jRightT4Shoulder',
          JOINT_RIGHT_SHOULDER: 'jRightShoulder',
          JOINT_RIGHT_ELBOW: 'jRightElbow',
          JOINT_RIGHT_WRIST: 'jRightWrist',
          JOINT_LEFT_T4_SHOULDER: 'jLeftT4Shoulder',
          JOINT_LEFT_SHOULDER: 'jLeftShoulder',
          JOINT_LEFT_ELBOW: 'jLeftElbow',
          JOINT_LEFT_WRIST: 'jLeftWrist',
          JOINT_RIGHT_HIP: 'jRightHip',
          JOINT_RIGHT_KNEE: 'jRightKnee',
          JOINT_RIGHT_ANKLE: 'jRightAnkle',
          JOINT_RIGHT_BALL_FOOT: 'jRightBallFoot',
          JOINT_LEFT_HIP: 'jLeftHip',
          JOINT_LEFT_KNEE: 'jLeftKnee',
          JOINT_LEFT_ANKLE: 'jLeftAnkle',
          JOINT_LEFT_BALL_FOOT: 'jLeftBallFoot'}

PARAMETER_JOINTS = {JOINT_L5_S1: 'j_l5_s1',
                    JOINT_L4_L3: 'j_l4_l3',
                    JOINT_L1_T12: 'j_l1_t12',
                    JOINT_T9_T8: 'j_t9_t8',
                    JOINT_T1_C7: 'j_t1_c7',
                    JOINT_C1_HEAD: 'j_c1_head',
                    JOINT_RIGHT_T4_SHOULDER: 'j_right_t4_shoulder',
                    JOINT_RIGHT_SHOULDER: 'j_right_shoulder',
                    JOINT_RIGHT_ELBOW: 'j_right_elbow',
                    JOINT_RIGHT_WRIST: 'j_right_wrist',
                    JOINT_LEFT_T4_SHOULDER: 'j_left_t4_shoulder',
                    JOINT_LEFT_SHOULDER: 'j_left_shoulder',
                    JOINT_LEFT_ELBOW: 'j_left_elbow',
                    JOINT_LEFT_WRIST: 'j_left_wrist',
                    JOINT_RIGHT_HIP: 'j_right_hip',
                    JOINT_RIGHT_KNEE: 'j_right_knee',
                    JOINT_RIGHT_ANKLE: 'j_right_ankle',
                    JOINT_RIGHT_BALL_FOOT: 'j_right_ball_foot',
                    JOINT_LEFT_HIP: 'j_left_hip',
                    JOINT_LEFT_KNEE: 'j_left_knee',
                    JOINT_LEFT_ANKLE: 'j_left_ankle',
                    JOINT_LEFT_BALL_FOOT: 'j_left_ball_foot'}


# ergonomic joints

ERGO_JOINT_T8_HEAD = 0
ERGO_JOINT_T8_LEFT_UPPER_ARM = 1
ERGO_JOINT_T8_RIGHT_UPPER_ARM = 2
ERGO_JOINT_PELVIS_T8 = 3
ERGO_JOINT_VERTICAL_PELVIS = 4
ERGO_JOINT_VERTICAL_T8 = 5

ERGO_JOINT_INDICES = {ERGO_JOINT_T8_HEAD,
                      ERGO_JOINT_T8_LEFT_UPPER_ARM,
                      ERGO_JOINT_T8_RIGHT_UPPER_ARM,
                      ERGO_JOINT_PELVIS_T8,
                      ERGO_JOINT_VERTICAL_PELVIS,
                      ERGO_JOINT_VERTICAL_T8}

ERGO_JOINTS = {ERGO_JOINT_T8_HEAD: 'T8_Head',
               ERGO_JOINT_T8_LEFT_UPPER_ARM: 'T8_LeftUpperArm',
               ERGO_JOINT_T8_RIGHT_UPPER_ARM: 'T8_RightUpperArm',
               ERGO_JOINT_PELVIS_T8: 'Pelvis_T8',
               ERGO_JOINT_VERTICAL_PELVIS: 'Vertical_Pelvis',
               ERGO_JOINT_VERTICAL_T8: 'Vertical_T8'}

ERGO_PARAMETER_JOINTS = {ERGO_JOINT_T8_HEAD: 't8_head',
                         ERGO_JOINT_T8_LEFT_UPPER_ARM: 't8_left_upper_arm',
                         ERGO_JOINT_T8_RIGHT_UPPER_ARM: 't8_right_upper_arm',
                         ERGO_JOINT_PELVIS_T8: 'pelvis_t8',
                         ERGO_JOINT_VERTICAL_PELVIS: 'vertical_p1elvis',
                         ERGO_JOINT_VERTICAL_T8: 'vertical_t8'}


# foot contacts

FOOT_CONTACT_LEFT_HEEL = 1
FOOT_CONTACT_LEFT_TOE = 2
FOOT_CONTACT_RIGHT_HEEL = 4
FOOT_CONTACT_RIGHT_TOE = 8

# some points

POINT_JOINT_ANKLE = 0
POINT_JOINT_BALL_FOOT = 1
POINT_HEEL_FOOT = 2
POINT_FIRST_METATARSAL = 3
POINT_FIFTH_METATARSAL = 4
POINT_PIVOT_FOOT = 5
POINT_HEEL_CENTER = 6
POINT_TOP_OF_FOOT = 7

POINTS_LEFT_FOOT = {
    POINT_JOINT_ANKLE: 'jLeftAnkle',
    POINT_JOINT_BALL_FOOT: 'jLeftBallFoot',
    POINT_HEEL_FOOT: 'pLeftHeelFoot',
    POINT_FIRST_METATARSAL: 'pLeftFirstMetatarsal',
    POINT_FIFTH_METATARSAL: 'pLeftFifthMetatarsal',
    POINT_PIVOT_FOOT: 'pLeftPivotFoot',
    POINT_HEEL_CENTER: 'pLeftHeelCenter',
    POINT_TOP_OF_FOOT: 'pLeftTopOfFoot',
}

POINTS_RIGHT_FOOT = {
    POINT_JOINT_ANKLE: 'jRightAnkle',
    POINT_JOINT_BALL_FOOT: 'jRightBallFoot',
    POINT_HEEL_FOOT: 'pRightHeelFoot',
    POINT_FIRST_METATARSAL: 'pRightFirstMetatarsal',
    POINT_FIFTH_METATARSAL: 'pRightFifthMetatarsal',
    POINT_PIVOT_FOOT: 'pRightPivotFoot',
    POINT_HEEL_CENTER: 'pRightHeelCenter',
    POINT_TOP_OF_FOOT: 'pRightTopOfFoot',
}

# Toe
# POINT_JOINT_BALL_FOOT = 0 # overlaps with foot segment
POINT_TOE = 1
POINT_JOINT_TOE = 2

POINTS_LEFT_TOE = {
    # POINT_JOINT_BALL_FOOT: 'jLeftBallFoot',
    POINT_TOE: 'pLeftToe',
    POINT_JOINT_TOE: 'jLeftToe',
}

POINTS_RIGHT_TOE = {
    # POINT_JOINT_BALL_FOOT: 'jRightBallFoot',
    POINT_TOE: 'pRightToe',
    POINT_JOINT_TOE: 'jRightToe',
}
