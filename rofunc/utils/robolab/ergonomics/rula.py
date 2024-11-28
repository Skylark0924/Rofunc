import numpy as np


def upperArm_score(header, data):
    """returns rula score for upper Arm
    takes all three anatomical angles for shoulder provided by MR3s myoMOTION
    """
    # No. 1 of RULA protocol
    angle_limits = [[-20, 20], [-20, 20, 45], [45, 90], [90]]  # rula definition
    angle_abdu = 30  # own definition/limit

    shoulder_flex = np.zeros((len(data), 2))  # [0,1 == left, right]
    shoulder_abdu = np.zeros((len(data), 2))  # [0,1 == left, right]
    shoulder_flex[:, 0] = data[:, np.where(header == "Schulter Flexion LT,Grad")[0][0]]
    shoulder_flex[:, 1] = data[:, np.where(header == "Schulter Flexion RT,Grad")[0][0]]
    shoulder_abdu[:, 0] = data[:, np.where(header == "Schulter Abduktion LT,Grad")[0][0]]
    shoulder_abdu[:, 1] = data[:, np.where(header == "Schulter Abduktion RT,Grad")[0][0]]

    score_left = np.zeros((len(data), 1))  # [0,1 == angleScore, additional]
    score_right = np.zeros((len(data), 1))  # [0,1 == angleScore, additional]

    for a in range(len(data)):
        # scoring for left side
        if shoulder_flex[a, 0] > angle_limits[0][0] and shoulder_flex[a, 0] <= angle_limits[0][1]:
            score_left[a, 0] = 1
        elif shoulder_flex[a, 0] < angle_limits[1][0]:
            score_left[a, 0] = 2
        elif shoulder_flex[a, 0] > angle_limits[1][1] and shoulder_flex[a, 0] <= angle_limits[1][2]:
            score_left[a, 0] = 2
        elif shoulder_flex[a, 0] > angle_limits[2][0] and shoulder_flex[a, 0] <= angle_limits[2][1]:
            score_left[a, 0] = 3
        elif shoulder_flex[a, 0] > angle_limits[3][0]:
            score_left[a, 0] = 4

        # scoring for right side
    for b in range(len(data)):
        if shoulder_flex[b, 1] > angle_limits[0][0] and shoulder_flex[a, 0] <= angle_limits[0][1]:
            score_right[b, 0] = 1
        elif shoulder_flex[b, 1] < angle_limits[1][0]:
            score_right[b, 0] = 2
        elif shoulder_flex[b, 1] > angle_limits[1][1] and shoulder_flex[a, 0] <= angle_limits[1][2]:
            score_right[b, 0] = 2
        elif shoulder_flex[b, 1] > angle_limits[2][0] and shoulder_flex[a, 0] <= angle_limits[2][1]:
            score_right[b, 0] = 3
        elif shoulder_flex[b, 1] > angle_limits[3][0]:
            score_right[b, 0] = 4

    abucted_arm_left = np.where(shoulder_abdu[:, 0] > angle_abdu)[0]
    abucted_arm_right = np.where(shoulder_abdu[:, 1] > angle_abdu)[0]

    score_left[abucted_arm_left, 0] += 1
    score_right[abucted_arm_right, 0] += 1

    print("\n???????????????\nQUESTION\n???????????????")
    while True:
        try:
            a = int(input(
                "Wie lange ist die Person angelehnt ODER\n der Arm unterstützt?\n [Angabe in % relativ zur Gesamtdauer des Vorgangs]\n\n\t"))
            a = a / 100
            break
        except ValueError:
            print("Oppsidaysi! That wasn't  an integer Value. Try again!")

    print('upper Arm:')
    print(np.mean(score_left[:, 0]))
    print(np.mean(score_right[:, 0]))

    return score_left, score_right
