import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def draw_arc_old(centerPt, Pt1, Pt2):
    image = Image.new('RGB', (90, 90), 'white')
    draw = ImageDraw.Draw(image)

    arc_color = (0, 0, 0, 255)  # RGBA: Black
    # First calculate the length from the center point to both ends of the arc
    length1_x = abs(centerPt[0] - Pt1[0])
    length1_y = abs(centerPt[1] - Pt1[1])
    length2_x = abs(centerPt[0] - Pt2[0])
    length2_y = abs(centerPt[1] - Pt2[1])
    length1 = math.sqrt((length1_x ** 2) + (length1_y ** 2))
    length2 = math.sqrt((length2_x ** 2) + (length2_y ** 2))
    # pick the longest one
    longer_len = length1 if (length1 > length2) else length2
    # The circumscribed rectangle of the arc
    arc_left_top_x = centerPt[0] - longer_len
    arc_left_top_y = centerPt[1] - longer_len
    arc_right_bottom_x = centerPt[0] + longer_len
    arc_right_bottom_y = centerPt[1] + longer_len
    # circumscribed rectangle
    # draw.rectangle([arc_left_top_x,arc_left_top_y,arc_right_bottom_x,arc_right_bottom_y],fill=None,outline='blue',width=1)
    # Calculate the angle from the center point to the end point of the arc
    start_angle = math.atan2(Pt2[1] - centerPt[1], Pt2[0] - centerPt[0]) * 180 / math.pi
    end_angle = math.atan2(Pt1[1] - centerPt[1], Pt1[0] - centerPt[0]) * 180 / math.pi
    # print([arc_left_top_x,arc_left_top_y,arc_right_bottom_x,arc_right_bottom_y])
    # print(f"Start angle {start_angle}")
    # print(f"End angle {end_angle}")

    # Draw an arc
    draw.arc([arc_left_top_x, arc_left_top_y, arc_right_bottom_x, arc_right_bottom_y], start=start_angle,
             end=end_angle, fill=arc_color, width=1)
    plt.show()


def draw_arc(center, radius, theta1, theta2, color):
    # (x-a)²+(y-b)²=r²
    a, b = center
    theta = np.linspace(theta1, theta2, 100)
    x = a + radius * np.cos(theta)
    y = b + radius * np.sin(theta)
    plt.plot(x, y, color=color)

    # plt.legend()
    # plt.show()
    xy = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)))).reshape((1, -1, 2))
    return xy


def draw_connect(l_curve, r_curve, type):
    for i in range(0, len(l_curve), 10):
        plt.plot([l_curve[i, 0], r_curve[i, 0]], [l_curve[i, 1], r_curve[i, 1]], ls=type, alpha=0.7)


if __name__ == '__main__':
    draw_arc([-1, 0], 1, 0 * np.pi / 3, 2 * np.pi / 3, color='red')
