import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, EllipseSelector

from rofunc.utils.logger.beauty_logger import beauty_print


def mouse_click_coords(fig, ax, point_style=""):
    """
    Get mouse click coordinates on a figure.

    :param fig: figure object
    :param ax: axes object
    :return: coordinates
    """
    coords = []

    beauty_print(f"Click on the figure to get {point_style} coordinates. Press ESC to exit.", type="module")

    # beauty_print("How many coordinates do you want to get?\n"
    #              "Input a number: ")
    # coords_num = int(input())
    # if coords_num == 0:
    #     return []

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        beauty_print(f'x = {ix}, y = {iy}')

        coords.append((ix, iy))
        ax.scatter(np.array(coords)[:, 0], np.array(coords)[:, 1], color='green', marker='.', s=10, linewidth=1.25)
        fig.canvas.draw()
        return coords

    def onpress(event):
        if event.key == "escape":
            fig.canvas.mpl_disconnect(cid)
            beauty_print('Got all coordinates, closing figure...')
            plt.pause(1)
            plt.close(fig)
        return coords

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onpress)
    return coords


def mouse_select_rec_region(fig, ax):
    """
    Select a rectangle region with mouse.

    :param fig: figure object
    :param ax: axes object
    :return: coordinates of the bbox
    """
    beauty_print("Select a region with mouse. Press ESC to exit.", type="module")
    bbox = []

    def select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        beauty_print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        beauty_print(f"bbox corners: {selector.corners}")
        bbox.append(x1)
        bbox.append(y1)
        bbox.append(x2)
        bbox.append(y2)
        return np.array(bbox)

    def toggle_selector(event):
        selector.set_active(True)
        if event.key == "escape":
            beauty_print('RectangleSelector deactivated.')
            selector.set_active(False)
            plt.close(fig)
        return np.array(bbox)

    selector = RectangleSelector(
        ax, select_callback,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True)

    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    return bbox


def mouse_select_ellip_region(fig, ax):
    """
    Select an ellipse region with mouse.

    :param fig: figure object
    :param ax: axes object
    :return: coordinates of the bbox
    """
    beauty_print("Select a region with mouse. Press ESC to exit.", type="module")

    x1, y1, x2, y2 = 0, 0, 0, 0

    def select_callback(eclick, erelease):
        global x1, y1, x2, y2

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        beauty_print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        beauty_print(f"bbox corners: {selector.corners}")

    def toggle_selector(event):
        selector.set_active(True)
        if event.key == "escape":
            beauty_print('RectangleSelector deactivated.')
            selector.set_active(False)
            plt.close(fig)

    selector = EllipseSelector(
        ax, select_callback,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True)

    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    return np.array([x1, y1, x2, y2])


if __name__ == '__main__':
    x = np.arange(-10, 10)
    y = x ** 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    cxc = mouse_click_coords(fig, ax)
    plt.show()
    print(cxc)

    # fig = plt.figure(layout='constrained')
    # ax = fig.add_subplot(111)
    # N = 100000  # If N is large one can see improvement by using blitting.
    # x = np.linspace(0, 10, N)
    #
    # ax.plot(x, np.sin(2 * np.pi * x))  # plot something
    # mouse_select_ellip_region(fig, ax)
    # plt.show()
