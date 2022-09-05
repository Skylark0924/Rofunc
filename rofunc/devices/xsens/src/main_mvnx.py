import os
import argparse
import matplotlib.pyplot as plt
from load_mvnx import load_mvnx


# Convert mvnx file to python data
def main(file_name):
    # Check for file existence
    if not os.path.isfile(file_name):
        raise Exception("File %s could not be found" % file_name)

    tokens = file_name.lower().split('.')
    extension = tokens[-1]

    # Check for file extension
    if not extension == 'mvnx':
        raise Exception("File must be an .mvnx file")

    # Load data
    mvnx_file = load_mvnx(file_name)

    # Read some basic data from the file
    comments = mvnx_file.comments
    frame_rate = mvnx_file.frame_rate
    configuration = mvnx_file.configuration
    original_file_name = mvnx_file.original_file_name
    recording_date = mvnx_file.recording_date
    actor_name = mvnx_file.actor_name
    frame_count = mvnx_file.frame_count
    version = mvnx_file.version
    segment_count = mvnx_file.segment_count
    joint_count = mvnx_file.joint_count

    # Read the data from the structure e.g. first segment
    idx = 0
    segment_name = mvnx_file.segment_name_from_index(idx)
    segment_pos = mvnx_file.get_segment_pos(idx)

    # Alternatively, use the generic method get_data() with the data set and field. E.g.:
    # segment_pos = mvnx_file.get_data('segment_data', 'pos', idx)

    if segment_pos:
        # Plot position of a segment
        plt.figure(0)
        plt.plot(segment_pos)
        plt.xlabel('frames')
        plt.ylabel('Position in the global frame')
        plt.title('Position of ' + segment_name + ' segment')
        plt.legend(['x', 'y', 'z'])
        plt.draw()

        # Plot 3D displacement of a segment
        x, y, z = map(list, zip(*[[frame[0], frame[1], frame[2]] for frame in segment_pos]))
        plt.figure(1)
        plt.axes(projection="3d")
        plt.plot(x, y, z)
        plt.xlabel('frames')
        plt.title('Position of ' + segment_name + ' segment in 3D')
        plt.draw()

        plt.show()


if __name__ == '__main__':

    # Program entry point
    parser = argparse.ArgumentParser()
    parser.add_argument('--mvnx_file', required=True, type=str, help='The MVNX file to load', nargs='?')
    args = parser.parse_args()

    try:
        main(args.mvnx_file)
    except Exception as e:
        print("Error: %s" % e)
