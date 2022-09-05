"""
Tests communication with and data acquisition from a Delsys Trigno wireless
EMG system. Delsys Trigno Control Utility needs to be installed and running,
and the device needs to be plugged in. Tests can be run with a device connected
to a remote machine if needed.
The tests run by this script are very simple and are by no means exhaustive. It
just sets different numbers of channels and ensures the data received is the
correct shape.
Use `-h` or `--help` for options.
"""

import argparse

try:
    import pytrigno
except ImportError:
    import sys
    sys.path.insert(0, '..')
    import pytrigno


def check_emg(host):
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=270,
                             host=host)

    # test single-channel
    # dev.start()
    # for i in range(4):
    #     data = dev.read()
    #     assert data.shape == (1, 270)
    # print('single-channel achieved')
    # dev.stop()

    # test multi-channel
    dev.set_channel_range((0, 5))
    dev.start()
    # for i in range(10):
    #     data = dev.read()
    #     assert data.shape == (6, 270)
    # print('multi-channel achieved')

    while True:
        data = dev.read() * 10000
        assert data.shape == (6, 270)
        print(data)
    dev.stop()


def check_accel(host):
    dev = pytrigno.TrignoAccel(channel_range=(0, 2), samples_per_read=10,
                               host=host)

    dev.start()
    for i in range(4):
        data = dev.read()
        assert data.shape == (3, 10)
    print('Accel achieved')
    dev.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--addr',
        dest='host',
        default='10.13.52.82',
        help="IP address of the machine running TCU. Default is localhost.")
    args = parser.parse_args()

    check_emg(args.host)
    check_accel(args.host)
