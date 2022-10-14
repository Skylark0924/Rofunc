import argparse
import os

import numpy as np

from .src import pytrigno


def record(host, n, samples_per_read, t):
    """
    Communication with and data acquisition from a Delsys Trigno wireless EMG system.
    Delsys Trigno Control Utility needs to be installed and running on, and the device
    needs to be plugged in. Records can be run with a device connected to a remote machine.

    Args:
        host: host of a remote machine
        n: number of emg channels
        samples_per_read： number of samples per read
        t: number of batches (running time * 2000 / samples_per_read)

    Returns: None
    """
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=samples_per_read,
                             host=host)

    dev.set_channel_range((0, n - 1))
    dev.start()
    data_sensor = []
    for i in range(int(t)):
        # while True:
        data = dev.read() * 1e6
        print(data)
        assert data.shape == (n, samples_per_read)
        data_sensor.append(data)
    print(n, '-channel achieved')
    dev.stop()

    data_sensor = np.reshape(np.transpose(np.array(data_sensor), (0, 2, 1)), (-1, n))
    np.save(os.path.join('./data', 'emg_data.npy'), data_sensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--addr',
        dest='host',
        default='10.13.166.60',
        help="IP address of the machine running TCU. Default is localhost.")
    args = parser.parse_args()

    # For instance, 4 channels, 2000 samples per read and 10 batches are chosen.
    import rofunc as rf
    rf.emg.record(args.host, 4, 2000, 10)
