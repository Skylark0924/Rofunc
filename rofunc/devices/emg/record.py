import argparse

import numpy as np

from src import pytrigno


def record(host, n, t):
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=1000,
                             host=host)

    # test n-channel
    dev.set_channel_range((0, n - 1))
    dev.start()

    data_sensor_1 = []
    data_sensor_2 = []
    data_sensor_3 = []
    data_sensor_4 = []
    data_sensor_5 = []
    data_sensor_6 = []
    for i in range(int(t)):
        # while True:
        data = dev.read() * 1e6
        print(data)
        assert data.shape == (n, 1000)
        data_1 = data[0:1]
        data_2 = data[1:2]
        data_3 = data[2:3]
        data_4 = data[3:4]
        data_5 = data[4:5]
        data_6 = data[5:6]
        data_sensor_1.append(data_1)
        data_sensor_2.append(data_2)
        data_sensor_3.append(data_3)
        data_sensor_4.append(data_4)
        data_sensor_5.append(data_5)
        data_sensor_6.append(data_6)
    print(n, '-channel achieved')
    dev.stop()
    return data_sensor_1, data_sensor_2, data_sensor_3, data_sensor_4, data_sensor_5, data_sensor_6


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--addr',
        dest='host',
        default='10.13.166.60',
        help="IP address of the machine running TCU. Default is localhost.")
    args = parser.parse_args()

    n = 6
    t = 30
    data_sensor_1, data_sensor_2, data_sensor_3, data_sensor_4, data_sensor_5, data_sensor_6 = record(args.host, n, t)
    data_sensor_1 = np.reshape(np.array(data_sensor_1), -1)
    data_sensor_2 = np.reshape(np.array(data_sensor_2), -1)
    data_sensor_3 = np.reshape(np.array(data_sensor_3), -1)
    data_sensor_4 = np.reshape(np.array(data_sensor_4), -1)
    data_sensor_5 = np.reshape(np.array(data_sensor_5), -1)
    data_sensor_6 = np.reshape(np.array(data_sensor_6), -1)
    # print(data_sensor_4)
    np.save("data_sensor_4_time_{}.npy".format(t), data_sensor_4)
    # np.savetxt('data_sensor_4_time_{}.txt'.format(t), data_sensor_4, delimiter=',', fmt='%.08f')
