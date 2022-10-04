from .src import pytrigno
import argparse

def record(host):
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=1000,
                             host=host)

    # test single-channel
    dev.start()
    # for i in range(4):
    while True:
        data = dev.read()
        print(data)
        assert data.shape == (1, 1000)
    print('single-channel achieved')
    dev.stop()

    # test multi-channel
    # dev.set_channel_range((0, 5))
    # dev.start()
    # for i in range(10):
    #     data = dev.read()
    #     print(data * 1e12)
    #     assert data.shape == (6, 270)
    # print('multi-channel achieved')
    #
    # while True:
    #     data = dev.read() * 1e10
    #     assert data.shape == (6, 270)
    #     print(data[0])
    # dev.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--addr',
        dest='host',
        default='10.13.58.70',
        help="IP address of the machine running TCU. Default is localhost.")
    args = parser.parse_args()

    record(args.host)
