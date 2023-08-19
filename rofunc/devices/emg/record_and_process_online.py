import argparse
from queue import Queue
from threading import Thread
from time import time

import numpy as np

from export import process_all_channels
from src import pytrigno

sample_rate = 2000
samples_per_read = 200
n = 2
k = 4


# A thread that produces data
def producer(out_q):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--addr',
        dest='host',
        default='10.13.166.60',
        help="IP address of the machine running TCU. Default is localhost.")
    args = parser.parse_args()
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=samples_per_read,
                             host=args.host)
    dev.set_channel_range((0, n - 1))
    dev.start()
    while True:
        # Produce some data
        start_time = time()
        data = dev.read() * 1e6
        assert data.shape == (n, samples_per_read)
        print(n, '-channel achieved')
        data = np.transpose(np.array(data), (1, 0))
        end_time = time()
        out_q.put(data)
        print(end_time - start_time)
    dev.stop()


# A thread that consumes data
def consumer(in_q):
    while True:
        # Get some data
        data = in_q.get()
        # Process the data
        data_filter, data_clean, data_mvc, data_abs = process_all_channels(data, n, sample_rate, k)
        print(data_mvc)
        # Indicate completion
        in_q.task_done()


if __name__ == '__main__':
    # Create the shared queue and launch both threads
    q = Queue()
    t1 = Thread(target=consumer, args=(q,))
    t2 = Thread(target=producer, args=(q,))
    t1.start()
    t2.start()

    # Wait for all produced items to be consumed
    q.join()
