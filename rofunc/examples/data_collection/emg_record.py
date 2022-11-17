"""
Delsys EMG Record
=================

This example shows how to record EMG data via Delsys Trigno Control Utility.
"""


import rofunc as rf
import argparse

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-a', '--addr',
    dest='host',
    default='10.13.166.60',
    help="IP address of the machine running TCU. Default is localhost.")
args = parser.parse_args()

# For instance, 4 channels, 2000 samples per read and 10 batches are chosen.
rf.emg.record(args.host, 4, 2000, 10)
