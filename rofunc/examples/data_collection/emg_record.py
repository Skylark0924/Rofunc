"""
Delsys EMG Record
=================

This example shows how to record EMG data via Delsys Trigno Control Utility.
"""

import rofunc as rf
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-a', '--addr',
    dest='host',
    default='10.13.90.12',
    help="IP address of the machine running TCU. Default is localhost.")
args = parser.parse_args()

root_dir = '/home/ubuntu/Data/emg_record'
exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

# For instance, 4 channels, 2000 samples per second and 10 seconds are chosen.
rf.emg.record(args.host, 4, 2000, 12, root_dir, exp_name)
