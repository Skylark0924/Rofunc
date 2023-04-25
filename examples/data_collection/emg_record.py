"""
Delsys EMG Record
=================

This example shows how to record EMG data via Delsys Trigno Control Utility.
"""

import argparse
import os.path
from datetime import datetime

import rofunc as rf

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-a', '--addr',
    dest='host',
    default='10.13.162.196',
    help="IP address of the machine running TCU. Default is localhost.")
args = parser.parse_args()

root_dir = '/home/ubuntu/Data/emg_record/20230306'
exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# For instance, 6 channels, 2000 samples per second and 30 seconds are chosen.
rf.emg.record(args.host, 6, 2000, 40, root_dir, exp_name)
