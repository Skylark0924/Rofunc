"""
Optitrack Record
================

This example shows how to record Optitrack data via network streaming.
"""

import rofunc as rf
from datetime import datetime

root_dir = '/home/skylark/Data/optitrack_record'
exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
ip = '192.168.13.118'
port = 6688

rf.optitrack.record(root_dir, exp_name, ip, port)
