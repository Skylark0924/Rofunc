"""
Xsens Record
================

This example shows how to record Xsens MTw Awinda via network streaming.
"""

import rofunc as rf
from datetime import datetime

root_dir = '/home/skylark/Data/xsens_record'
exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
ip = '192.168.13.20'
port = 9763

rf.xsens.record(root_dir, exp_name, ip, port)
