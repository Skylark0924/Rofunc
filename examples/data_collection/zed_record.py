"""
ZED Record
================

This example shows how to record multiple ZED cameras simultaneously.
"""

import rofunc as rf
from datetime import datetime

root_dir = '/home/skylark/Data/zed_record'
exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

rf.zed.record(root_dir, exp_name)
