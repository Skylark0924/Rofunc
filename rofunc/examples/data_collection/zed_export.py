"""
ZED Export
=============

This example shows how to export a SVO file to a video or image sequence.
"""

import rofunc as rf

# Export a single svo file
rf.zed.export('/home/skylark/Data/zed_record/HD1080_SN36605831_18-03-10.svo', 0)

# Export a batch of SVO files in a directory
# rf.zed.export_batch('/home/ubuntu/Data/06_24/Video/20220624_1649', core_num=20)
