"""
ZED Export
=============

This example shows how to export a SVO file to a video or image sequence.
"""

import rofunc as rf

# Export a single svo file
# rf.zed.export('/home/skylark/Data/zed_record/HD1080_SN36605831_18-03-10.svo', 0)

# Export a batch of SVO files in a directory
rf.zed.export_batch('/home/skylark/Data/zed_record/20221215_173322', all_mode=False, mode_lst=[0], core_num=20)
