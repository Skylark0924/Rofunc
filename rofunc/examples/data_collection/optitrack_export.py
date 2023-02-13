"""
Optitrack Export
================

This example shows how to clean the .csv files outputted by Optitrack.
"""

import rofunc as rf

data = rf.optitrack.export('/home/ubuntu/Data/optitrack/Take 2022-12-21 05.41.07 PM.csv')
print(data)
