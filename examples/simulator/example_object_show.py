"""
Object Simulator and Camera Tracking
============================================================

Example for showing objects in the simulator and tracking cameras with rgb, depth, and segmentation modes.
"""

import isaacgym
import rofunc as rf

args = rf.config.get_sim_config("Objects")
asset_files = [
    "urdf/ycb/002_master_chef_can/002_master_chef_can.urdf",
    "urdf/ycb/003_cracker_box/003_cracker_box.urdf",
    "urdf/ycb/004_sugar_box/004_sugar_box.urdf",
    "urdf/ycb/005_tomato_soup_can/005_tomato_soup_can.urdf",
    "urdf/ycb/006_mustard_bottle/006_mustard_bottle.urdf",
    "urdf/ycb/007_tuna_fish_can/007_tuna_fish_can.urdf",
    "urdf/ycb/008_pudding_box/008_pudding_box.urdf",
    "urdf/ycb/009_gelatin_box/009_gelatin_box.urdf",
    "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
    "urdf/ycb/011_banana/011_banana.urdf",
    "urdf/ycb/048_hammer/048_hammer.urdf",
    "urdf/ycb/019_pitcher_base/019_pitcher_base.urdf",
    "urdf/ycb/032_knife/032_knife.urdf",
]
object_sim = rf.sim.ObjectSim(args, asset_file=asset_files)
object_sim.create_track_cameras()
object_sim.show(mode="seg")
