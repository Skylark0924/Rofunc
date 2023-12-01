"""
CURI Interactive Mode
============================================================

Show the interactive mode of the CURI simulator.
"""
import isaacgym
import rofunc as rf

args = rf.config.get_sim_config("Objects")
asset_files = ["urdf/ycb/002_master_chef_can/002_master_chef_can.urdf",
                "urdf/ycb/003_cracker_box/003_cracker_box.urdf",
                "urdf/ycb/004_sugar_box/004_sugar_box.urdf",
                # "urdf/ycb/005_tomato_soup_can/005_tomato_soup_can.urdf",
                # "urdf/ycb/006_mustard_bottle/006_mustard_bottle.urdf",
               ]
object_sim = rf.sim.ObjectSim(args, asset_file=asset_files)
object_sim.show()
