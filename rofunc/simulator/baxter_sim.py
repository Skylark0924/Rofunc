from rofunc.simulator.base.base_sim import RobotSim


class BaxterSim(RobotSim):
    def __init__(self, args, **kwargs):
        super().__init__(args, robot_name="baxter", **kwargs)
