from rofunc.simulator.base.base_sim import RobotSim


class SawyerSim(RobotSim):
    def __init__(self, args, **kwargs):
        super().__init__(args, robot_name="sawyer", **kwargs)
