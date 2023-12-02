class Tasks:
    def __init__(self, env_type="isaacgym"):
        if env_type == "isaacgym":
            # Isaac Gym tasks
            from .isaacgym.ant import AntTask
            from .isaacgym.cartpole import CartpoleTask
            from .isaacgym.curi_cabinet import CURICabinetTask
            from .isaacgym.curi_cabinet_image import CURICabinetImageTask
            from .isaacgym.curi_cabinet_bimanual import CURICabinetBimanualTask
            from .isaacgym.curi_coffee_stirring import CURICoffeeStirringTask
            from .isaacgym.franka_cabinet import FrankaCabinetTask
            from .isaacgym.franka_cube_stack import FrankaCubeStackTask
            from .isaacgym.humanoid import HumanoidTask
            from .isaacgym.humanoid_amp import HumanoidAMPTask
            from .isaacgym.ase.humanoid_amp_getup import HumanoidAMPGetupTask
            from .isaacgym.ase.humanoid_perturb import HumanoidPerturbTask
            from .isaacgym.ase.humanoid_heading import HumanoidHeadingTask
            from .isaacgym.ase.humanoid_location import HumanoidLocationTask
            from .isaacgym.ase.humanoid_reach import HumanoidReachTask
            from .isaacgym.ase.humanoid_strike import HumanoidStrikeTask
            from .isaacgym.ase.humanoid_view_motion import HumanoidASEViewMotionTask
            from .isaacgym.hotu.humanoid_view_motion import HumanoidHOTUViewMotionTask

            self.task_map = {
                "Ant": AntTask,
                "Cartpole": CartpoleTask,
                "FrankaCabinet": FrankaCabinetTask,
                "FrankaCubeStack": FrankaCubeStackTask,
                "CURICabinet": CURICabinetTask,
                "CURICabinetImage": CURICabinetImageTask,
                "CURICabinetBimanual": CURICabinetBimanualTask,
                "CURICoffeeStirring": CURICoffeeStirringTask,
                "Humanoid": HumanoidTask,
                "HumanoidAMP": HumanoidAMPTask,
                "HumanoidASEGetupSwordShield": HumanoidAMPGetupTask,
                "HumanoidASEPerturbSwordShield": HumanoidPerturbTask,
                "HumanoidASEHeadingSwordShield": HumanoidHeadingTask,
                "HumanoidASELocationSwordShield": HumanoidLocationTask,
                "HumanoidASEReachSwordShield": HumanoidReachTask,
                "HumanoidASEStrikeSwordShield": HumanoidStrikeTask,
                "HumanoidASEViewMotion": HumanoidASEViewMotionTask,
                "HumanoidHOTUViewMotion": HumanoidHOTUViewMotionTask,
            }
        elif env_type == "omniisaacgym":
            # OmniIsaacGym tasks
            from .omniisaacgym.allegro_hand import AllegroHandOmniTask
            from .omniisaacgym.ant import AntOmniTask
            from .omniisaacgym.anymal import AnymalOmniTask
            from .omniisaacgym.anymal_terrain import AnymalTerrainOmniTask
            from .omniisaacgym.ball_balance import BallBalanceOmniTask
            from .omniisaacgym.cartpole import CartpoleOmniTask
            from .omniisaacgym.crazyflie import CrazyflieOmniTask
            from .omniisaacgym.franka_cabinet import FrankaCabinetOmniTask
            from .omniisaacgym.humanoid import HumanoidOmniTask
            from .omniisaacgym.ingenuity import IngenuityOmniTask
            from .omniisaacgym.quadcopter import QuadcopterOmniTask
            from .omniisaacgym.shadow_hand import ShadowHandOmniTask
            from .omniisaacgym.factory.factory_task_nut_bolt_pick import FactoryNutBoltPickOmniTask
            self.task_map = {
                "AllegroHandOmni": AllegroHandOmniTask,
                "AntOmni": AntOmniTask,
                "AnymalOmni": AnymalOmniTask,
                "AnymalTerrainOmni": AnymalTerrainOmniTask,
                "BallBalanceOmni": BallBalanceOmniTask,
                "CartpoleOmni": CartpoleOmniTask,
                "CrazyflieOmni": CrazyflieOmniTask,
                "FrankaCabinetOmni": FrankaCabinetOmniTask,
                "HumanoidOmni": HumanoidOmniTask,
                "IngenuityOmni": IngenuityOmniTask,
                "QuadcopterOmni": QuadcopterOmniTask,
                "ShadowHandOmni": ShadowHandOmniTask,
                "FactoryNutBoltPickOmni": FactoryNutBoltPickOmniTask,
            }
