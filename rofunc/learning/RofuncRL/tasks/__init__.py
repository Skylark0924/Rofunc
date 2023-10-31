class Tasks:
    def __init__(self, env_type="isaacgym"):
        if env_type == "isaacgym":
            # Isaac Gym tasks
            from .ant import AntTask
            from .cartpole import CartpoleTask
            from .curi_cabinet import CURICabinetTask
            from .curi_cabinet_image import CURICabinetImageTask
            from .curi_cabinet_bimanual import CURICabinetBimanualTask
            from .curi_coffee_stirring import CURICoffeeStirringTask
            from .franka_cabinet import FrankaCabinetTask
            from .franka_cube_stack import FrankaCubeStackTask
            from .humanoid import HumanoidTask
            from .humanoid_amp import HumanoidAMPTask
            from .ase.humanoid_amp_getup import HumanoidAMPGetupTask
            from .ase.humanoid_perturb import HumanoidPerturbTask
            from .ase.humanoid_heading import HumanoidHeadingTask
            from .ase.humanoid_location import HumanoidLocationTask
            from .ase.humanoid_reach import HumanoidReachTask
            from .ase.humanoid_strike import HumanoidStrikeTask
            from .ase.humanoid_view_motion import HumanoidViewMotionTask

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
                "HumanoidViewMotion": HumanoidViewMotionTask,
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
