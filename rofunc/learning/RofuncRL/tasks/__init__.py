class Tasks:
    def __init__(self, env_type="isaacgym"):
        if env_type == "isaacgym":
            # Isaac Gym tasks
            from .isaacgymenv.ant import AntTask
            from .isaacgymenv.cartpole import CartpoleTask
            from .isaacgymenv.curi_cabinet import CURICabinetTask
            from .isaacgymenv.curi_cabinet_image import CURICabinetImageTask
            from .isaacgymenv.curi_cabinet_bimanual import CURICabinetBimanualTask
            from .isaacgymenv.curi_coffee_stirring import CURICoffeeStirringTask
            from .isaacgymenv.franka_cabinet import FrankaCabinetTask
            from .isaacgymenv.franka_cube_stack import FrankaCubeStackTask
            from .isaacgymenv.humanoid import HumanoidTask
            from .isaacgymenv.humanoid_amp import HumanoidAMPTask
            from .isaacgymenv.ase.humanoid_amp_getup import HumanoidAMPGetupTask
            from .isaacgymenv.ase.humanoid_perturb import HumanoidPerturbTask
            from .isaacgymenv.ase.humanoid_heading import HumanoidHeadingTask
            from .isaacgymenv.ase.humanoid_location import HumanoidLocationTask
            from .isaacgymenv.ase.humanoid_reach import HumanoidReachTask
            from .isaacgymenv.ase.humanoid_strike import HumanoidStrikeTask
            from .isaacgymenv.ase.humanoid_view_motion import HumanoidASEViewMotionTask
            from .isaacgymenv.hotu.humanoid_view_motion import HumanoidHOTUViewMotionTask
            from .isaacgymenv.hands.shadow_hand_over import ShadowHandOverTask

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
                "ShadowHandOver": ShadowHandOverTask,
            }
        elif env_type == "omniisaacgym":
            # OmniIsaacGym tasks
            from .omniisaacgymenv.allegro_hand import AllegroHandOmniTask
            from .omniisaacgymenv.ant import AntOmniTask
            from .omniisaacgymenv.anymal import AnymalOmniTask
            from .omniisaacgymenv.anymal_terrain import AnymalTerrainOmniTask
            from .omniisaacgymenv.ball_balance import BallBalanceOmniTask
            from .omniisaacgymenv.cartpole import CartpoleOmniTask
            from .omniisaacgymenv.crazyflie import CrazyflieOmniTask
            from .omniisaacgymenv.franka_cabinet import FrankaCabinetOmniTask
            from .omniisaacgymenv.humanoid import HumanoidOmniTask
            from .omniisaacgymenv.ingenuity import IngenuityOmniTask
            from .omniisaacgymenv.quadcopter import QuadcopterOmniTask
            from .omniisaacgymenv.shadow_hand import ShadowHandOmniTask
            from .omniisaacgymenv.factory.factory_task_nut_bolt_pick import FactoryNutBoltPickOmniTask
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
