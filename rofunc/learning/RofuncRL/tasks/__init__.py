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
            from .isaacgymenv.physhoi.humanoid_physhoi import HumanoidPhysHOITask
            # from .isaacgymenv.physhoi.physhoi import PhysHOI_BallPlay
            from .isaacgymenv.hotu.humanoid_hotu import HumanoidHOTUTask
            from .isaacgymenv.hotu.humanoid_view_motion import HumanoidHOTUViewMotionTask
            from .isaacgymenv.hands.shadow_hand_block_stack import ShadowHandBlockStackTask
            from .isaacgymenv.hands.shadow_hand_bottle_cap import ShadowHandBottleCapTask
            from .isaacgymenv.hands.shadow_hand_catch_abreast import ShadowHandCatchAbreastTask
            from .isaacgymenv.hands.shadow_hand_catch_over2underarm import ShadowHandCatchOver2UnderarmTask
            from .isaacgymenv.hands.shadow_hand_catch_underarm import ShadowHandCatchUnderarmTask
            from .isaacgymenv.hands.shadow_hand_door_open_inward import ShadowHandDoorOpenInwardTask
            from .isaacgymenv.hands.shadow_hand_door_open_outward import ShadowHandDoorOpenOutwardTask
            from .isaacgymenv.hands.shadow_hand_door_close_inward import ShadowHandDoorCloseInwardTask
            from .isaacgymenv.hands.shadow_hand_door_close_outward import ShadowHandDoorCloseOutwardTask
            from .isaacgymenv.hands.shadow_hand_grasp_and_place import ShadowHandGraspAndPlaceTask
            from .isaacgymenv.hands.shadow_hand_lift_underarm import ShadowHandLiftUnderarmTask
            from .isaacgymenv.hands.shadow_hand_over import ShadowHandOverTask
            from .isaacgymenv.hands.shadow_hand_pen import ShadowHandPenTask
            from .isaacgymenv.hands.shadow_hand_point_cloud import ShadowHandPointCloudTask
            from .isaacgymenv.hands.shadow_hand_push_block import ShadowHandPushBlockTask
            from .isaacgymenv.hands.shadow_hand_re_orientation import ShadowHandReOrientationTask
            from .isaacgymenv.hands.shadow_hand_scissors import ShadowHandScissorsTask
            from .isaacgymenv.hands.shadow_hand_swing_cup import ShadowHandSwingCupTask
            from .isaacgymenv.hands.shadow_hand_switch import ShadowHandSwitchTask
            from .isaacgymenv.hands.shadow_hand_two_catch_underarm import ShadowHandTwoCatchUnderarmTask
            from .isaacgymenv.hands.qbsofthand_grasp import QbSoftHandGraspTask
            from .isaacgymenv.hands.bi_qbhand_grasp_and_place import BiQbSoftHandGraspAndPlaceTask
            from .isaacgymenv.hands.bi_qbhand_synergy_grasp import BiQbSoftHandSynergyGraspTask
            from .isaacgymenv.hands.qbhand_synergy_grasp import QbSoftHandSynergyGraspTask
            from .isaacgymenv.hands.shadow_hand_grasp import ShadowHandGraspTask
            from .isaacgymenv.grasp.curi_qbhand_synergy_grasp import CURIQbSoftHandSynergyGraspTask

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
                "HumanoidPhysHOI": HumanoidPhysHOITask,
                # "HumanoidPhysHOI": PhysHOI_BallPlay,
                "HumanoidHOTU": HumanoidHOTUTask,
                "HumanoidHOTUViewMotion": HumanoidHOTUViewMotionTask,
                "BiShadowHandOver": ShadowHandOverTask,
                "BiShadowHandBlockStack": ShadowHandBlockStackTask,
                "BiShadowHandBottleCap": ShadowHandBottleCapTask,
                "BiShadowHandCatchAbreast": ShadowHandCatchAbreastTask,
                "BiShadowHandCatchOver2Underarm": ShadowHandCatchOver2UnderarmTask,
                "BiShadowHandCatchUnderarm": ShadowHandCatchUnderarmTask,
                "BiShadowHandDoorOpenInward": ShadowHandDoorOpenInwardTask,
                "BiShadowHandDoorOpenOutward": ShadowHandDoorOpenOutwardTask,
                "BiShadowHandDoorCloseInward": ShadowHandDoorCloseInwardTask,
                "BiShadowHandDoorCloseOutward": ShadowHandDoorCloseOutwardTask,
                "BiShadowHandGraspAndPlace": ShadowHandGraspAndPlaceTask,
                "BiShadowHandLiftUnderarm": ShadowHandLiftUnderarmTask,
                "BiShadowHandPen": ShadowHandPenTask,
                "BiShadowHandPointCloud": ShadowHandPointCloudTask,
                "BiShadowHandPushBlock": ShadowHandPushBlockTask,
                "BiShadowHandReOrientation": ShadowHandReOrientationTask,
                "BiShadowHandScissors": ShadowHandScissorsTask,
                "BiShadowHandSwingCup": ShadowHandSwingCupTask,
                "BiShadowHandSwitch": ShadowHandSwitchTask,
                "BiShadowHandTwoCatchUnderarm": ShadowHandTwoCatchUnderarmTask,
                "QbSoftHandGrasp": QbSoftHandGraspTask,
                "BiQbSoftHandGraspAndPlace": BiQbSoftHandGraspAndPlaceTask,
                "BiQbSoftHandSynergyGrasp": BiQbSoftHandSynergyGraspTask,
                "QbSoftHandSynergyGrasp": QbSoftHandSynergyGraspTask,
                "ShadowHandGrasp": ShadowHandGraspTask,
                "CURIQbSoftHandSynergyGrasp": CURIQbSoftHandSynergyGraspTask,
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
            from .omniisaacgym.aubo_cube import AuboCubeOmniTask
            from .omniisaacgym.elfin_bag import ElfinBagOmniTask
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
                "AuboCubeOmni": AuboCubeOmniTask,
                "ElfinBagOmni": ElfinBagOmniTask,
            }
