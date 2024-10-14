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
            # from .isaacgymenv.hotu.humanoid_hotu import HumanoidHOTUTask
            # from .isaacgymenv.hotu.humanoid_hotu_getup import HumanoidHOTUGetupTask
            # from .isaacgymenv.hotu.humanoid_hotu_perturb import HumanoidHOTUPerturbTask
            # from .isaacgymenv.hotu.humanoid_view_motion import HumanoidHOTUViewMotionTask
            # from .isaacgymenv.hotu.humanoid_hotu_heading import HumanoidHOTUHeadingTask
            # from .isaacgymenv.hotu.humanoid_hotu_location import HumanoidHOTULocationTask
            # from .isaacgymenv.hotu.humanoid_hotu_style import HumanoidHOTUStyleTask

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
                # "HumanoidHOTUGetup": HumanoidHOTUGetupTask,
                # "HumanoidHOTUPerturb": HumanoidHOTUPerturbTask,
                # "HumanoidHOTUViewMotion": HumanoidHOTUViewMotionTask,
                # "HumanoidHOTUHeading": HumanoidHOTUHeadingTask,
                # "HumanoidHOTULocation": HumanoidHOTULocationTask,
                # "HumanoidHOTUStyle": HumanoidHOTUStyleTask,

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
