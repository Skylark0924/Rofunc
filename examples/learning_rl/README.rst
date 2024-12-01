Reinforcement learning class
-----------------------------

The following are examples of reinforcement learning methods for robot learning.

.. tabs::

    .. tab:: IsaacGym

        .. tabs::

            .. tab:: Ant

                .. code-block:: shell

                    '''Training''' 
                    python examples/learning_rl/IsaacGym_RofuncRL/example_Ant_RofuncRL.py --agent=[ppo|a2c|td3|sac]

                    '''Inference with pre-trained model in model zoo'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_Ant_RofuncRL.py --agent=ppo --inference
                                 
            .. tab:: CURICabinet

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_CURICabinet_RofuncRL.py --agent=ppo

                    '''Inference with pre-trained model in model zoo'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_CURICabinet_RofuncRL.py --agent=ppo --inference

            .. tab:: CURIQbSoftHandSynergyGrasp

                .. code-block:: shell

                    # Available objects: Hammer, Spatula, Large_Clamp, Mug, Power_Drill, Knife, Scissors, Large_Marker, Phillips_Screw_Driver

                    '''Training'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_DexterousHands_RofuncRL.py --task=CURIQbSoftHandSynergyGrasp --agent=ppo --objects=Hammer

                    '''Inference with pre-trained model in model zoo'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_DexterousHands_RofuncRL.py --task=CURIQbSoftHandSynergyGrasp --agent=ppo --inference --objects=Hammer

            .. tab:: FrankaCabinet

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_FrankaCabinet_RofuncRL.py --agent=ppo

                    '''Inference with pre-trained model in model zoo'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_FrankaCabinet_RofuncRL.py --agent=ppo --inference

            .. tab:: FrankaCubeStack

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_FrankaCubeStack_RofuncRL.py --agent=ppo


            .. tab:: Humanoid

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_Humanoid_RofuncRL.py --agent=ppo

                    '''Inference with pre-trained model in model zoo'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_Humanoid_RofuncRL.py --agent=ppo --inference

            .. tab:: HumanoidAMP
                    
                .. code-block:: shell

                    '''Training'''
                    # Backflip
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_backflip --agent=amp
                    # Walk
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_walk --agent=amp
                    # Run
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_run --agent=amp
                    # Dance
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_dance --agent=amp
                    # Hop
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_hop --agent=amp

                    '''Inference with pre-trained model in model zoo'''
                    # Backflip
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_backflip --agent=amp --inference
                    # Walk
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_walk --agent=amp --inference
                    # Run
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_run --agent=amp --inference
                    # Dance
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_dance --agent=amp --inference
                    # Hop
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidAMP_RofuncRL.py --task=HumanoidAMP_hop --agent=amp --inference

            .. tab:: HumanoidASE

                .. code-block:: shell

                    '''Training'''
                    # Getup
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEGetupSwordShield --agent=ase
                    # Getup with perturbation
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEPerturbSwordShield --agent=ase
                    # Heading
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEHeadingSwordShield --agent=ase
                    # Reach
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEReachSwordShield --agent=ase
                    # Location
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASELocationSwordShield --agent=ase
                    # Strike
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEStrikeSwordShield --agent=ase

                    '''Inference with pre-trained model in model zoo'''
                    # Getup
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEGetupSwordShield --agent=ase --inference
                    # Getup with perturbation
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEPerturbSwordShield --agent=ase --inference
                    # Heading
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEHeadingSwordShield --agent=ase --inference
                    # Reach
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEReachSwordShield --agent=ase --inference
                    # Location
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASELocationSwordShield --agent=ase --inference
                    # Strike
                    python examples/learning_rl/IsaacGym_RofuncRL/example_HumanoidASE_RofuncRL.py --task=HumanoidASEStrikeSwordShield --agent=ase --inference

            .. tab:: DexterousHand

                .. code-block:: shell

                    '''Training'''
                    # Available tasks: BiShadowHandOver, BiShadowHandBlockStack, BiShadowHandBottleCap, BiShadowHandCatchAbreast,
                    #                  BiShadowHandCatchOver2Underarm, BiShadowHandCatchUnderarm, BiShadowHandDoorOpenInward,
                    #                  BiShadowHandDoorOpenOutward, BiShadowHandDoorCloseInward, BiShadowHandDoorCloseOutward,
                    #                  BiShadowHandGraspAndPlace, BiShadowHandLiftUnderarm, BiShadowHandPen, BiShadowHandPointCloud,
                    #                  BiShadowHandPushBlock, BiShadowHandReOrientation, BiShadowHandScissors, BiShadowHandSwingCup,
                    #                  BiShadowHandSwitch, BiShadowHandTwoCatchUnderarm
                    python examples/learning_rl/IsaacGym_RofuncRL/example_DexterousHands_RofuncRL.py --task=BiShadowHandOver --agent=ppo

                    '''Inference with pre-trained model in model zoo'''
                    python examples/learning_rl/IsaacGym_RofuncRL/example_DexterousHands_RofuncRL.py --task=BiShadowHandOver --agent=ppo --inference


        .. table:: Task Overview
            :widths: 20 35 35 10

            +------------------+-----------------------------+-------------------------------+-------------+
            | Tasks            | Animation                   | Performance                   | |ModelZoo|  |
            +==================+=============================+===============================+=============+
            | |Ant|            | |Ant-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |Cartpole|       |                             |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+    
            | |FrC|            | |FrC-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |FCS|            |                             |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |CUC|            | |CUC-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |CCI|            | |CCI-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+            
            | |CCB|            |                             |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |CSG|            | |CSG-gif1|                  |                               |    ✅       |
            |                  | |CSG-gif2|                  |                               |             |
            |                  | |CSG-gif3|                  |                               |             |
            |                  | |CSG-gif4|                  |                               |             |
            |                  | |CSG-gif5|                  |                               |             |
            |                  | |CSG-gif6|                  |                               |             |
            |                  | |CSG-gif7|                  |                               |             |
            |                  | |CSG-gif8|                  |                               |             |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |Hod|            | |Hod-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HAB|            | |HAB-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HAW|            |                             |                               |             |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HAR|            | |HAR-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HAD|            | |HAD-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HAH|            | |HAH-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HEG|            | |HEG-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HEP|            | |HEP-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HEH|            | |HEH-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HER|            |                             |                               |             |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HEL|            | |HEL-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |HES|            | |HES-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SBS|            | |SBS-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SBC|            | |SBC-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SCA|            | |SCA-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SU2|            | |SU2-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SCU|            | |SCU-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SOI|            | |SOI-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SOO|            | |SOO-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SCI|            | |SCI-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SCO|            | |SCO-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SGP|            | |SGP-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SLU|            | |SLU-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SHO|            | |SHO-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SPE|            | |SPE-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SPC|            |                             |                               |             |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SPB|            | |SPB-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SRO|            | |SRO-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SSC|            | |SSC-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SSW|            | |SSW-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |SWH|            | |SWH-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+
            | |STC|            | |STC-gif|                   |                               |    ✅       |
            +------------------+-----------------------------+-------------------------------+-------------+           



        .. |Ant-gif| image:: ../../img/task_gif/AntRofuncRLPPO.gif
        .. |FrC-gif| image:: ../../img/task_gif/FrankaCabinetRofuncRLPPO.gif
        .. |CUC-gif| image:: ../../img/task_gif/CURICabinetRofuncRLPPO.gif
        .. |CCI-gif| image:: ../../img/task_gif/CURICabinetRofuncRLPPO.gif
        .. |CSG-gif1| image:: ../../img/task_gif/CURIQbSoftHandSynergyGraspSpatulaRofuncRLPPO.gif
        .. |CSG-gif2| image:: ../../img/task_gif/CURIQbSoftHandSynergyGraspHammerRofuncRLPPO.gif
        .. |CSG-gif3| image:: ../../img/task_gif/CURIQbSoftHandSynergyGraspKnifeRofuncRLPPO.gif
        .. |CSG-gif4| image:: ../../img/task_gif/CURIQbSoftHandSynergyGraspLarge_clampRofuncRLPPO.gif
        .. |CSG-gif5| image:: ../../img/task_gif/CURIQbSoftHandSynergyGraspMugRofuncRLPPO.gif
        .. |CSG-gif6| image:: ../../img/task_gif/CURIQbSoftHandSynergyGraspPhillips_Screw_DriverRofuncRLPPO.gif
        .. |CSG-gif7| image:: ../../img/task_gif/CURIQbSoftHandSynergyGraspPower_drillRofuncRLPPO.gif
        .. |CSG-gif8| image:: ../../img/task_gif/CURIQbSoftHandSynergyGraspScissorsRofuncRLPPO.gif
        .. |Hod-gif| image:: ../../img/task_gif/HumanoidRofuncRLPPO.gif
        .. |HAB-gif| image:: ../../img/task_gif/HumanoidFlipRofuncRLAMP.gif
        .. |HAR-gif| image:: ../../img/task_gif/HumanoidRunRofuncRLAMP.gif
        .. |HAD-gif| image:: ../../img/task_gif/HumanoidDanceRofuncRLAMP.gif
        .. |HAH-gif| image:: ../../img/task_gif/HumanoidHopRofuncRLAMP.gif
        .. |HEG-gif| image:: ../../img/task_gif/HumanoidASEGetupSwordShieldRofuncRLASE.gif
        .. |HEP-gif| image:: ../../img/task_gif/HumanoidASEPerturbSwordShieldRofuncRLASE.gif
        .. |HEH-gif| image:: ../../img/task_gif/HumanoidASEHeadingSwordShieldRofuncRLASE.gif
        .. |HEL-gif| image:: ../../img/task_gif/HumanoidASELocationSwordShieldRofuncRLASE.gif
        .. |HES-gif| image:: ../../img/task_gif/HumanoidASEStrikeSwordShieldRofuncRLASE.gif
        .. |SBS-gif| image:: ../../img/task_gif/BiShadowHandBlockStackRofuncRLPPO.gif
        .. |SBC-gif| image:: ../../img/task_gif/BiShadowHandBottleCapRofuncRLPPO.gif
        .. |SCA-gif| image:: ../../img/task_gif/BiShadowHandCatchAbreastRofuncRLPPO.gif
        .. |SU2-gif| image:: ../../img/task_gif/BiShadowHandCatchOver2UnderarmRofuncRLPPO.gif
        .. |SCU-gif| image:: ../../img/task_gif/BiShadowHandCatchUnderarmRofuncRLPPO.gif
        .. |SOI-gif| image:: ../../img/task_gif/BiShadowHandDoorOpenInwardRofuncRLPPO.gif
        .. |SOO-gif| image:: ../../img/task_gif/BiShadowHandDoorOpenOutwardRofuncRLPPO.gif
        .. |SCI-gif| image:: ../../img/task_gif/BiShadowHandDoorCloseInwardRofuncRLPPO.gif
        .. |SCO-gif| image:: ../../img/task_gif/BiShadowHandDoorCloseOutwardRofuncRLPPO.gif
        .. |SGP-gif| image:: ../../img/task_gif/BiShadowHandGraspAndPlaceRofuncRLPPO.gif
        .. |SLU-gif| image:: ../../img/task_gif/BiShadowHandLiftUnderarmRofuncRLPPO.gif
        .. |SHO-gif| image:: ../../img/task_gif/BiShadowHandOverRofuncRLPPO.gif
        .. |SPE-gif| image:: ../../img/task_gif/BiShadowHandPenRofuncRLPPO.gif
        .. |SPB-gif| image:: ../../img/task_gif/BiShadowHandPushBlockRofuncRLPPO.gif
        .. |SRO-gif| image:: ../../img/task_gif/BiShadowHandReOrientationRofuncRLPPO.gif
        .. |SSC-gif| image:: ../../img/task_gif/BiShadowHandScissorsRofuncRLPPO.gif
        .. |SSW-gif| image:: ../../img/task_gif/BiShadowHandSwingCupRofuncRLPPO.gif
        .. |SWH-gif| image:: ../../img/task_gif/BiShadowHandSwitchRofuncRLPPO.gif
        .. |STC-gif| image:: ../../img/task_gif/BiShadowHandTwoCatchUnderarmRofuncRLPPO.gif

        .. |ModelZoo| replace:: `Model Zoo <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/config/learning/model_zoo.json>`__
        .. |Ant| replace:: `Ant <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ant.py>`__
        .. |Cartpole| replace:: `Cartpole <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/cartpole.py>`__
        .. |FrC| replace:: `FrankaCabinet <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/franka_cabinet.py>`__
        .. |FCS| replace:: `FrankaCubeStack <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/franka_cube_stack.py>`__
        .. |CUC| replace:: `CURICabinet <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/curi_cabinet.py>`__
        .. |CCI| replace:: `CURICabinet Image <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/curi_cabinet_image.py>`__
        .. |CCB| replace:: `CURICabinet Bimanual <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/curi_cabinet_bimanual.py>`__
        .. |CSG| replace:: `CURIQbSoftHand SynergyGrasp <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/grasp/curi_qbhand_synergy_grasp.py>`__
        .. |Hod| replace:: `Humanoid <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid.py>`__
        .. |HAB| replace:: `HumanoidAMP Backflip <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`__
        .. |HAW| replace:: `HumanoidAMP Walk <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`__
        .. |HAR| replace:: `HumanoidAMP Run <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`__
        .. |HAD| replace:: `HumanoidAMP Dance <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`__
        .. |HAH| replace:: `HumanoidAMP Hop <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`__
        .. |HEG| replace:: `HumanoidASE GetupSwordShield <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_amp_getup.py>`__
        .. |HEP| replace:: `HumanoidASE PerturbSwordShield <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_perturb.py>`__
        .. |HEH| replace:: `HumanoidASE HeadingSwordShield <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_heading.py>`__
        .. |HER| replace:: `HumanoidASE ReachSwordShield <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_reach.py>`__
        .. |HEL| replace:: `HumanoidASE LocationSwordShield <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_location.py>`__
        .. |HES| replace:: `HumanoidASE StrikeSwordShield <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_strike.py>`__
        .. |SBS| replace:: `BiShadowHand BlockStack <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_block_stack.py>`__
        .. |SBC| replace:: `BiShadowHand BottleCap <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_bottle_cap.py>`__
        .. |SCA| replace:: `BiShadowHand CatchAbreast <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_catch_abreast.py>`__
        .. |SU2| replace:: `BiShadowHand CatchOver2Underarm <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_catch_over2underarm.py>`__
        .. |SCU| replace:: `BiShadowHand CatchUnderarm <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_catch_underarm.py>`__
        .. |SOI| replace:: `BiShadowHand DoorOpenInward <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_door_open_inward.py>`__
        .. |SOO| replace:: `BiShadowHand DoorOpenOutward <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_door_open_outward.py>`__
        .. |SCI| replace:: `BiShadowHand DoorCloseInward <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_door_close_inward.py>`__
        .. |SCO| replace:: `BiShadowHand DoorCloseOutward <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_door_close_outward.py>`__
        .. |SGP| replace:: `BiShadowHand GraspAndPlace <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_grasp_and_place.py>`__
        .. |SLU| replace:: `BiShadowHand LiftUnderarm <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_lift_underarm.py>`__
        .. |SHO| replace:: `BiShadowHand Over <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_over.py>`__
        .. |SPE| replace:: `BiShadowHand Pen <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_pen.py>`__
        .. |SPC| replace:: `BiShadowHand PointCloud <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_point_cloud.py>`__
        .. |SPB| replace:: `BiShadowHand PushBlock <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_push_block.py>`__
        .. |SRO| replace:: `BiShadowHand ReOrientation <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_re_orientation.py>`__
        .. |SSC| replace:: `BiShadowHand Scissors <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ahandsse/shadow_hand_scissors.py>`__
        .. |SSW| replace:: `BiShadowHand SwingCup <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_swing_cup.py>`__
        .. |SWH| replace:: `BiShadowHand Switch <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_switch.py>`__
        .. |STC| replace:: `BiShadowHand TwoCatchUnderarm <https://github.com/Skylark0924/Rofunc/tree/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_two_catch_underarm.py>`__

    .. tab:: OmniIsaacGym

        .. tabs::

            .. tab:: AllegroHand

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_AllegroHandOmni_RofuncRL.py --agent=ppo

            .. tab:: Ant

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_AntOmni_RofuncRL.py --agent=ppo

            .. tab:: Anymal

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_AnymalOmni_RofuncRL.py --agent=ppo

            .. tab:: AnymalTerrain

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_AnymalTerrainOmni_RofuncRL.py --agent=ppo


            .. tab:: BallBalance

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_BallBalanceOmni_RofuncRL.py --agent=ppo

            .. tab:: Cartpole

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_CartpoleOmni_RofuncRL.py --agent=ppo

            .. tab:: Crazyflie

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_CrazyflieOmni_RofuncRL.py --agent=ppo

            .. tab:: FactoryNutBoltPick

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_FactoryNutBoltPickOmni_RofuncRL.py --agent=ppo


            .. tab:: FrankaCabinet

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_FrankaCabinetOmni_RofuncRL.py --agent=ppo

            .. tab:: Humanoid

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_HumanoidOmni_RofuncRL.py --agent=ppo

            .. tab:: Ingenuity

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_IngenuityOmni_RofuncRL.py --agent=ppo

            .. tab:: Quadcopter

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_QuadcopterOmni_RofuncRL.py --agent=ppo

            .. tab:: ShadowHand

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OmniIsaacGym_RofuncRL/example_ShadowHandOmni_RofuncRL.py --agent=ppo

    .. tab:: OpenAI Gym

        .. tabs::

            .. tab:: Pendulum

                .. code-block:: shell

                    '''Training'''
                    python examples/learning_rl/OpenAIGym_RofuncRL/example_GymTasks_RofuncRL.py --task=Gym_Pendulum-v1 --agent=[ppo|a2c|td3|sac]

            .. tab:: CartPole

                .. code-block:: shell

                    '''Training''' 
                    python examples/learning_rl/OpenAIGym_RofuncRL/example_GymTasks_RofuncRL.py --task=Gym_CartPole-v1 --agent=[ppo|a2c|td3|sac]

            .. tab:: Acrobot

                .. code-block:: shell

                    '''Training''' 
                    python examples/learning_rl/OpenAIGym_RofuncRL/example_GymTasks_RofuncRL.py --task=Gym_Acrobot-v1 --agent=[ppo|a2c|td3|sac]

    .. tab:: D4RL

        .. code-block:: shell

            '''Training'''
            # Hopper
            python examples/learning_rl/D4RL_Rofunc/example_D4RL_RofuncRL.py --task=Hopper --agent=dtrans
            # Walker2d
            python examples/learning_rl/D4RL_Rofunc/example_D4RL_RofuncRL.py --task=Walker2d --agent=dtrans
            # HalfCheetah
            python examples/learning_rl/D4RL_Rofunc/example_D4RL_RofuncRL.py --task=HalfCheetah --agent=dtrans
            # Reacher2d
            python examples/learning_rl/D4RL_Rofunc/example_D4RL_RofuncRL.py --task=Reacher2d --agent=dtrans
