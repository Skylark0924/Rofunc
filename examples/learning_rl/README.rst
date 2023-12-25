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

        .. list-table:: Task Overview
           :widths: 25 40 40 10
           :header-rows: 1

           * - Tasks
             - Animation
             - Performance
             - ModelZoo
           * - `Ant <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ant.py>`_
             - 
             - 
             -
           * - `Cartpole <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/cartpole.py>`_
             - 
             -
             - 
           * - `FrankaCabinet <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/franka_cabinet.py>`_
             - 
             - 
             -
           * - `FrankaCubeStack <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/franka_cube_stack.py>`_
             - 
             - 
             -
           * - `CURICabinet <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/curi_cabinet.py>`_
             - .. image:: ../../doc/img/CURICabinet.gif
             - 
             -
           * - `CURICabinetImage <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/curi_cabinet_image.py>`_
             - .. image:: ../../doc/img/CURICabinet.gif
             - 
             -
           * - `CURICabinetBimanual <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/curi_cabinet_bimanual.py>`_
             - .. image:: ../../doc/img/CURICabinet.gif
             - 
             -
           * - `Humanoid <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid.py>`_
             - 
             -
             - 
           * - `HumanoidAMP_backflip <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`_
             - .. image:: ../../doc/img/RofuncAMP_HumanoidFlip.gif
             - 
             -
           * - `HumanoidAMP_walk <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`_
             - 
             -
             - 
           * - `HumanoidAMP_run <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`_
             - .. image:: ../../doc/img/RofuncAMP_HumanoidRun.gif
             - 
             -
           * - `HumanoidAMP_dance <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`_
             - .. image:: ../../doc/img/RofuncAMP_HumanoidDance.gif
             -
             -
           * - `HumanoidAMP_hop <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/humanoid_amp.py>`_
             - .. image:: ../../doc/img/RofuncAMP_HumanoidHop.gif
             -
             -
           * - `HumanoidASEGetupSwordShield <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_amp_getup.py>`_
             - .. image:: ../../doc/img/ASE3.gif
             -
             -
           * - `HumanoidASEPerturbSwordShield <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_perturb.py>`_
             - .. image:: ../../doc/img/ASE1.gif
             -
             -
           * - `HumanoidASEHeadingSwordShield <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_heading.py>`_
             - .. image:: ../../doc/img/ASE5.gif
             -
             -
           * - `HumanoidASELocationSwordShield <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_location.py>`_
             - 
             -
             -
           * - `HumanoidASEReachSwordShield <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_reach.py>`_
             - 
             -
             -
           * - `HumanoidASEStrikeSwordShield <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ase/humanoid_strike.py>`_
             - .. image:: ../../doc/img/ASE4.gif
             - 
             -
           * - `BiShadowHandBlockStack <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_block_stack.py>`_
             - 
             - 
             -
           * - `BiShadowHandBottleCap <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_bottle_cap.py>`_
             - 
             -
             - 
           * - `BiShadowHandCatchAbreast <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_catch_abreast.py>`_
             - 
             -
             - 
           * - `BiShadowHandCatchOver2Underarm <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_catch_over2underarm.py>`_
             - 
             -
             - 
           * - `BiShadowHandCatchUnderarm <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_catch_underarm.py>`_
             - 
             -
             - 
           * - `BiShadowHandDoorOpenInward <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_door_open_inward.py>`_
             - 
             -
             - 
           * - `BiShadowHandDoorOpenOutward <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_door_open_outward.py>`_
             - 
             - 
             -
           * - `BiShadowHandDoorCloseInward <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_door_close_inward.py>`_
             - 
             - 
             -
           * - `BiShadowHandDoorCloseOutward <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_door_close_outward.py>`_
             - 
             -
             - 
           * - `BiShadowHandGraspAndPlace <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_grasp_and_place.py>`_
             - 
             -
             - 
           * - `BiShadowHandLiftUnderarm <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_lift_underarm.py>`_
             - 
             - 
             -
           * - `BiShadowHandOver <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_over.py>`_
             - 
             - 
             -
           * - `BiShadowHandPen <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_pen.py>`_
             - 
             -
             - 
           * - `BiShadowHandPointCloud <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_point_cloud.py>`_
             - 
             -
             - 
           * - `BiShadowHandPushBlock <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_push_block.py>`_
             - 
             - 
             -
           * - `BiShadowHandReOrientation <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_re_orientation.py>`_
             - 
             -
             -
           * - `BiShadowHandScissors <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/ahandsse/shadow_hand_scissors.py>`_
             - 
             -
             -
           * - `BiShadowHandSwingCup <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_swing_cup.py>`_
             - 
             -
             -
           * - `BiShadowHandSwitch <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_switch.py>`_
             - 
             -
             -
           * - `BiShadowHandTwoCatchUnderarm <https://github.com/Skylark0924/Rofunc/blob/main/rofunc/learning/RofuncRL/tasks/isaacgymenv/hands/shadow_hand_two_catch_underarm.py>`_
             - 
             -
             -







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
