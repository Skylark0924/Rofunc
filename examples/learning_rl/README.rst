Reinforcement learning class
-----------------------------

The following are examples of reinforcement learning methods for robot learning.

.. tabs::


    .. tab:: OpenAIGym

        .. tabs::

            .. tab:: Pendulum

                .. code-block:: shell

                    # Training 
                    python examples/learning_rl/example_GymTasks_RofuncRL.py --task=Gym_Pendulum-v1 --agent=[ppo|a2c|td3|sac]

            .. tab:: CartPole

                .. code-block:: shell

                    # Training 
                    python examples/learning_rl/example_GymTasks_RofuncRL.py --task=Gym_CartPole-v1 --agent=[ppo|a2c|td3|sac]

            .. tab:: Acrobot

                .. code-block:: shell

                    # Training 
                    python examples/learning_rl/example_GymTasks_RofuncRL.py --task=Gym_Acrobot-v1 --agent=[ppo|a2c|td3|sac]                    

    .. tab:: IsaacGym

        .. tabs::

            .. tab:: Ant

                .. code-block:: shell

                    # Training 
                    python examples/learning_rl/example_Ant_RofuncRL.py --agent=[ppo|a2c|td3|sac]

                    # Inference with pre-trained model
                    python examples/learning_rl/example_Ant_RofuncRL.py --agent=ppo --inference   
                                 
            .. tab:: CURICabinet

                .. code-block:: shell

                    # Training
                    python examples/learning_rl/example_CURICabinet_RofuncRL.py --agent=ppo

                    # Inference with pre-trained model
                    python examples/learning_rl/example_CURICabinet_RofuncRL.py --agent=ppo --inference           

            .. tab:: D4RL 

                .. code-block:: shell

                    # Training
                    python examples/learning_rl/example_D4RL_RofuncRL.py --task=[Hopper|Walker2d|HalfCheetah|Reacher2d] --agent=dtrans

            .. tab:: FrankaCabinet

                .. code-block:: shell

                    # Training
                    python examples/learning_rl/example_FrankaCabinet_RofuncRL.py --agent=ppo

                    # Inference with pre-trained model
                    python examples/learning_rl/example_FrankaCabinet_RofuncRL.py --agent=ppo --inference         

            .. tab:: Humanoid

                .. code-block:: shell

                    # Training
                    python examples/learning_rl/example_Humanoid_RofuncRL.py --agent=ppo

                    # Inference with pre-trained model
                    python examples/learning_rl/example_Humanoid_RofuncRL.py --agent=ppo --inference      

            .. tab:: HumanoidAMP
                    
                .. code-block:: shell

                    # Training
                    python examples/learning_rl/example_HumanoidAMP_RofuncRL.py --task=[HumanoidAMP_backflip|HumanoidAMP_walk|HumanoidAMP_run|HumanoidAMP_dance|HumanoidAMP_hop] --agent=amp

                    # Inference with pre-trained model
                    python examples/learning_rl/example_HumanoidAMP_RofuncRL.py --task=[HumanoidAMP_backflip|HumanoidAMP_walk|HumanoidAMP_run|HumanoidAMP_dance|HumanoidAMP_hop] --agent=amp --inference

            .. tab:: HumanoidASE

                .. code-block:: shell

                    # Training
                    python examples/learning_rl/example_HumanoidASE_RofuncRL.py --task=[HumanoidASEGetupSwordShield|HumanoidASEPerturbSwordShield|HumanoidASEHeadingSwordShield|HumanoidASEReachSwordShield|HumanoidASELocationSwordShield|HumanoidASEStrikeSwordShield] --agent=ase

                    # Inference with pre-trained model
                    python examples/learning_rl/example_HumanoidASE_RofuncRL.py --task=[HumanoidASEGetupSwordShield|HumanoidASEPerturbSwordShield|HumanoidASEHeadingSwordShield|HumanoidASEReachSwordShield|HumanoidASELocationSwordShield|HumanoidASEStrikeSwordShield] --agent=ase --inference


    .. tab:: OmniIsaacGym
