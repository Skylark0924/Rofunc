Installation
==============================

.. attention::

   If you want to use the provided examples and dataset, you need to choose the **Nightly version**. The PyPI package only contains source codes.



.. tabs::

    .. tab:: Stable version (PyPI)

         .. tabs::

            .. tab:: IsaacGym 

               .. code-block:: shell

                  # Install rofunc
                  pip install rofunc
                  # [Option] Install with baseline RL frameworks (SKRL, RLlib, Stable Baselines3) and Envs (gymnasium[all], mujoco_py)
                  pip install rofunc[baselines]

                  # [Required] Install pbdlib and IsaacGym
                  pip install https://github.com/Skylark0924/Rofunc/releases/download/v0.0.2.3/pbdlib-0.1-py3-none-any.whl
                  pip install https://github.com/Skylark0924/Rofunc/releases/download/v0.0.0.9/isaacgym-1.0rc4-py3-none-any.whl

            .. tab:: OmniIsaacGym

               :guilabel:`Isaac Sim` has to be installed firstly by following this `documentation <https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html>`_. Note that the :guilabel:`Isaac Sim` version should be :guilabel:`2022.2.1`, :guilabel:`2023.1.0` is not supported yet since its default python version is `3.10` which is not compatible with :guilabel:`rofunc`.

               Find the :guilabel:`Isaac Sim` installation path (the default path should be :guilabel:`/home/[user_name]/.local/share/ov/pkg/isaac_sim-2022.2.1`), and run the following command to set up :guilabel:`OmniIsaacGym`.

               .. code-block:: shell

                  # Alias the Isaac Sim python in .bashrc
                  gedit ~/.bashrc 
                  # Add the following line to the end of the file
                  alias pysim2="/home/[user_name]/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh"
                  # Save and close the file
                  source ~/.bashrc

                  # Install rofunc
                  pysim2 -m pip install rofunc
                  # [Option] Install with baseline RL frameworks (SKRL, RLlib, Stable Baselines3) and Envs (gymnasium[all], mujoco_py)
                  pysim2 -m pip install rofunc[baselines]

                  # [Required] Install pbdlib
                  pysim2 -m pip install https://github.com/Skylark0924/Rofunc/releases/download/v0.0.2.3/pbdlib-0.1-py3-none-any.whl



    .. tab:: Nightly version (recommended)

         .. tabs::

            .. tab:: IsaacGym 
           
               .. code-block:: shell

                  git clone https://github.com/Skylark0924/Rofunc.git
                  cd Rofunc

                  # Create a conda environment
                  # Python 3.8 is strongly recommended
                  conda create -n rofunc python=3.8

                  # For Linux user
                  sh ./scripts/install.sh
                  # [Option] Install with baseline RL frameworks (SKRL, RLlib, Stable Baselines3)
                  sh ./scripts/install_w_baselines.sh
                  # [Option] For MacOS user (brew is required, Isaac Gym based simulator is not supported on MacOS)
                  sh ./scripts/mac_install.sh

            .. tab:: OmniIsaacGym

               :guilabel:`Isaac Sim` has to be installed firstly by following this `documentation <https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html>`_. Note that the :guilabel:`Isaac Sim` version should be :guilabel:`2022.2.1`, :guilabel:`2023.1.0` is not supported yet since its default python version is `3.10` which is not compatible with :guilabel:`rofunc`.

               Find the :guilabel:`Isaac Sim` installation path (the default path should be :guilabel:`/home/[user_name]/.local/share/ov/pkg/isaac_sim-2022.2.1`), and run the following command to set up :guilabel:`OmniIsaacGym`.

               .. code-block:: shell

                  # Alias the Isaac Sim python in .bashrc
                  gedit ~/.bashrc 
                  # Add the following line to the end of the file
                  alias pysim2="/home/[user_name]/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh"
                  # Save and close the file
                  source ~/.bashrc

                  git clone https://github.com/Skylark0924/Rofunc.git
                  cd Rofunc

                  # Install rofunc
                  pysim2 -m pip install .   


.. note::

   If you want to use functions related to ZED camera, you need to install `ZED SDK <https://www.stereolabs.com/developers/release/#downloads>`_ manually. (We have tried to package it as a :guilabel:`.whl` file to add it to :guilabel:`requirements.txt`, unfortunately, the ZED SDK is not very friendly and doesn't support direct installation.)




