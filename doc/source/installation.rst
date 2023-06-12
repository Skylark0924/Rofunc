Installation
==============================


Install from PyPI (stable version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Requirements need to be installed manually**

.. code-block:: shell

   # Install pbdlib
   pip install https://github.com/Skylark0924/Rofunc/releases/download/v0.0.0.7.1/pbdlib-0.1-py3-none-any.whl
   # Install Isaac Gym
   pip install https://github.com/Skylark0924/Rofunc/releases/download/v0.0.0.9/isaacgym-1.0rc4-py3-none-any.whl

Then

.. code-block:: shell

   pip install rofunc

   # [Option] Install with baseline RL frameworks (SKRL, RLlib, Stable Baselines3) and Envs (gymnasium[all], mujoco_py)
   pip install rofunc[baselines]

.. note::

   If you want to use functions related to ZED camera, you need to install `ZED SDK <https://www.stereolabs.com/developers/release/#downloads>`_ manually. (We have tried to package it as a :guilabel:`.whl` file to add it to :guilabel:`requirements.txt`, unfortunately, the ZED SDK is not very friendly and doesn't support direct installation.)


Install from Source (nightly version, recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. note::

   If you want to use functions related to ZED camera, you need to install `ZED SDK <https://www.stereolabs.com/developers/release/#downloads>`_ manually. (We have tried to package it as a :guilabel:`.whl` file to add it to :guilabel:`requirements.txt`, unfortunately, the ZED SDK is not very friendly and doesn't support direct installation.)









