.. image:: ../img/logo7.png
   :alt: 

Rofunc: The Full Process Python Package for Robot Learning from Demonstration and Robot Manipulation
====================================================================================================

|image1|  |image2| |image3|  |image4| |image5| |image6| |image7|

   **Repository address:** https://github.com/Skylark0924/Rofunc

| Rofunc package focuses on the **Imitation Learning (IL), Reinforcement
  Learning (RL) and Learning from Demonstration (LfD)** for **(Humanoid) Robot Manipulation**. It provides valuable and convenient
  python functions, including *demonstration collection, data
  pre-processing, LfD algorithms, planning, and control methods*. We
  also provide an Isaac Gym-based robot simulator for evaluation. This package aims to advance the field by building a
  full-process toolkit and validation platform that simplifies and standardizes the process of demonstration data
  collection, processing, learning, and its deployment on robots.

.. image:: ../img/pipeline.png
   :alt: 


Available functions
--------------------

The available functions and plans can be found as follows.

   | **Note**
   | ✅: Achieved 🔃: Reformatting ⛔: TODO




Citation
--------

If you use rofunc in a scientific publication, we would appreciate
citations to the following paper:

.. code:: 

   @misc{Rofunc2022,
         author = {Liu, Junjia and Li, Chenzui and Delehelle, Donatien and Li, Zhihao and Chen, Fei},
         title = {Rofunc: The full process python package for robot learning from demonstration and robot manipulation},
         year = {2022},
         publisher = {GitHub},
         journal = {GitHub repository},
         howpublished = {\url{https://github.com/Skylark0924/Rofunc}},
   }

Related Papers
--------------

1. Robot cooking with stir-fry: Bimanual non-prehensile manipulation of
   semi-fluid objects (`IEEE RA-L
   2022 <https://arxiv.org/abs/2205.05960>`__ \|
   `Code <./rofunc/learning/dl/structured_transformer/strans.py>`__)

.. code:: 

   @article{liu2022robot,
            title={Robot cooking with stir-fry: Bimanual non-prehensile manipulation of semi-fluid objects},
            author={Liu, Junjia and Chen, Yiting and Dong, Zhipeng and Wang, Shixiong and Calinon, Sylvain and Li, Miao and Chen, Fei},
            journal={IEEE Robotics and Automation Letters},
            volume={7},
            number={2},
            pages={5159--5166},
            year={2022},
            publisher={IEEE}
   }

1. SoftGPT: Learn Goal-oriented Soft Object Manipulation Skills by
   Generative Pre-trained Heterogeneous Graph Transformer (IROS 2023)

2. Learning Robot Generalized Bimanual Coordination using Relative
   Parameterization Method on Human Demonstration (IEEE CDC 2023 \|
   `Code <./rofunc/learning/ml/tpgmm.py>`__)

The Team
--------

Rofunc is developed and maintained by the `CLOVER Lab (Collaborative and
Versatile Robots Laboratory) <https://feichenlab.com/>`__, CUHK.

Acknowledge
-----------

We would like to acknowledge the following projects:

Learning from Demonstration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. `pbdlib <https://gitlab.idiap.ch/rli/pbdlib-python>`__

2. `Ray RLlib <https://docs.ray.io/en/latest/rllib/index.html>`__

3. `ElegantRL <https://github.com/AI4Finance-Foundation/ElegantRL>`__

4. `SKRL <https://github.com/Toni-SM/skrl>`__

Planning and Control
~~~~~~~~~~~~~~~~~~~~

1. `Robotics codes from scratch
   (RCFS) <https://gitlab.idiap.ch/rli/robotics-codes-from-scratch>`__

.. |image1| image:: https://img.shields.io/github/v/release/Skylark0924/Rofunc
   :target: https://pypi.org/project/rofunc/
.. |image2| image:: https://img.shields.io/github/license/Skylark0924/Rofunc?color=blue
.. |image3| image:: https://img.shields.io/github/downloads/skylark0924/Rofunc/total
.. |image4| image:: https://img.shields.io/github/issues-closed-raw/Skylark0924/Rofunc?color=brightgreen
   :target: https://github.com/Skylark0924/Rofunc/issues?q=is%3Aissue+is%3Aclosed
.. |image5| image:: https://img.shields.io/github/issues-raw/Skylark0924/Rofunc?color=orange
   :target: https://github.com/Skylark0924/Rofunc/issues?q=is%3Aopen+is%3Aissue
.. |image6| image:: https://readthedocs.org/projects/rofunc/badge/?version=latest
   :target: https://rofunc.readthedocs.io/en/latest/?badge=latest
.. |image7| image:: https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FSkylark0924%2FRofunc%2Fbadge%3Fref%3Dmain&style=flat
   :target: https://actions-badge.atrox.dev/Skylark0924/Rofunc/goto?ref=main
.. |image8| image:: https://img.shields.io/badge/Documentation-Access-brightgreen?style=for-the-badge
   :target: https://rofunc.readthedocs.io/en/latest/
.. |image9| image:: https://img.shields.io/badge/Example Gallery-Access-brightgreen?style=for-the-badge
   :target: https://rofunc.readthedocs.io/en/latest/auto_examples/index.html
.. |image10| image:: https://api.star-history.com/svg?repos=Skylark0924/Rofunc&type=Date
   :target: https://star-history.com/#Skylark0924/Rofunc&Date
