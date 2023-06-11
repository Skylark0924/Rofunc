.. Rofunc documentation master file, created by
   sphinx-quickstart on Tue Sep 13 08:58:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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

The available functions and future plans can be found as follows.

   | **Note**
   | ✅: Achieved 🔃: Reformatting ⛔: TODO


+----------+----+----------+----+----------+----+----------+----+----------+----+
| Data     |    | Learning |    | P&C      |    | Tools    |    | S        |    |
|          |    |          |    |          |    |          |    | imulator |    |
+==========+====+==========+====+==========+====+==========+====+==========+====+
| `        | ✅ | ``DMP``  | ⛔ | ```LQT`  | ✅ | ``       | ✅ | ```Fra   | ✅ |
| ``xsens. |    |          |    | ` <https |    | Config`` |    | nka`` <h |    |
| record`` |    |          |    | ://rofun |    |          |    | ttps://r |    |
|  <https: |    |          |    | c.readth |    |          |    | ofunc.re |    |
| //rofunc |    |          |    | edocs.io |    |          |    | adthedoc |    |
| .readthe |    |          |    | /en/late |    |          |    | s.io/en/ |    |
| docs.io/ |    |          |    | st/plann |    |          |    | latest/s |    |
| en/lates |    |          |    | ing/lqt. |    |          |    | imulator |    |
| t/device |    |          |    | html>`__ |    |          |    | /franka. |    |
| s/xsens. |    |          |    |          |    |          |    | html>`__ |    |
| html>`__ |    |          |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| `        | ✅ | ``GMR``  | ✅ |``LQTBi`` | ✅ | `        | ✅ | ``       | ✅ |
| ``xsens. |    |          |    |          |    | `robolab |    | `CURI``  |    |
| export`` |    |          |    |          |    | .coord`` |    | <https:/ |    |
|  <https: |    |          |    |          |    |          |    | /rofunc. |    |
| //rofunc |    |          |    |          |    |          |    | readthed |    |
| .readthe |    |          |    |          |    |          |    | ocs.io/e |    |
| docs.io/ |    |          |    |          |    |          |    | n/latest |    |
| en/lates |    |          |    |          |    |          |    | /simulat |    |
| t/device |    |          |    |          |    |          |    | or/curi. |    |
| s/xsens. |    |          |    |          |    |          |    | html>`__ |    |
| html>`__ |    |          |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| `        | ✅ | `        | ✅ | ```L     | ✅ | ``robo   | ✅ | ``CU     | 🔃 |
| ``xsens. |    | `TPGMM`` |    | QTFb`` < |    | lab.fk`` |    | RIMini`` |    |
| visual`` |    |          |    | https:// |    |          |    |          |    |
|  <https: |    |          |    | rofunc.r |    |          |    |          |    |
| //rofunc |    |          |    | eadthedo |    |          |    |          |    |
| .readthe |    |          |    | cs.io/en |    |          |    |          |    |
| docs.io/ |    |          |    | /latest/ |    |          |    |          |    |
| en/lates |    |          |    | planning |    |          |    |          |    |
| t/device |    |          |    | /lqt_fb. |    |          |    |          |    |
| s/xsens. |    |          |    | html>`__ |    |          |    |          |    |
| html>`__ |    |          |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| ```o     | ✅ | ``T      | ✅ | ```L     | ✅ | ``robo   | ✅ | ``CURISo | ✅ |
| pti.reco |    | PGMMBi`` |    | QTCP`` < |    | lab.ik`` |    | ftHand`` |    |
| rd`` <ht |    |          |    | https:// |    |          |    |          |    |
| tps://ro |    |          |    | rofunc.r |    |          |    |          |    |
| func.rea |    |          |    | eadthedo |    |          |    |          |    |
| dthedocs |    |          |    | cs.io/en |    |          |    |          |    |
| .io/en/l |    |          |    | /latest/ |    |          |    |          |    |
| atest/de |    |          |    | planning |    |          |    |          |    |
| vices/op |    |          |    | /lqt_cp. |    |          |    |          |    |
| titrack. |    |          |    | html>`__ |    |          |    |          |    |
| html>`__ |    |          |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| ```o     | ✅ | ``TPGMM  | ✅ | ``LQ     | ✅ | ``robo   | ⛔ | ``       | ✅ |
| pti.expo |    | _RPCtl`` |    | TCPDMP`` |    | lab.fd`` |    | Walker`` |    |
| rt`` <ht |    |          |    |          |    |          |    |          |    |
| tps://ro |    |          |    |          |    |          |    |          |    |
| func.rea |    |          |    |          |    |          |    |          |    |
| dthedocs |    |          |    |          |    |          |    |          |    |
| .io/en/l |    |          |    |          |    |          |    |          |    |
| atest/de |    |          |    |          |    |          |    |          |    |
| vices/op |    |          |    |          |    |          |    |          |    |
| titrack. |    |          |    |          |    |          |    |          |    |
| html>`__ |    |          |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| ```o     | ✅ | ``TPGMM_ | ✅ | ``LQR``  | ✅ | ``robo   | ⛔ | `        | 🔃 |
| pti.visu |    | RPRepr`` |    |          |    | lab.id`` |    | `Gluon`` |    |
| al`` <ht |    |          |    |          |    |          |    |          |    |
| tps://ro |    |          |    |          |    |          |    |          |    |
| func.rea |    |          |    |          |    |          |    |          |    |
| dthedocs |    |          |    |          |    |          |    |          |    |
| .io/en/l |    |          |    |          |    |          |    |          |    |
| atest/de |    |          |    |          |    |          |    |          |    |
| vices/op |    |          |    |          |    |          |    |          |    |
| titrack. |    |          |    |          |    |          |    |          |    |
| html>`__ |    |          |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| ```ze    | ✅ | `        | ✅ | ``Po     | ✅ | `        | ✅ | ``       | 🔃 |
| d.record |    | `TPGMR`` |    | GLQRBi`` |    | `visuala |    | Baxter`` |    |
| `` <http |    |          |    |          |    | b.dist`` |    |          |    |
| s://rofu |    |          |    |          |    |          |    |          |    |
| nc.readt |    |          |    |          |    |          |    |          |    |
| hedocs.i |    |          |    |          |    |          |    |          |    |
| o/en/lat |    |          |    |          |    |          |    |          |    |
| est/devi |    |          |    |          |    |          |    |          |    |
| ces/zed. |    |          |    |          |    |          |    |          |    |
| html>`__ |    |          |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| ```ze    | ✅ | ``T      | ✅ | `        | 🔃 | ``       | ✅ | ``       | 🔃 |
| d.export |    | PGMRBi`` |    | ``iLQR`` |    | visualab |    | Sawyer`` |    |
| `` <http |    |          |    |  <https: |    | .ellip`` |    |          |    |
| s://rofu |    |          |    | //rofunc |    |          |    |          |    |
| nc.readt |    |          |    | .readthe |    |          |    |          |    |
| hedocs.i |    |          |    | docs.io/ |    |          |    |          |    |
| o/en/lat |    |          |    | en/lates |    |          |    |          |    |
| est/devi |    |          |    | t/planni |    |          |    |          |    |
| ces/zed. |    |          |    | ng/ilqr. |    |          |    |          |    |
| html>`__ |    |          |    | html>`__ |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| ```ze    | ✅ | ``       | ✅ | ``       | 🔃 | `        | ✅ | ``Multi  | ✅ |
| d.visual |    | TPHSMM`` |    | iLQRBi`` |    | `visuala |    | -Robot`` |    |
| `` <http |    |          |    |          |    | b.traj`` |    |          |    |
| s://rofu |    |          |    |          |    |          |    |          |    |
| nc.readt |    |          |    |          |    |          |    |          |    |
| hedocs.i |    |          |    |          |    |          |    |          |    |
| o/en/lat |    |          |    |          |    |          |    |          |    |
| est/devi |    |          |    |          |    |          |    |          |    |
| ces/zed. |    |          |    |          |    |          |    |          |    |
| html>`__ |    |          |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| ``emg.   | ✅ | `        | 🔃 | ``       | 🔃 |          |    |          |    |
| record`` |    | `BCO(Rof |    | iLQRFb`` |    |          |    |          |    |
|          |    | uncIL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| ``emg.   | ✅ | ``       | ⛔ | ``       | 🔃 |          |    |          |    |
| export`` |    | BC-Z(Rof |    | iLQRCP`` |    |          |    |          |    |
|          |    | uncIL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| ``emg.   | ✅ | ``ST     | ⛔ | ``iL     | 🔃 |          |    |          |    |
| visual`` |    | rans(Rof |    | QRDyna`` |    |          |    |          |    |
|          |    | uncIL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| `        | ⛔ | ``       | ⛔ | ``i      | 🔃 |          |    |          |    |
| `mmodal. |    | RT-1(Rof |    | LQRObs`` |    |          |    |          |    |
| record`` |    | uncIL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
| `        | ✅ | ``PPO    | ✅ | ``MPC``  | ⛔ |          |    |          |    |
| `mmodal. |    | (SKRL)`` |    |          |    |          |    |          |    |
| export`` |    |          |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``SAC    | ✅ | ``RMP``  | ⛔ |          |    |          |    |
|          |    | (SKRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``TD3    | ✅ |          |    |          |    |          |    |
|          |    | (SKRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``PP     | ⛔ |          |    |          |    |          |    |
|          |    | O(SB3)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``SA     | ⛔ |          |    |          |    |          |    |
|          |    | C(SB3)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``TD     | ⛔ |          |    |          |    |          |    |
|          |    | 3(SB3)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``PPO(   | ✅ |          |    |          |    |          |    |
|          |    | RLlib)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``SAC(   | ✅ |          |    |          |    |          |    |
|          |    | RLlib)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``TD3(   | ✅ |          |    |          |    |          |    |
|          |    | RLlib)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``PPO(E  | ✅ |          |    |          |    |          |    |
|          |    | legRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``SAC(E  | ✅ |          |    |          |    |          |    |
|          |    | legRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``TD3(E  | ✅ |          |    |          |    |          |    |
|          |    | legRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ```PP    | ✅ |          |    |          |    |          |    |
|          |    | O(Rofunc |    |          |    |          |    |          |    |
|          |    | RL)`` <h |    |          |    |          |    |          |    |
|          |    | ttps://r |    |          |    |          |    |          |    |
|          |    | ofunc.re |    |          |    |          |    |          |    |
|          |    | adthedoc |    |          |    |          |    |          |    |
|          |    | s.io/en/ |    |          |    |          |    |          |    |
|          |    | latest/l |    |          |    |          |    |          |    |
|          |    | fd/Rofun |    |          |    |          |    |          |    |
|          |    | cRL/PPO. |    |          |    |          |    |          |    |
|          |    | html>`__ |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ```SA    | 🔃 |          |    |          |    |          |    |
|          |    | C(Rofunc |    |          |    |          |    |          |    |
|          |    | RL)`` <h |    |          |    |          |    |          |    |
|          |    | ttps://r |    |          |    |          |    |          |    |
|          |    | ofunc.re |    |          |    |          |    |          |    |
|          |    | adthedoc |    |          |    |          |    |          |    |
|          |    | s.io/en/ |    |          |    |          |    |          |    |
|          |    | latest/l |    |          |    |          |    |          |    |
|          |    | fd/Rofun |    |          |    |          |    |          |    |
|          |    | cRL/SAC. |    |          |    |          |    |          |    |
|          |    | html>`__ |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ```TD    | 🔃 |          |    |          |    |          |    |
|          |    | 3(Rofunc |    |          |    |          |    |          |    |
|          |    | RL)`` <h |    |          |    |          |    |          |    |
|          |    | ttps://r |    |          |    |          |    |          |    |
|          |    | ofunc.re |    |          |    |          |    |          |    |
|          |    | adthedoc |    |          |    |          |    |          |    |
|          |    | s.io/en/ |    |          |    |          |    |          |    |
|          |    | latest/l |    |          |    |          |    |          |    |
|          |    | fd/Rofun |    |          |    |          |    |          |    |
|          |    | cRL/TD3. |    |          |    |          |    |          |    |
|          |    | html>`__ |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``ODT    | ⛔ |          |    |          |    |          |    |
|          |    | rans(Rof |    |          |    |          |    |          |    |
|          |    | uncRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``       | ⛔ |          |    |          |    |          |    |
|          |    | RT-1(Rof |    |          |    |          |    |          |    |
|          |    | uncRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | `        | ⛔ |          |    |          |    |          |    |
|          |    | `CQL(Rof |    |          |    |          |    |          |    |
|          |    | uncRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``T      | ⛔ |          |    |          |    |          |    |
|          |    | D3BC(Rof |    |          |    |          |    |          |    |
|          |    | uncRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``DT     | 🔃 |          |    |          |    |          |    |
|          |    | rans(Rof |    |          |    |          |    |          |    |
|          |    | uncRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+
|          |    | ``       | ⛔ |          |    |          |    |          |    |
|          |    | EDAC(Rof |    |          |    |          |    |          |    |
|          |    | uncRL)`` |    |          |    |          |    |          |    |
+----------+----+----------+----+----------+----+----------+----+----------+----+


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



.. Hidden TOCs

.. toctree::
   :maxdepth: 3
   :caption: Get Started
   :hidden:
   :glob:

   overview
   quickstart
   auto_examples/index

.. toctree::
   :maxdepth: 3
   :caption: Core Modules
   :hidden:
   :glob:

   devices/index
   lfd/index
   planning/index
   tools/index
   simulator/index

.. .. toctree::
..    :maxdepth: 1
..    :hidden:

..    releaselog

Indices and tables
-------------------  

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


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
