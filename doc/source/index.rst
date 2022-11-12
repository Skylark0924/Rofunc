.. RoFunc documentation master file, created by
   sphinx-quickstart on Tue Sep 13 08:58:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Rofunc: The Full Process Python Package for Robot Learning from Demonstration
==================================


Getting Started
---------------

:doc:`overview`
   Show motivation, pipeline and list the available functions of :guilabel:`rofunc` package.
:doc:`quickstart`
   A bimanual dough rolling example that takes you through the whole process of robot learning from demonstration.
:doc:`auto_examples/index`
   A list of examples that demonstrate the usage of :guilabel:`rofunc` package.

Core Modules
------------

:doc:`devices/index`
   How to record, process, visual and export the multimodal demonstration data.
:doc:`lfd/index`
   Provide baseline methods that belong to various classes for learning from demonstration.
:doc:`planning/index`
   Provide robot planning and control methods.
:doc:`tools/index`
   Provide a basic visualization library `visualab`, a robotics library `robolab`, and tools like coordinate transform and a logger.
:doc:`simulator/index`
   Provide Isaac Gym based robot simulator.


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
