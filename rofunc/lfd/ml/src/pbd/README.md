### Papers-related pages

If you came there reading one of the following works, please directly go to the related page.

- [Bayesian Gaussian mixture model for robotic policy imitation](https://gitlab.idiap.ch/rli/pbdlib-python/blob/master/pop/readme.md)
- [Variational Inference with Mixture Model Approximation for Applications in Robotics](https://gitlab.idiap.ch/rli/pbdlib-python/blob/master/vmm/readme.md)


# pbdlib

Pbdlib is a python library for robot programming by demonstration. The goal is to provide users with an easy to use library of algorithms that can be used to learn movement primitives using probabilistic tools.

This Python version of Pbdlib is maintained by Emmanuel Pignat at the Idiap Research Institute. Pbdlib is a collection of codes available in different languages and developed as a joint effort between the Idiap Research Institute and the Italian Institute of Technology (IIT). 

For more information see http://www.idiap.ch/software/pbdlib/.

## References

If you find these codes useful for your research, please acknowledge the authors by citing:

Pignat, E. and Calinon, S. (2017). [Learning adaptive dressing assistance from human demonstration](http://doi.org/10.1016/j.robot.2017.03.017). Robotics and Autonomous Systems 93, 61-75.


# Installation

This requires v2.7 of Python.

Following these instructions install the library without copying the files, allowing to edit the sources files without reinstalling.


    cd pbdlib-python
    pip install -e .

If pip is not install, you can get it that way:

    sudo apt-get install python-pip

## Launching the notebooks tutorial

Launch jupyter server with:

    jupyter notebook notebooks/

Then navigate through folders and click on desired notebook.

| Filename | Description |
|----------|-------------|
| pbdlib - basics.ipynb| Overview of the main functionalities of pbdlib.|
| pbdlib - lqr.ipynb| Linear quadratic regulator to regerate trajectories.|
| pbdlib - Multiple coordinate systems.ipynb| This example shows how motions can adapt to various positions and orientations of objects by projecting the demonstrations in several coordinate systems.|


### MEMMO related examples

Launch jupyter server with:

    jupyter notebook notebooks/MEMMO/


## User interface for recording data with the mouse

### Installation

    sudo apt-get intall python-tk

### Use

    cd notebooks
    python record_demo.py -p /path/to/folder -f filename

You can click on move the mouse on the left panel to record demonstrations. Press "h" for help. Save with "q".

To record demos with additional moving objects, run

    python record_demo.py -p /path/to/folder -f filename -m -c number_of_object

By pressing, the number corresponding to the desired object on your keyboard, you will make it appear and be able to move it.
Rotate them by holding the key of the number and turning the scrollwheel of your mouse.
