# RCFS - Setup Python Environment

As the examples presented in RCFS are made from scratch. You only need a few dependencies to run the python script without any problems:

* Python 3.X (tested with Python 3.9)
* Numpy, as Linear Algebra library
* Scipy, as a scientific library
* Matplotlib, to generate the plots

Nowadays, Python is installed by default on every operating system worthy of the name (even on Windows 10). If for a reason or another it's not the case on your machine, you can take a look on this [page](https://www.python.org/downloads/) where installer for Windows and Mac OS are given.

To check if python is installed you can run the following command in a terminal:

```bash
$ python --version
```

It should output you the version number of your python installation (beware that the returned version should be greater than 3.0.0). If the command is returning you an error or a 2.X.Y version, you should also consider to try with:

```bash
$ python3 --version
```

Depending how your OS is installed, you may need to explicitly specify that you want to use version 3.

To install the two packages (numpy and matplotlib) you can follow one of the two solutions presented below:

## Conda based solution

In this solution, we assume that you already have miniconda installed on your machine. If not you can download it [here](https://docs.conda.io/en/latest/miniconda.html) (be careful to choose the one corresponding to your operating system). If you have troubles installing miniconda, [this](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) could be useful. 

Conda is an open source package management system and environment management system that runs on Windows, macOs, and Linux. It allows you to create multiple environments that are independents from each others. Such that if you break one of your environment, the others will not be impacted. More than the environment management system, it also allow you to manage packages (similarly to pip), with the advantages to not only deal with python packages. 

1. Go inside the RCFS repository

```bash
$ cd <path_to_rcfs>
```

2. Go inside the python directory

```bash
$ cd python
```

3. Install the conda environment

```bash
$ conda env create --file=setup_conda.yml
```

4. Activate the conda environment (do not forget to do this each time you want to use python scripts of RCFS)

```bash
$ conda activate rcfs
```

5. Enjoy

## pip based solution

Beware that if only ``python3 --version`` is working and you choose to use the pip based installation, you will need to replace ``pip`` by ``pip3``.

In this solution we assume that you have a working python installation with pip. Pip is a package manager for Python.

1. Install the needed packages

```bash
$ pip install numpy matplotlib scipy
```

2. Enjoy
