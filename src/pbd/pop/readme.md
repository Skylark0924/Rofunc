# Bayesian Gaussian Mixture Model for Robotic Policy Imitation [\[pdf\]](https://arxiv.org/abs/1904.10716)

## Video

[![IMAGE ALT TEXT](http://img.youtube.com/vi/90tp3vwOiDE/0.jpg)](https://www.youtube.com/watch?v=90tp3vwOiDE "Video Title")

## Codes

### Installation

This requires v2.7 of Python.

Following these instructions install the library without copying the files, allowing to edit the sources files without reinstalling.

    git clone https://gitlab.idiap.ch/rli/pbdlib-python.git
    cd pbdlib-python
    pip install -e .

If pip is not install, you can get it that way:

    sudo apt-get install python-pip


### Run notebooks

Launch jupyter server with:

    cd pbdlib-python/pop
    jupyter notebook

Then click on desired notebook.

| Filename | Description |
|----------|-------------|
| Test Bayesian GMM - Policy - Time dependent LQR | - |
| Test Bayesian GMM - Position - Velocity Product | - |


### Run notebooks

Launch jupyter server with:

    cd pbdlib-python/pop
    jupyter notebook

Then click on desired notebook.

| Filename | Description |
|----------|-------------|
| Test Bayesian GMM - Policy - Time dependent LQR | - |
| Test Bayesian GMM - Position - Velocity Product | - |

### Recording demonstrations with the mouse

### Additional installation

    sudo apt-get intall python-tk

### Usage

    cd pbdlib-python/notebooks
    python record_demo.py -p /path_to_pbdlib/pop/data/tests -f filename

You can click on move the mouse on the left panel to record demonstrations. Press "h" for help. Save with "q".

