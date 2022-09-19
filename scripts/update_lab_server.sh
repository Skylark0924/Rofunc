#!/bin/bash -i
shopt -s expand_aliases
alias pip3a="/home/skylark/anaconda3/bin/pip"
alias pyana3="/home/skylark/anaconda3/bin/python3.9"

cd ../dist
pip3a uninstall -y rofunc
cd ..
pyana3 setup.py bdist_wheel sdist
cd ./dist
pip3a install rofunc-0.0.0.9-py3-none-any.whl
pip3a install numpy==1.23.3