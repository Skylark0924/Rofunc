#!/bin/bash -i
shopt -s expand_aliases
alias pip3rf="/home/skylark/anaconda3/envs/rofunc/bin/pip"
alias pyana3rf="/home/skylark/anaconda3/envs/rofunc/bin/python3.8"

cd ../dist
pip3rf uninstall -y rofunc
cd ..
pyana3rf setup.py bdist_wheel sdist
cd ./dist
pip3rf install rofunc-0.0.0.9-py3-none-any.whl
pip3rf install numpy==1.23.3