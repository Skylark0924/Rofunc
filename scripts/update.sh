#!/bin/bash -i
shopt -s expand_aliases
alias pip3p="/home/ubuntu/anaconda3/envs/plast/bin/pip"
alias pyana3p="/home/ubuntu/anaconda3/envs/plast/bin/python3.7"
# Expand aliases defined in the shell ~/.bashrc

cd ../scripts
pip3p uninstall -y rofunc
cd ..
pyana3p setup.py bdist_wheel sdist
cd ./dist
pip3p install rofunc-0.0.0.9-py3-none-any.whl