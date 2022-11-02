#!/bin/bash -i
shopt -s expand_aliases
alias pip3p="/home/ubuntu/anaconda3/envs/plast/bin/pip"
alias pyana3p="/home/ubuntu/anaconda3/envs/plast/bin/python3.7"
# Expand aliases defined in the shell ~/.bashrc

pip3p uninstall -y rofunc
cd ..
rm -rf ./build
rm -rf ./dist
pyana3p setup.py bdist_wheel sdist
cd ./dist
pip3p install rofunc-0.0.0.9.1-py3-none-any.whl