#!/bin/bash -i
alias pip3rf="$HOME/anaconda3/envs/rofunc/bin/pip"
alias pyana3rf="$HOME/anaconda3/envs/rofunc/bin/python3.8"
# Expand aliases defined in the shell ~/.bashrc

pip3rf uninstall -y rofunc
cd ..
rm -rf ./build
rm -rf ./dist
# pip3rf install -r requirements.txt
pyana3rf setup.py bdist_wheel sdist
cd ./dist
pip3rf install *.whl