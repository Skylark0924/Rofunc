# Define aliases for the conda environment
alias pip3rf="$HOME/anaconda3/envs/rofunc/bin/pip"
alias pyana3rf="$HOME/anaconda3/envs/rofunc/bin/python3.8"

# Box3d-py requires swig
brew install swig

# Install pinocchio
#brew tap gepetto/homebrew-gepetto
#brew install pinocchio
#
#export PATH=/opt/openrobots/bin:$PATH
#export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
#export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
#export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH

# Downgrade pip to 21.0 for avoiding useless installation of different versions
pip3rf install pip==21.3.1

# Install the requirements and rofunc
pip3rf install -r requirements.txt
pip3rf install . --use-deprecated=legacy-resolver





