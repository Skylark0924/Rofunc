# Define aliases for the conda environment
alias pip3rf="$HOME/anaconda3/envs/rofunc/bin/pip"
alias py3rf="$HOME/anaconda3/envs/rofunc/bin/python3.8"

# Box3d-py requires swig
sudo apt-get install swig

# Install pinocchio
#sudo apt install -qqy lsb-release gnupg2 curl
#echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list
#curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
#sudo apt-get update
#sudo apt install -qqy robotpkg-py3*-pinocchio
#
#export PATH=/opt/openrobots/bin:$PATH
#export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
#export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
#export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH # Adapt your desired python version here
#export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH

# Downgrade pip to 21.0 for avoiding useless installation of different versions
pip3rf install pip==21.3.1

# Install the requirements and rofunc
pip3rf install -r requirements.txt
pip3rf install gdown==5.2.0

# Download data
cd ./examples/
gdown https://drive.google.com/uc?id=1pOzD61CQJcy4L2hXveT1cGiD0AkIDt_c&export=download
unzip -q data.zip
rm data.zip
cd ../

pip3rf install . --use-deprecated=legacy-resolver





