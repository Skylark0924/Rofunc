# Define aliases for the conda environment
alias pip3rfl="$HOME/anaconda3/envs/rofunc_isaaclab/bin/pip"
alias py3rfl="$HOME/anaconda3/envs/rofunc_isaaclab/bin/python3.10"

# Download IsaacLab and install it
cd ../
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
ln -s $HOME/.local/share/ov/pkg/isaac-sim-4.1.0 _isaac_sim
./isaaclab.sh --install

# Install the requirements and rofunc
pip3rfl install shutup
pip3rfl install omegaconf
pip3rfl install hydra-core
pip3rfl install nestle
pip3rfl install gdown