# Define aliases for the conda environment
alias pip3rf="$HOME/anaconda3/envs/rofunc/bin/pip"
alias pyana3rf="$HOME/anaconda3/envs/rofunc/bin/python3.8"

# Install the requirements and rofunc
cd ../
pip3rf install -r requirements.txt
pip3rf install .



