# Turn to the directory of this script
cd $(dirname $0) || exit 1

# Load common functions and variables from the common script
COMMON_SCRIPT="./common.sh"
source "$COMMON_SCRIPT"

install_zed(){
  print_divider "Install Zed" started
  if [ $(lsb_release -sc) == jammy ]; then #ubuntu 22.04
    SDK_ADDRESS=https://download.stereolabs.com/zedsdk/3.8/cu117/ubuntu22
  elif [ $(lsb_release -sc) == focal ]; then #ubuntu 20.04
    SDK_ADDRESS=https://download.stereolabs.com/zedsdk/3.8/cu117/ubuntu20
  elif [ $(lsb_release -sc) == bionic ]; then #ubuntu 18.04
    SDK_ADDRESS=https://download.stereolabs.com/zedsdk/3.8/cu117/ubuntu18
  else
    echo_failure "Only Ubuntu 22.04, 20.04 and 18.04 are supported in Rofunc right now."
    return
  fi

## Download the zed SDK and set its name as ZED_SDK.run
  cd $HOME/$hostname\Downloads || exit 1
  wget $SDK_ADDRESS -O ZED_SDK.run
  sudo apt install zstd
  chmod +x ZED_SDK.run
  ./ZED_SDK.run
  print_divider "Install Zed" finished
  sudo rm ZED_SDK.run

}

help() {
  echo "===== Setup Script User Guide ====="
  echo "Syntax: setup.sh [option]"
  echo "options:"
  echo "no option     Install all packages and do the tests, recommended for new machine."
  echo "-h | --help   This is the ZED camera setup tool"
}

if [ $# -eq 0 ]; then
  install_zed
else
  case "$1" in
  -h | --help)
   help
   ;;
  esac
fi

exit 0
