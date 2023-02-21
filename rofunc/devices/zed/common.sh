PURPLE='\033[0;35m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
LIGHT_GREEN='\033[1;32m'
NC='\033[0m' # No Color

echo_info() {
  echo -e $PURPLE$1$NC
}

echo_success() {
  echo -e $LIGHT_GREEN$1$NC
}

echo_warning() {
  echo -e $YELLOW$1$NC
}

echo_failure() {
  echo -e $RED$1$NC
}

print_divider() {
  TITLE=$(echo "$1" | tr [:lower:] [:upper:])
  STATUS=$(echo "$2" | tr [:lower:] [:upper:])

  if [ $STATUS == FINISHED ]; then
    printf "$LIGHT_GREEN─%.0s$NC"  $(seq 1 103)
    printf "\n"
    printf "$LIGHT_GREEN%-90s : %10s$NC\n" "$TITLE" "$STATUS"
  else
    printf "$PURPLE%-90s : %10s$NC\n" "$TITLE" "$STATUS"
    printf "$PURPLE─%.0s$NC"  $(seq 1 103)
    printf "\n"
  fi
}
