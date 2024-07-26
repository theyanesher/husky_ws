#!/usr/bin/env bash
set -e

# setup ros environment
source "/home/nuc/husky_ws/devel/setup.bash"

exec "$@"