#!/bin/bash

SCRIPTS_ROOT=$(dirname "$(readlink -e $0)")

python $SCRIPTS_ROOT/init_env.py "$@"
