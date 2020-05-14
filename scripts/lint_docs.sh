#!/bin/bash

set -e

EVE_ROOT="$(dirname $(dirname $(readlink -e $0)))"
flake8 --select=D --ignore= --doctests --show-source ${EVE_ROOT}/src
#pytest   --collect-only --doctest-modules
