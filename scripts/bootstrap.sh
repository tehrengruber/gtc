#!/bin/bash

set -e

EVE_PYTHON_CMD=${EVE_PYTHON_CMD:-python3}
EVE_VENV_NAME=${EVE_VENV_NAME:-".eve.venv"}

EVE_ROOT="$(dirname $(dirname $(readlink -e $0)))"
EVE_VENV_PATH=${EVE_ROOT}/${EVE_VENV_NAME}

echo -e "\nCreating Python virtual environment..."
${EVE_PYTHON_CMD} -m venv ${EVE_VENV_PATH}
source ${EVE_VENV_PATH}/bin/activate
if [ "$(which ${EVE_PYTHON_CMD})" != "${EVE_VENV_PATH}/bin/${EVE_PYTHON_CMD}" ]; then
    echo -e "\nERROR: virtual environment has not been sucessfully created."
    echo -e "Exiting...\n"
    return
fi
pip install --upgrade pip setuptools wheel
pip install -r ${EVE_ROOT}/requirements_dev.txt
pip install -e ${EVE_ROOT}[cpp]
echo -e "\nDone!"

echo -e "\nCreating local debug folder ('_local') ..."
mkdir -p ${EVE_ROOT}/_local
echo -e "\nDone!"
echo ""

deactivate
