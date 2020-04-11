# This file should be sourced by a Bash-compatible shell

EVE_PYTHON_CMD=${EVE_PYTHON_CMD:-python3}
EVE_VENV_NAME=${EVE_VENV_NAME:-".eve.venv"}

EVE_ROOT="$(dirname $(dirname $(readlink -e $BASH_SOURCE)))"
EVE_VENV_PATH=${EVE_ROOT}/${EVE_VENV_NAME}

if [ "$0" = ${BASH_SOURCE} ]; then
    echo "Error: this script needs to be sourced!"
    echo "    $> source ${BASH_SOURCE}"
    exit -1
fi

# If a virtual env is active, deactivate it first
deactivate >/dev/null 2>/dev/null

if [ ! -f ${EVE_VENV_PATH}/bin/activate ]; then
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
    echo ""
else
    source ${EVE_VENV_PATH}/bin/activate
fi

unset EVE_ROOT EVE_VENV_PATH

echo -e "\nDeactivate by:"
echo -e "    $> deactivate"
