#!/bin/bash


EVE_PYTHON_CMD=${EVE_PYTHON_CMD:-python3}
EVE_ROOT="$(dirname $(dirname $(readlink -e $0)))"

echo "pip user = $PIP_USER"

MAKE_VENV="true"

# Help function
usage() {
    echo -e "\nUsage: init_env.sh [OPTIONS] [env_name]"
    echo -e ""
    echo -e " Args:"
    echo -e "   env_name : name of the venv (default: '.eve.venv')"
    echo -e ""
    echo -e " Options:"
    echo -e "   -h : show help"
    echo -e "   -i : skip venv creation and only install packages"
    echo -e ""
}

# Parse options
while getopts "hi" opt; do
    case "${opt}" in
        h)
            usage
            exit 0
            ;;
        i)
            MAKE_VENV="false"
            ;;
        :)
            echo -e "\nError: missing argument for option '${OPTARG}'"
            ;;
        *)
            echo -e "\nError: invalid option '${opt}'"
            usage
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))

if [ $# -gt 1 ]; then
    echo -e "\nError: wrong number of arguments"
    usage
    exit 2
fi

if [ "$MAKE_VENV" = "true" ]; then
    EVE_VENV_NAME=${1:-".eve.venv"}
    EVE_VENV_PATH=${EVE_ROOT}/${EVE_VENV_NAME}

    # If a virtual env is active, deactivate it first
    deactivate >/dev/null 2>/dev/null

    echo -e "\nCreating Python virtual environment..."
    ${EVE_PYTHON_CMD} -m venv ${EVE_VENV_PATH}
    source ${EVE_VENV_PATH}/bin/activate
    if [ "$(which ${EVE_PYTHON_CMD})" != "${EVE_VENV_PATH}/bin/${EVE_PYTHON_CMD}" ]; then
        echo -e "\nERROR: virtual environment has not been sucessfully created."
        echo -e "Exiting...\n"
        exit 10
    fi
    echo -e "\nDone!"
fi

set -e

echo -e "\nInstalling Python packages..."
${EVE_PYTHON_CMD} -m pip install --upgrade pip setuptools wheel
${EVE_PYTHON_CMD} -m pip install -r ${EVE_ROOT}/requirements_dev.txt
${EVE_PYTHON_CMD} -m pip install -e ${EVE_ROOT}[cpp]
echo -e "\nDone!"

echo -e "\nCreating local debug folder ('_local') ..."
mkdir -p ${EVE_ROOT}/_local
echo -e "\nDone!"
echo ""

echo -e "\nGenerating documentation ..."
cd ${EVE_ROOT}/docs && make html && cd ${EVE_ROOT}
echo -e "\nDone!"
echo ""

