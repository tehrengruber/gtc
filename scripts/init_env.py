#!/usr/bin/env python

import argparse
import os
import re
import subprocess
from typing import Optional

DEFAULT_MANAGER = "venv"
DEFAULT_NAME = "gtc"
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class VenvManager:
    @staticmethod
    def create(py_version: str, venv_path: str) -> None:
        cmd = f"python{py_version} -m venv {venv_path}"
        print(f"Creating virtual environment at '{os.path.abspath(venv_path)}' ...")
        print(f"{cmd}")
        try:
            subprocess.check_call(cmd.split())
            cmd = f"{venv_path}/bin/pip3 install --upgrade wheel pip setuptools"
            subprocess.check_call(cmd.split())
        except subprocess.CalledProcessError as e:
            print(f"ERROR!\n{e}")
            exit(-10)

    @staticmethod
    def install(requirements: str, venv_path: Optional[str] = None) -> None:
        pip_prefix = f"{venv_path}/bin/pip3" if venv_path else "pip3"
        try:
            if requirements == "run":
                cmd = f"{pip_prefix} install -e {REPO_ROOT}[cpp]"
            elif requirements == "develop":
                cmd = (
                    f"{pip_prefix} install -e {REPO_ROOT}[cpp] -r {REPO_ROOT}/requirements_dev.txt"
                )
            print(f"{cmd}")
            subprocess.check_call(cmd.split())
        except subprocess.CalledProcessError as e:
            print(f"ERROR!\n{e}")
            exit(-10)


class VirtualenvManager:
    @staticmethod
    def create(py_version: str, venv_path: str) -> None:
        cmd = f"virtualenv -p python{py_version} {venv_path}"
        print(f"Creating virtual environment at '{os.path.abspath(venv_path)}' ...")
        print(f"{cmd}")
        try:
            subprocess.check_call(cmd.split())
        except subprocess.CalledProcessError as e:
            print(f"ERROR!\n{e}")
            exit(-10)

    install = VenvManager.install


class CondaManager:
    @staticmethod
    def create(py_version: str, venv_path: str) -> None:
        cmd = f"conda create -p {venv_path} -y python={py_version}"
        print(f"Creating virtual environment at '{os.path.abspath(venv_path)}' ...")
        print(f"{cmd}")
        try:
            subprocess.check_call(cmd.split())
        except subprocess.CalledProcessError as e:
            print(f"ERROR!\n{e}")
            exit(-10)

    @staticmethod
    def install(requirements: str, venv_path: Optional[str] = None) -> None:
        pip_prefix = f"conda run -p {venv_path} pip3" if venv_path else "pip3"
        try:
            if requirements == "run":
                cmd = f"{pip_prefix} install -e {REPO_ROOT}[cpp]"
            elif requirements == "develop":
                cmd = (
                    f"{pip_prefix} install -e {REPO_ROOT}[cpp] -r {REPO_ROOT}/requirements_dev.txt"
                )
            print(f"{cmd}")
            subprocess.check_call(cmd.split())
        except subprocess.CalledProcessError as e:
            print(f"ERROR!\n{e}")
            exit(-10)


MANAGERS = {"venv": VenvManager, "virtualenv": VirtualenvManager, "conda": CondaManager}


# -- Parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "name",
    default=DEFAULT_NAME,
    nargs="?",
    help="virtual environment name (default: '%(default)s')",
)
parser.add_argument("-p", "--python", default="3.8", help="python version (default: %(default)s)")
parser.add_argument(
    "-d", "--dir", default=".venvs", help="parent directory for the venv (default: %(default)s)"
)
parser.add_argument(
    "-m",
    "--manager",
    choices=list(MANAGERS.keys()),
    default=DEFAULT_MANAGER,
    help="virtual environment manager (default: %(default)s)",
)
parser.add_argument(
    "-r",
    "--requirements",
    choices=["none", "run", "develop"],
    default="develop",
    help="select requirements to install after creation (default: %(default)s)",
)
parser.add_argument(
    "-i",
    "--install-only",
    action="store_true",
    help="skip venv creation and install packages in current environment (default: %(default)s)",
)

args = parser.parse_args()
print("\nOptions:")
for key in sorted(vars(args)):
    print(f"    {key}: {getattr(args, key)}")
print("")

if not re.match(r"3.\d\d?(\.\d\d?)?$", args.python):
    print(f"\nError: invalid Python version specification ('{args.python}')\n")
    exit(-1)

# -- Install
manager = MANAGERS[args.manager]
venv_path = None

if not args.install_only:
    venv_path = f"{args.dir}/{args.name}"
    manager.create(args.python, venv_path)

if args.requirements != "none":
    manager.install(args.requirements, venv_path)

print("Done")
