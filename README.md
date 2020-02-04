Eve: A stencil toolchain in pure Python
=======================================

Installation instructions
-------------------------

Eve contains a standard `setup.py` installation script which can be
used to install the package with *pip*. As usual, create first a new
virtual environment for your project:

    # Create a virtual environment using the 'venv' module
    python -m venv path_for_the_new_venv

    # Activate the virtual environment and make sure that 'wheel' is installed
    source path_for_the_new_venv/bin/activate
    pip install --upgrade wheel


Next, clone the repository and use an _editable_ installation of Eve:

    # First, clone the repository
    git clone https://github.com/egparedes/eve.git

    # Then install the Python package directly from the local repository
    # adding the '-e' flag to get an editable installation
    pip install -e ./eve

    # Finally, install the additional development tools
    pip install -r ./eve/requirements_dev.txt
