# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Version specification."""


import os

from packaging.version import parse  # type: ignore


version_file_path = os.path.join(os.path.dirname(__file__), "_SCM_VERSION.txt")
if not os.path.isfile(version_file_path):
    # Fallback to static version file
    version_file_path = os.path.join(os.path.dirname(__file__), "_VERSION.txt")

with open(version_file_path, "r") as version_file:
    __version__: str = version_file.read().strip()
    __versioninfo__ = parse(__version__)
