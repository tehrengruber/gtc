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


from setuptools import setup


if __name__ == "__main__":
    setup(
        use_scm_version={
            "fallback_version": "X.X.X.unknown",
            "local_scheme": "dirty-tag",
            "version_scheme": "python-simplified-semver",
            "write_to": "src/eve/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
        }
    )