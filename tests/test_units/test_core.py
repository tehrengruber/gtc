# -*- coding: utf-8 -*-
#
# Eve Toolchain - GridTools Project
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from devtools import debug
import objectpath

from eve import ir


def test_ir_build():
    v_3 = ir.LiteralExpr(value="3", data_type=ir.DataType.INT32)
    v_5 = ir.LiteralExpr(value="5", data_type=ir.DataType.INT32)
    a = ir.BinaryOpExpr(op=ir.BinaryOperator.ADD, left=v_3, right=v_5)
    s = ir.BinaryOpExpr(op=ir.BinaryOperator.SUB, left=v_3, right=v_5)
    m = ir.BinaryOpExpr(op=ir.BinaryOperator.MUL, left=a, right=s)

    assert m is not None and m.left.left == v_3 and m.right.right == v_5


# tree = objectpath.Tree(m, cfg=dict(debug=True))
# tree.data = m.dict()
# tree.setObjectGetter(getattr)
# tree.object_getter(m, "op")
# list(tree.execute("$..*[str(@.op) is '+']"))
