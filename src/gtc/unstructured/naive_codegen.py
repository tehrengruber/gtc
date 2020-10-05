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


from types import MappingProxyType
from typing import ClassVar, Mapping

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gtc import common

from .naive import LocationType


class NaiveCodeGenerator(codegen.TemplatedGenerator):
    DATA_TYPE_TO_STR: ClassVar[Mapping[common.DataType, str]] = MappingProxyType(
        {
            common.DataType.BOOLEAN: "bool",
            common.DataType.INT32: "int",
            common.DataType.UINT32: "unsigned_int",
            common.DataType.FLOAT32: "float",
            common.DataType.FLOAT64: "double",
        }
    )

    LOCATION_TYPE_TO_STR_MAP: ClassVar[Mapping[LocationType, Mapping[str, str]]] = MappingProxyType(
        {
            LocationType.Node: MappingProxyType({"singular": "vertex", "plural": "vertices"}),
            LocationType.Edge: MappingProxyType({"singular": "edge", "plural": "edges"}),
            LocationType.Face: MappingProxyType({"singular": "cell", "plural": "cells"}),
        }
    )

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    def visit_DataType(self, node, **kwargs) -> str:
        return self.DATA_TYPE_TO_STR[node]

    def visit_LocationType(self, node, **kwargs) -> Mapping[str, str]:
        return self.LOCATION_TYPE_TO_STR_MAP[node]

    Node = as_mako("${_this_node.__class__.__name__.upper()}")  # only for testing

    UnstructuredField = as_mako(
        """<%
loc_type = location_type["singular"]
sparseloc = "sparse_" if _this_node.sparse_location_type else ""
%>
dawn::${ sparseloc }${ loc_type }_field_t<LibTag, ${ data_type }>& ${ name };"""
    )

    FieldAccessExpr = as_mako(
        """<%
sparse_index = "m_sparse_dimension_idx, " if _this_node.is_sparse else ""
field_acc_itervar = outer_iter_var if _this_node.is_sparse else iter_var
%>${ name }(deref(LibTag{}, ${ field_acc_itervar }), ${ sparse_index } k)"""
    )

    AssignmentExpr = as_fmt("{left} = {right}")

    VarAccessExpr = as_fmt("{name}")

    BinaryOp = as_fmt("{left} {op} {right}")

    ExprStmt = as_fmt("\n{expr};")

    VarDeclStmt = as_fmt("\n{data_type} {name};")

    TemporaryFieldDeclStmt = as_mako(
        """using dawn::allocateEdgeField;
        auto ${ name } = allocate${ location_type['singular'].capitalize() }Field<${ data_type }>(mesh);"""
    )

    ForK = as_mako(
        """<%
if _this_node.loop_order == _this_module.common.LoopOrder.FORWARD:
    k_init = '0'
    k_cond = 'k < k_size'
    k_step = '++k'
else:
    k_init = 'k_size -1'
    k_cond = 'k >= 0'
    k_step = '--k'
%>for (int k = ${k_init}; ${k_cond}; ${k_step}) {
int m_sparse_dimension_idx;
${ "".join(horizontal_loops) }\n}"""
    )

    HorizontalLoop = as_mako(
        """<%
loc_type = location_type['plural'].title()
%>for(auto const & t: get${ loc_type }(LibTag{}, mesh)) ${ ast }"""
    )

    def visit_HorizontalLoop(self, node, **kwargs) -> str:
        return self.generic_visit(node, iter_var="t", **kwargs)

    BlockStmt = as_mako("{${ ''.join(statements) }\n}")

    ReduceOverNeighbourExpr = as_mako(
        """<%
right_loc_type = right_location_type["singular"].title()
loc_type = location_type["singular"].title()
%>(m_sparse_dimension_idx=0,reduce${ right_loc_type }To${ loc_type }(mesh, ${ outer_iter_var }, ${ init }, [&](auto& lhs, auto const& ${ iter_var }) {
lhs ${ operation }= ${ right };
m_sparse_dimension_idx++;
return lhs;
}))"""
    )

    def visit_ReduceOverNeighbourExpr(self, node, *, iter_var, **kwargs) -> str:
        outer_iter_var = iter_var
        return self.generic_visit(node, outer_iter_var=outer_iter_var, iter_var="redIdx", **kwargs,)

    LiteralExpr = as_fmt("({data_type}){value}")

    Stencil = as_mako(
        """
void ${name}() {
using dawn::deref;

${ "\\n".join(declarations) if _this_node.declarations else ""}

${ "".join(k_loops) }
}
"""
    )

    Computation = as_mako(
        """<%
stencil_calls = '\\n'.join("{name}();".format(name=s.name) for s in _this_node.stencils)
ctor_field_params = ', '.join(
    'dawn::{sparse_loc}{loc_type}_field_t<LibTag, {data_type}>& {name}'.format(
        loc_type=_this_generator.LOCATION_TYPE_TO_STR_MAP[p.location_type]['singular'],
        name=p.name,
        data_type=_this_generator.DATA_TYPE_TO_STR[p.data_type],
        sparse_loc="sparse_" if p.sparse_location_type else ""
    )
    for p in _this_node.params
)
ctor_field_initializers = ', '.join(
    '{name}({name})'.format(name=p.name) for p in _this_node.params
)
%>#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class generated {
private:
dawn::mesh_t<LibTag>& mesh;
int const k_size;

${ ''.join(params) }
${ ''.join(stencils) }

public:
generated(dawn::mesh_t<LibTag>& mesh, int k_size, ${ ctor_field_params }): mesh(mesh), k_size(k_size), ${ ctor_field_initializers } {}

void run() {
${ stencil_calls }
}
};
}
}

"""
    )
