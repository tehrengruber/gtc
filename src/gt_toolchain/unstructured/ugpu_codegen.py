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

from mako import template as mako_tpl

from eve import codegen

from .ugpu import LocationType


# from gt_toolchain import common


class UgpuCodeGenerator(codegen.TemplatedGenerator):
    LOCATION_TYPE_TO_STR: ClassVar[Mapping[LocationType, Mapping[str, str]]] = MappingProxyType(
        {LocationType.Vertex: "vertex", LocationType.Edge: "edge", LocationType.Cell: "cell"}
    )

    class Templates:
        Node = mako_tpl.Template("${_this_node.__class__.__name__.upper()}")  # only for testing

        #         UnstructuredField = mako_tpl.Template(
        #             """<%
        #     loc_type = _this_generator.LOCATION_TYPE_TO_STR[_this_node.location_type]["singular"]
        #     data_type = _this_generator.DATA_TYPE_TO_STR[_this_node.data_type]
        #     sparseloc = "sparse_" if _this_node.sparse_location_type else ""
        # %>
        #   dawn::${ sparseloc }${ loc_type }_field_t<LibTag, ${ data_type }>& ${ name };"""
        #         )

        Kernel = mako_tpl.Template(
            """<%
            primary_location = _this_generator.LOCATION_TYPE_TO_STR[_this_node.primary_connectivity]
            primary_sid_name = _this_node.primary_sid_composite.name
            primary_origins = primary_sid_name + "_origins"
            primary_strides = primary_sid_name + "_strides"
            primary_ptrs = primary_sid_name + "_ptrs"
            primary_sid_param = "{0}_t {0}, {1}_t {1}".format(primary_origins, primary_strides)
            %>template<class ${ primary_location }_conn_t>
        __global__ void ${ name }(${ primary_location }_conn_t ${ primary_location }_conn, ${ primary_sid_param }) {
            auto idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= gridtools::next::connectivity::size(${ primary_location }_conn))
                return;

            auto ${ primary_ptrs } = ${ primary_origins }();
            gridtools::sid::shift(${ primary_ptrs }, gridtools::device::at_key<${ primary_location }>(${ primary_strides }), idx);

            ${ "".join(ast) }
        }
        """
        )

        FieldAccess = "gridtools::device::at_key<{tag}>({sid_composite}_ptrs)"

        AssignStmt = "{left} = {right};"

        BinaryOp = "{left} {op} {right}"

        SidTag = "struct {name};"

        Computation = mako_tpl.Template(
            """${ ''.join(tags) }

        ${ ''.join(kernels) }"""
        )

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    # def visit_HorizontalLoop(self, node, **kwargs) -> str:
    #     return self.generic_visit(node, iter_var="t", **kwargs)

    # def visit_ReduceOverNeighbourExpr(self, node, *, iter_var, **kwargs) -> str:
    #     outer_iter_var = iter_var
    #     return self.generic_visit(node, outer_iter_var=outer_iter_var, iter_var="redIdx", **kwargs,)
