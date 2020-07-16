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
from gt_toolchain.unstructured.ugpu import Kernel, LocationType


# from gt_toolchain import common


class UgpuCodeGenerator(codegen.TemplatedGenerator):
    LOCATION_TYPE_TO_STR: ClassVar[Mapping[LocationType, Mapping[str, str]]] = MappingProxyType(
        {LocationType.Vertex: "vertex", LocationType.Edge: "edge", LocationType.Cell: "cell"}
    )

    def make_kernel_call(self, kernel: Kernel):
        location = self.LOCATION_TYPE_TO_STR[kernel.primary_connectivity]
        return mako_tpl.Template(
            """{
            auto primary_connectivity = gridtools::next::mesh::connectivity<${ location }>(mesh);

            auto ${ kernel.primary_sid_composite.name }_fields = tu::make<gridtools::sid::composite::keys<${ ','.join(e.name + '_tag' for e in kernel.primary_sid_composite.entries) }>::values>(
                ${ ','.join(e.name for e in kernel.primary_sid_composite.entries) });
            static_assert(gridtools::is_sid<decltype(${ kernel.primary_sid_composite.name }_fields)>{});

            auto [blocks, threads_per_block] = gridtools::next::cuda_util::cuda_setup(gridtools::next::connectivity::size(${ kernel.primary_sid_composite.name }_fields));
            ${ kernel.name }_impl_::${ kernel.name }<<<blocks, threads_per_block>>>(
                primary_connectivity, gridtools::sid::get_origin(${ kernel.primary_sid_composite.name }_fields), gridtools::sid::get_strides(${ kernel.primary_sid_composite.name }_fields));
            GT_CUDA_CHECK(cudaDeviceSynchronize());
        }"""
        ).render(kernel=kernel, location=location)

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

        # TODO improve tag handling
        FieldAccess = "gridtools::device::at_key<{tag}_tag>({sid_composite}_ptrs)"

        AssignStmt = "{left} = {right};"

        BinaryOp = "{left} {op} {right}"

        SidTag = "struct {name};"

        Computation = mako_tpl.Template(
            """<%
                sid_tags = set()
                for k in _this_node.kernels:
                    for e in k.primary_sid_composite.entries:
                        sid_tags.add("struct " + e.name + "_tag;")
                kernel_calls = map(_this_generator.make_kernel_call, _this_node.kernels)
            %>#include <gridtools/next/cuda_util.hpp>
#include <gridtools/next/tmp_gpu_storage.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/next/atlas_adapter.hpp>
#include <gridtools/next/atlas_array_view_adapter.hpp>
#include <gridtools/next/atlas_field_util.hpp>

        namespace ${ name }_impl_ {
        ${ ''.join(sid_tags) }

        ${ ''.join(kernels) }
        }

        template<class mesh_t, ${ ','.join('class ' + p.name + '_t' for p in _this_node.parameters) }>
        void ${ name }(mesh_t&& mesh, ${ ','.join(p.name + '_t&& ' + p.name for p in _this_node.parameters) }){
            ${ ''.join(kernel_calls) }
        }
        """
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
