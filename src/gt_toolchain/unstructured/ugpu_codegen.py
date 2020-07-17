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
from gt_toolchain.unstructured.ugpu import Computation, Kernel, LocationType


# Ugpu Codegen convention:
# - fields of the same LocationType are composed in a SID composite with name <LocationType>_sid,
#   and the respective SID entities are <LocationType>_origins, <LocationType>_ptrs, <LocationType>_strides.
#   Sparse fields are in the composite of the primary location.
#   (i.e. the correct composite name can be found by the location type, let's see where this goes)
# - connectivities are named <From>2<To>_conn
# - tags have a suffix `_tag`


class UgpuCodeGenerator(codegen.TemplatedGenerator):
    LOCATION_TYPE_TO_STR: ClassVar[Mapping[LocationType, Mapping[str, str]]] = MappingProxyType(
        {LocationType.Vertex: "vertex", LocationType.Edge: "edge", LocationType.Cell: "cell"}
    )

    def make_kernel_call(self, kernel: Kernel):
        location = self.LOCATION_TYPE_TO_STR[kernel.primary_connectivity]
        primary_sid_name = self.LOCATION_TYPE_TO_STR[kernel.primary_sid_composite.location_type]
        return mako_tpl.Template(
            """{
            auto primary_connectivity = gridtools::next::mesh::connectivity<${ location }>(mesh);

            auto ${ primary_sid_name }_fields = tu::make<gridtools::sid::composite::keys<${ ','.join(e.name + '_tag' for e in kernel.primary_sid_composite.entries) }>::values>(
                ${ ','.join(e.name for e in kernel.primary_sid_composite.entries) });
            static_assert(gridtools::is_sid<decltype(${ primary_sid_name }_fields)>{});

            auto [blocks, threads_per_block] = gridtools::next::cuda_util::cuda_setup(gridtools::next::connectivity::size(primary_connectivity));
            ${ kernel.name }_impl_::${ kernel.name }<<<blocks, threads_per_block>>>(
                primary_connectivity, gridtools::sid::get_origin(${ primary_sid_name }_fields), gridtools::sid::get_strides(${ primary_sid_name }_fields));
            GT_CUDA_CHECK(cudaDeviceSynchronize());
        }"""
        ).render(kernel=kernel, location=location, primary_sid_name=primary_sid_name)

    def location_type_from_dimensions(self, dimensions):
        location_type = [dim for dim in dimensions if isinstance(dim, LocationType)]
        if len(location_type) != 1:
            raise ValueError("Doesn't contain a LocationType!")
        return location_type[0]

    class Templates:
        Node = mako_tpl.Template("${_this_node.__class__.__name__.upper()}")  # only for testing

        Kernel = mako_tpl.Template(
            """<%
            primary_location = _this_generator.LOCATION_TYPE_TO_STR[_this_node.primary_connectivity]
            primary_sid_name =_this_generator.LOCATION_TYPE_TO_STR[_this_node.primary_sid_composite.location_type]
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

        FieldAccess = mako_tpl.Template(
            """<%
            # TODO lookup symbol table for the tag/name of the field (instead of kwargs)
            usid = [p for p in computation_fields if p.name == _this_node.name]
            if len(usid) != 1:
                raise ValueError("Symbol not found or not unique!")
            location_type = _this_generator.location_type_from_dimensions(usid[0].dimensions)
            location_str = _this_generator.LOCATION_TYPE_TO_STR[location_type]
        %>gridtools::device::at_key<${ name }_tag>(${ location_str }_ptrs)"""
        )

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

        NeighborLoop = mako_tpl.Template(
            """<%
            body_location = _this_generator.LOCATION_TYPE_TO_STR[_this_node.body_location_type]
            parent_location = _this_generator.LOCATION_TYPE_TO_STR[_this_node.location_type]
            %>for (int neigh = 0; neigh < gridtools::next::connectivity::max_neighbors(${ parent_location }2${ body_location }); ++neigh) {
                // body
                auto absolute_neigh_index = *gridtools::device::at_key<connectivity_tag>(${ body_location }_ptrs);
                auto ${ body_location }_ptrs = ${ body_location }_origins();
                gridtools::sid::shift(
                    ${ body_location }_ptrs, gridtools::device::at_key<${ body_location }>(${ body_location }_strides), absolute_neigh_index);

                ${ ''.join(body) }
                // body end

                gridtools::sid::shift(${ parent_location }_ptrs, gridtools::device::at_key<neighbor>(${ parent_location }_strides), 1);
            }
            gridtools::sid::shift(${ parent_location }_ptrs,
                gridtools::device::at_key<neighbor>(${ parent_location }_strides),
                -gridtools::next::connectivity::max_neighbors(${ parent_location }2${ body_location }));
            """
        )

        Literal = "{value}"

        VarAccess = "{name}"

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    def visit_Computation(self, node: Computation, **kwargs) -> str:
        return self.generic_visit(
            node, computation_fields=node.parameters + node.temporaries, **kwargs
        )

    # def visit_ReduceOverNeighbourExpr(self, node, *, iter_var, **kwargs) -> str:
    #     outer_iter_var = iter_var
    #     return self.generic_visit(node, outer_iter_var=outer_iter_var, iter_var="redIdx", **kwargs,)
