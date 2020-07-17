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
from gt_toolchain.unstructured.ugpu import (
    Computation,
    Kernel,
    LocationType,
    NeighborChain,
    SidComposite,
)


# Ugpu Codegen convention:
# - fields of the same LocationType are composed in a SID composite with name <LocationType>_sid,
#   and the respective SID entities are <LocationType>_origins, <LocationType>_ptrs, <LocationType>_strides.
#   Sparse fields are in the composite of the primary location.
#   (i.e. the correct composite name can be found by the location type, let's see where this goes)
#   (pure vertical fields should probably go into the primary composite)
# - connectivities are named <From>2<To>_connectivity
# - tags have a suffix `_tag`

# TODO: think about having only one composite with all fields


class UgpuCodeGenerator(codegen.TemplatedGenerator):
    LOCATION_TYPE_TO_STR: ClassVar[Mapping[LocationType, Mapping[str, str]]] = MappingProxyType(
        {LocationType.Vertex: "vertex", LocationType.Edge: "edge", LocationType.Cell: "cell"}
    )

    def make_connectivity_name(self, chain: NeighborChain):
        return "2".join(self.LOCATION_TYPE_TO_STR[e] for e in chain.chain) + "_connectivity"

    def make_kernel_call(self, kernel: Kernel):
        def compose_sid_composite(composite: SidComposite, **kwargs):
            tags = [e.name + "_tag" for e in composite.entries]
            # tags = ",".join(e.name + "_tag" for e in composite.entries)
            fields = [e.name for e in composite.entries]

            connectivity = kwargs.get("connectivity", None)
            if connectivity:
                tags.append("connectivity_tag")
                fields.append(
                    "gridtools::next::connectivity::neighbor_table({})".format(
                        self.make_connectivity_name(connectivity)
                    )
                )

            sid_name = self.LOCATION_TYPE_TO_STR[composite.location_type]
            return mako_tpl.Template(
                """
                auto ${ sid_name }_fields = tu::make<gridtools::sid::composite::keys<${ ','.join(tags) }>::values>(
                    ${ ','.join(fields)});
                static_assert(gridtools::is_sid<decltype(${ sid_name }_fields)>{});
                """
            ).render(tags=tags, fields=fields, sid_name=sid_name)

        def make_composite_args(composite: SidComposite):
            sid_name = self.LOCATION_TYPE_TO_STR[composite.location_type]
            return mako_tpl.Template(
                "gridtools::sid::get_origin(${ sid_name }_fields), gridtools::sid::get_strides(${ sid_name }_fields)"
            ).render(sid_name=sid_name)

        def make_connectivities(chain: NeighborChain):
            return mako_tpl.Template(
                "auto ${ make_connectivity_name(chain) } = gridtools::next::mesh::connectivity<std::tuple<${ ','.join(loc2str[e] for e in chain.chain) }>>(mesh);"
            ).render(
                chain=chain,
                loc2str=self.LOCATION_TYPE_TO_STR,
                make_connectivity_name=self.make_connectivity_name,
            )

        location = self.LOCATION_TYPE_TO_STR[kernel.primary_connectivity]

        # FIXME we need to be able to pass more than one
        first_other_connectivity = None
        if kernel.other_connectivities and len(kernel.other_connectivities) == 1:
            first_other_connectivity = kernel.other_connectivities[0]

        all_composites = [kernel.primary_sid_composite] + (kernel.other_sid_composites or [])
        composed_sids = [
            compose_sid_composite(
                kernel.primary_sid_composite, connectivity=first_other_connectivity
            )
        ]
        composed_sids.extend(map(compose_sid_composite, kernel.other_sid_composites or []))
        composed_sids_arguments = map(make_composite_args, all_composites)
        connectivities = map(make_connectivities, kernel.other_connectivities or [])
        connectivity_args = ["primary_connectivity"]
        connectivity_args.extend(
            map(self.make_connectivity_name, kernel.other_connectivities or [])
        )
        return mako_tpl.Template(
            """{
            auto primary_connectivity = gridtools::next::mesh::connectivity<${ location }>(mesh);
            ${ ''.join(connectivities) }

            ${ ''.join(composed_sids) }

            auto [blocks, threads_per_block] = gridtools::next::cuda_util::cuda_setup(gridtools::next::connectivity::size(primary_connectivity));
            ${ kernel.name }_impl_::${ kernel.name }<<<blocks, threads_per_block>>>(${','.join(connectivity_args)}, ${ ','.join(composed_sids_arguments) });
            GT_CUDA_CHECK(cudaDeviceSynchronize());
        }"""
        ).render(
            kernel=kernel,
            location=location,
            composed_sids=composed_sids,
            composed_sids_arguments=composed_sids_arguments,
            connectivities=connectivities,
            connectivity_args=connectivity_args,
        )
        # auto ${ primary_sid_name }_fields = tu::make<gridtools::sid::composite::keys<${ ','.join(e.name + '_tag' for e in kernel.primary_sid_composite.entries) }>::values>(
        #     ${ ','.join(e.name for e in kernel.primary_sid_composite.entries) });
        # ${ ''.join(other_sid_composites)}

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
            connectivities = [_this_generator.LOCATION_TYPE_TO_STR[_this_node.primary_connectivity] + "_connectivity" ]
            connectivities.extend(map(_this_generator.make_connectivity_name, _this_node.other_connectivities or []))
            all_sids = [_this_node.primary_sid_composite] + (_this_node.other_sid_composites or [])
            all_sid_names = map(lambda s: _this_generator.LOCATION_TYPE_TO_STR[s.location_type], all_sids)
            primary_sid_name =_this_generator.LOCATION_TYPE_TO_STR[_this_node.primary_sid_composite.location_type]
            primary_origins = primary_sid_name + "_origins"
            primary_strides = primary_sid_name + "_strides"
            primary_ptrs = primary_sid_name + "_ptrs"
            primary_sid_params = map(lambda s: "{0}_origins_t {0}_origins, {0}_strides_t {0}_strides".format(s), all_sid_names)
            %>template<${ ','.join("class {}_t".format(c) for c in connectivities) }>
        __global__ void ${ name }(${ ','.join( "{0}_t {0}".format(c) for c in connectivities ) }, ${ ','.join(primary_sid_params) }) {
            auto idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= gridtools::next::connectivity::size(${ primary_location }_connectivity))
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

        BinaryOp = "({left} {op} {right})"

        SidTag = "struct {name};"

        Computation = mako_tpl.Template(
            """<%
                sid_tags = set()
                sid_tags.add("struct connectivity_tag;")
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
            %>for (int neigh = 0; neigh < gridtools::next::connectivity::max_neighbors(${ parent_location }2${ body_location }_connectivity); ++neigh) {
                // body
                auto ${ body_location }_ptrs = ${ body_location }_origins();
                auto absolute_neigh_index = *gridtools::device::at_key<connectivity_tag>(${ body_location }_ptrs);
                gridtools::sid::shift(
                    ${ body_location }_ptrs, gridtools::device::at_key<${ body_location }>(${ body_location }_strides), absolute_neigh_index);

                ${ ''.join(body) }
                // body end

                gridtools::sid::shift(${ parent_location }_ptrs, gridtools::device::at_key<neighbor>(${ parent_location }_strides), 1);
            }
            gridtools::sid::shift(${ parent_location }_ptrs,
                gridtools::device::at_key<neighbor>(${ parent_location }_strides),
                -gridtools::next::connectivity::max_neighbors(${ parent_location }2${ body_location }_connectivity));
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
