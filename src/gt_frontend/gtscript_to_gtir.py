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
import copy
from typing import List, Union, cast

import eve
import gtc.common as common
import gtc.unstructured.gtir as gtir

from .built_in_types import BuiltInTypeMeta
from .gtscript import Field, Local, Location, Mesh, TemporaryField
from .gtscript_ast import (
    Argument,
    Assign,
    BinaryOp,
    Call,
    Computation,
    Constant,
    Generator,
    Interval,
    IterationOrder,
    LocationComprehension,
    LocationSpecification,
    Pass,
    Stencil,
    SubscriptMultiple,
    SubscriptSingle,
    Symbol,
)


_reduction_mapping = {
    "sum": gtir.ReduceOperator.ADD,
    "product": gtir.ReduceOperator.MUL,
    "min": gtir.ReduceOperator.MIN,
    "max": gtir.ReduceOperator.MAX,
}

# Types
#  - BuiltInType
#  - LocationType
#  - DataType


class SymbolTable:
    def __init__(self, types, constants):
        self.types = types
        self.constants = constants

    def __contains__(self, key):
        return key in self.types

    def __getitem__(self, item):
        return self.types[item]

    def __setitem__(self, item, val):
        if item in self.types:
            if (
                self.types[item] == val
            ):  # todo: just a workaround. remove when symbol table has proper scope!
                return self.types[item]
            raise ValueError("Symbol `{}` already in symbol table.".format(item))

        self.types[item] = val
        return self.types[item]

    def materialize_constant(self, name, expected_type=None):
        """
        Materialize constant symbol with name `name`, i.e. return the value of that symbol. Currently the only constants
        are types.

        Example:
        ```
        self._materialize_constant("Vertex") == LocationType.Vertex
        ```
        """
        if name not in self.types:
            raise ValueError("Symbol {} not found".format(name))
        if name not in self.constants:
            raise ValueError("Symbol {} : {} is not a constant".format(name, self.types[name]))
        val = self.constants[name]
        if expected_type is not None and not isinstance(val, expected_type):
            raise ValueError(
                "Expected a symbol {} of type {}, but got {}".format(
                    name, expected_type, self.types[name]
                )
            )
        return val


class NodeCanonicalizer(eve.NodeModifier):
    @classmethod
    def apply(cls, gt4py_ast: Computation):
        return cls().visit(gt4py_ast)

    def visit_SubscriptSingle(self, node: SubscriptSingle):
        # todo: do canonicalization properly. this should happen in the translation from python to gtscript
        return self.visit(SubscriptMultiple(value=node.value, indices=[Symbol(name=node.index)]))

    def visit_Computation(self, node: Computation):
        # canonicalize nested stencils
        stencils = []
        for stencil in node.stencils:
            if all(isinstance(body_node, Stencil) for body_node in stencil.body):
                # if we find a nested stencil flatten it
                for nested_stencil in stencil.body:
                    assert isinstance(nested_stencil, Stencil)
                    # todo: validate iteration_spec otherwise the TemporaryFieldDeclExtractor fails
                    flattened_stencil = Stencil(
                        iteration_spec=self.generic_visit(stencil.iteration_spec)
                        + self.generic_visit(nested_stencil.iteration_spec),
                        body=self.generic_visit(nested_stencil.body),
                    )
                    if any(isinstance(body_node, Stencil) for body_node in flattened_stencil.body):
                        raise ValueError("Nesting a stencil inside a nested stencil not allowed.")
                    stencils.append(flattened_stencil)
            elif not any(isinstance(body_node, Stencil) for body_node in stencil.body):
                # if we have a non-nested stencil just keep it as is
                stencils.append(self.visit(stencil))
            else:
                raise ValueError("Mixing nested and unnested stencils not allowed.")

        node.stencils = stencils

    def visit_Call(self, node: Call):
        # todo: this could be done by the call inliner
        # neighbor accessor canonicalization
        neighbor_selector_mapping = {
            "vertices": Symbol(name="Vertex"),  # todo: common.LocationType.Vertex,
            "edges": Symbol(name="Edge"),  # common.LocationType.Edge,
            "cells": Symbol(name="Cell"),  # common. LocationType.Cell
        }
        if node.func in neighbor_selector_mapping:
            return Call(
                func="neighbors",
                args=[self.generic_visit(node.args[0]), neighbor_selector_mapping[node.func]],
            )

        return self.generic_visit(node)


# poor mans variable declarations extractor
# todo: first type inference, than symbol table population?
class VarDeclExtractor(eve.NodeVisitor):
    """
    Extract all variable declarations and deduce their type.

     - Location - in location comprehensions and stencil iteration specifications
        `... for v in vertices(e)`
        `with Location(Vertex) as e: ...`
    - Field - in the stencils arguments
        `field_1: Field[Edge, dtype]`
        `field_2: Field[Edge, dtype]`
        `field_3: Field[Edge, dtype]`
    - Temporary Fields: - implicitly by assigning to a previously unknown variable
        `field_3 = field_1+field_2`
    """

    def __init__(self, symbol_table, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbol_table = symbol_table

    @classmethod
    def apply(cls, symbol_table, gt4py_ast: Computation):
        return cls(symbol_table).visit(gt4py_ast)

    def visit_LocationComprehension(self, node: LocationComprehension):
        assert node.iterator.func == "neighbors"
        assert isinstance(node.iterator.args[-1], Symbol)
        location_type = self.symbol_table.materialize_constant(node.iterator.args[-1].name)
        self.symbol_table[node.target.name] = Location[location_type]
        self.generic_visit(node.iterator)

    def visit_LocationSpecification(self, node: LocationSpecification):
        location_type = self.symbol_table.materialize_constant(node.location_type)
        self.symbol_table[node.name.name] = Location[location_type]

    # def visit_Call(self, node : Call):
    #    assert isinstance(self.symbol_table[node.func], Callable)
    #    return self.generic_visit(node)

    def visit_Argument(self, node: Argument):
        if node.name not in self.symbol_table:
            raise ValueError("Argument declarations need to be handled in the frontend.")


class TemporaryFieldDeclExtractor(eve.NodeVisitor):
    symbol_table: SymbolTable
    primary_location: Union[
        None, BuiltInTypeMeta
    ]  # todo: is there a way to tell mypy this is a Location

    # todo: enhance to support sparse dimension
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @classmethod
    def apply(cls, symbol_table, gt4py_ast: Computation):
        return cls(symbol_table).visit(gt4py_ast)

    def visit_LocationSpecification(self, node: LocationSpecification):
        assert self.primary_location is None
        self.primary_location = self.symbol_table[node.name.name]

    def visit_Assign(self, node: Assign):
        # extract target symbol
        if isinstance(node.target, SubscriptMultiple):
            target = node.target.value
        elif isinstance(node.target, Symbol):
            target = node.target
        assert isinstance(target, Symbol)

        if target.name not in self.symbol_table:
            assert self.primary_location is not None
            location_type = self.primary_location.args[0]
            self.symbol_table[target.name] = TemporaryField[
                location_type, self.symbol_table.materialize_constant("dtype")
            ]

    def visit_Stencil(self, node: Stencil):
        self.primary_location = None
        self.generic_visit(node)
        self.primary_location = None


class SymbolResolutionValidation(eve.NodeVisitor):
    """
    Ensure all occurring symbols are in the symbol table
    """

    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @classmethod
    def apply(cls, symbol_table, gt4py_ast: Computation):
        return cls(symbol_table).visit(gt4py_ast)

    def visit_Argument(self, node: Argument):
        # we don't visit any arguments, as their types (which are also symbols)
        #  are parsed in the frontend
        pass

    def visit_Symbol(self, node: Symbol):
        # every symbol not yet parsed must be supplied externally
        if node.name not in self.symbol_table:
            raise ValueError("Reference to undefined symbol `{}`".format(node.name))


class GTScriptToGTIR(eve.NodeTranslator):
    # todo: the current way of passing the location_stack is tidious and error prone

    def __init__(self, symbol_table, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # new scope so copy everything
        self.symbol_table = copy.deepcopy(symbol_table)

    @classmethod
    def apply(cls, symbol_table, gt4py_ast: Computation):
        return cls(symbol_table).visit(gt4py_ast)

    # def _get_location_type_by_symbol(self, location_stack, location):
    #    if not issubclass(self.symbol_table[location], Location):
    #        raise ValueError("`{}` is not a location.".format(location))
    #    for location_compr in location_stack:
    #        location_compr : gtir.LocationComprehension
    #        if location_compr.name == location:
    #            return location_compr.chain[-1]
    #
    #    raise ValueError("No location with name ``{} found.".format(location))

    def visit_IterationOrder(self, node: IterationOrder) -> gtir.common.LoopOrder:
        return gtir.common.LoopOrder[node.order]

    def visit_Interval(self, node: Interval):
        return None

    def visit_LocationSpecification(
        self, node: LocationSpecification, **kwargs
    ) -> gtir.LocationComprehension:
        loc_type = self.symbol_table.materialize_constant(
            node.location_type, expected_type=common.LocationType
        )
        return gtir.LocationComprehension(
            name=node.name.name, chain=gtir.NeighborChain(elements=[loc_type]), of=gtir.Domain()
        )

    def visit_LocationComprehension(
        self, node: LocationComprehension, *, location_stack, **kwargs
    ) -> gtir.LocationComprehension:
        if not node.iterator.func == "neighbors":
            raise ValueError(
                "Invalid neighbor specification. Expected a call to `neighbors`, but got `{}`".format(
                    node.iterator.func
                )
            )

        of = self.visit(node.iterator.args[0], location_stack=location_stack, **kwargs)
        if not isinstance(of, gtir.LocationRef):
            raise ValueError("Expected a `LocationRef` node, but got `{}`".format(type(of)))

        src_comprehension = location_stack[-1]
        assert src_comprehension.name == of.name
        elements = [src_comprehension.chain.elements[-1]]
        for loc_type in node.iterator.args[1:]:
            assert isinstance(loc_type, Symbol)
            elements.append(
                self.symbol_table.materialize_constant(
                    loc_type.name, expected_type=gtir.common.LocationType
                )
            )
        chain = gtir.NeighborChain(elements=elements)

        return gtir.LocationComprehension(name=node.target.name, chain=chain, of=of)

    def visit_Call(self, node: Call, *, location_stack, **kwargs):
        # todo: all of this can be done with the symbol table and the call inliner
        # reductions
        if node.func in _reduction_mapping:
            if not len(node.args):
                raise ValueError(
                    "Invalid number of arguments specified for function {}. Expected 1, but {} were given.".format(
                        node.func, len(node.args)
                    )
                )
            if not isinstance(node.args[0], Generator) or len(node.args[0].generators) != 1:
                raise ValueError("Invalid argument to {}".format(node.func))

            op = _reduction_mapping[node.func]
            neighbors = self.visit(
                node.args[0].generators[0], **{**kwargs, "location_stack": location_stack}
            )

            # operand gets new location stack
            new_location_stack = location_stack + [neighbors]

            operand = self.visit(
                node.args[0].elt, **{**kwargs, "location_stack": new_location_stack}
            )

            return gtir.NeighborReduce(
                op=op,
                operand=operand,
                neighbors=neighbors,
                location_type=location_stack[-1].chain.elements[-1],
            )

        raise ValueError()

    def visit_Constant(self, node: Constant, *, location_stack, **kwargs):
        py_dtype_to_eve = {  # todo: check
            int: common.DataType.INT32,
            float: common.DataType.FLOAT64,
        }
        return gtir.Literal(
            value=str(node.value),  # type: ignore
            vtype=py_dtype_to_eve[type(node.value)],
            location_type=location_stack[-1].chain.elements[-1],
        )

    def visit_Symbol(self, node: Symbol, *, location_stack):
        assert node.name in self.symbol_table
        if issubclass(self.symbol_table[node.name], Field) or issubclass(
            self.symbol_table[node.name], TemporaryField
        ):
            return gtir.FieldAccess(
                name=node.name,  # type: ignore
                location_type=location_stack[-1].chain.elements[-1],
                subscript=[gtir.LocationRef(name=location_stack[0].name)],
            )  # todo: just visit the subscript symbol
        elif issubclass(self.symbol_table[node.name], Location):
            return gtir.LocationRef(name=node.name)

        raise ValueError()

    def visit_SubscriptMultiple(self, node: SubscriptMultiple, *, location_stack):
        assert node.value.name in self.symbol_table
        if issubclass(self.symbol_table[node.value.name], Field) or issubclass(
            self.symbol_table[node.value.name], TemporaryField
        ):
            assert all(
                isinstance(index, Symbol) and issubclass(self.symbol_table[index.name], Location)
                for index in node.indices
            )
            # todo: just visit the index symbol
            return gtir.FieldAccess(
                name=node.value.name,  # type: ignore
                subscript=[
                    gtir.LocationRef(name=index.name) for index in cast(List[Symbol], node.indices)
                ],
                location_type=location_stack[-1].chain.elements[-1],
            )

        raise ValueError()

    def visit_Assign(self, node: Assign, *, location_stack, **kwargs) -> gtir.AssignStmt:
        return gtir.AssignStmt(
            left=self.visit(node.target, **{"location_stack": location_stack, **kwargs}),
            right=self.visit(node.value, **{"location_stack": location_stack, **kwargs}),
            location_type=location_stack[-1].chain.elements[-1],
        )

    def visit_BinaryOp(self, node: BinaryOp, location_stack, **kwargs):
        return gtir.BinaryOp(
            op=node.op,
            left=self.visit(node.left, **{"location_stack": location_stack, **kwargs}),
            right=self.visit(node.right, **{"location_stack": location_stack, **kwargs}),
            location_type=location_stack[-1].chain.elements[-1],
        )

    def visit_Stencil(self, node: Stencil, **kwargs) -> gtir.Stencil:
        loop_order, primary_location = None, None
        for it_spec in node.iteration_spec:
            if isinstance(it_spec, IterationOrder):
                assert loop_order is None
                loop_order = self.visit(it_spec)
            elif isinstance(it_spec, LocationSpecification):
                assert primary_location is None
                primary_location = self.visit(it_spec)
            elif isinstance(it_spec, Interval):
                # todo: implement
                pass
            else:
                raise ValueError()
        assert loop_order is not None
        assert primary_location is not None

        location_stack = [primary_location]  # todo: we should store dimensions here

        horizontal_loops = []
        for stmt in node.body:
            horizontal_loops.append(
                gtir.HorizontalLoop(
                    stmt=self.visit(stmt, location_stack=location_stack, **kwargs),
                    location=primary_location,
                )
            )

        return gtir.Stencil(
            vertical_loops=[
                gtir.VerticalLoop(loop_order=loop_order, horizontal_loops=horizontal_loops)
            ]
        )

    @staticmethod
    def _transform_field_type(name, field_type):
        assert issubclass(field_type, Field) or issubclass(field_type, TemporaryField)
        *location_types, vtype, = field_type.args

        assert isinstance(vtype, common.DataType)

        if len(location_types) == 1:
            assert isinstance(location_types[0], common.LocationType)
            horizontal_dim = gtir.HorizontalDimension(primary=location_types[0])
        elif len(location_types) == 2:
            assert isinstance(location_types[0], common.LocationType)
            assert issubclass(location_types[1], Local) and isinstance(
                location_types[1].args[0], common.LocationType
            )
            horizontal_dim = gtir.HorizontalDimension(
                primary=location_types[0],
                secondary=gtir.NeighborChain(elements=[location_types[1].args[0]]),
            )
        else:
            raise ValueError()

        return gtir.UField(
            name=name, vtype=vtype, dimensions=gtir.Dimensions(horizontal=horizontal_dim),
        )

    def visit_Computation(self, node: Computation) -> gtir.Computation:
        # parse arguments
        if not issubclass(self.symbol_table[node.arguments[0].name], Mesh):
            raise ValueError("First stencil argument must be a gtscript.Mesh")

        field_args = []
        for arg in node.arguments[1:]:
            field_args.append(self._transform_field_type(arg.name, self.symbol_table[arg.name]))

        # parse temporary fields
        temporary_field_decls = []
        for name, type_ in self.symbol_table.types.items():
            if issubclass(type_, TemporaryField):
                temporary_field_decls.append(self._transform_field_type(name, type_))

        return gtir.Computation(
            name=node.name,  # type: ignore
            params=field_args,
            stencils=self.visit(node.stencils),
            declarations=temporary_field_decls,
        )

    def visit_Pass(self, node: Pass):
        raise RuntimeError("Not implemented.")
