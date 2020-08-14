# -*- coding: utf-8 -*-
import copy
import enum
import inspect
import typing

from gt_frontend.gtscript import *

import eve
import gtc.unstructured.gtir as gtir


# poor mans variable declarations extractor
# todo: first type inference, than symbol table population?
class VarDeclExtractor(eve.NodeVisitor):
    def __init__(self, symbol_table, externals, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbol_table = symbol_table
        self.externals = externals

    def visit_LocationComprehension(self, node: LocationComprehension):
        self.symbol_table[node.target.id] = Location
        self.visit(node.iter)

    def visit_LocationSpecification(self, node: LocationSpecification):
        self.symbol_table[node.name.id] = Location

    # def visit_Call(self, node : Call):
    #    assert isinstance(self.symbol_table[node.func], Callable)
    #    return self.generic_visit(node)

    def visit_Argument(self, node: Argument):
        if not node.name in self.symbol_table:
            raise ValueError("Argument declarations need to be handled in the frontend.")

    # def visit_Assign(self, node : Assign):
    #    pass
    #    #if not node.target in self.symbol_table:
    #    #    self.symbol_table[node.target] = TemporaryField[dtype, location_type]

    def visit_Name(self, node: Name):
        # every symbol not yet parsed must be supplied externally
        assert node.id in self.symbol_table


class CallCanonicalizer(eve.NodeModifier):
    def visit_Call(self, node: Call):
        # todo: this could be done by the call inliner
        # neighbor accessor canonicalization
        neighbor_selector_mapping = {
            "vertices": Name(id="Vertex"),  # todo: common.LocationType.Vertex,
            "edges": Name(id="Vertex"),  # common.LocationType.Edge,
            "cells": Name(id="Cell"),  # common. LocationType.Cell
        }
        if node.func in neighbor_selector_mapping:
            return Call(
                func="neighbors",
                args=[self.visit(node.args[0]), neighbor_selector_mapping[node.func]],
            )

        return self.generic_visit(node)


# Types
#  - BuiltInType
#  - LocationType
#  - DataType

class GTScriptToGTIR(eve.NodeTranslator):
    def __init__(self, symbol_table, constants, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # new scope so copy everything
        # todo: externals should be read-only to avoid potentially costly copy
        self.symbol_table = copy.deepcopy(symbol_table)
        self.constants = copy.deepcopy(constants)

    def _materialize_constant(self, name, expected_type=None):
        """
        Materialize constant symbol with name `name`, i.e. return the value of that symbol. Currently the only constants
        are types.

        Example:
        ```
        self._materialize_constant("Vertex") == LocationType.Vertex
        ```
        """
        if not name in self.symbol_table:
            raise ValueError("Symbol {} not found".format(name))
        if not name in self.constants:
            raise ValueError("Symbol {} : {} is not a constant".format(name, self.symbol_table[name]))
        val = self.constants[name]
        if expected_type != None and not isinstance(val, expected_type):
            raise ValueError(
                "Expected a symbol {} of type {}, but got {}".format(name, expected_type, self.symbol_table[name]))
        return val

    def visit_IterationOrder(self, node: IterationOrder) -> gtir.common.LoopOrder:
        return gtir.common.LoopOrder[node.order]

    def visit_Interval(self, node: Interval):
        return None

    def visit_LocationSpecification(self, node: LocationSpecification, **kwargs):
        loc_type = self._materialize_constant(node.location_type, expected_type=common.LocationType)
        return gtir.LocationComprehension(
            name=node.name.id, chain=gtir.NeighborChain(elements=[loc_type]), of=gtir.Domain()
        )

    def visit_LocationComprehension(self, node: LocationComprehension, **kwargs):
        assert node.iter.func == "neighbors"

        of = self.visit(node.iter.args[0], **kwargs)
        assert isinstance(of, gtir.LocationRef)
        src_comprehension = kwargs["location_stack"][-1]
        assert src_comprehension.name == of.name

        elements = [src_comprehension.chain.elements[-1]] + [
            self._materialize_constant(loc_type.id, expected_type=gtir.common.LocationType)
            for loc_type in node.iter.args[1:]
        ]

        chain = gtir.NeighborChain(elements=elements)

        return gtir.LocationComprehension(name=node.target.id, chain=chain, of=of)

    def visit_Call(self, node: Call, **kwargs):
        location_stack = kwargs["location_stack"]

        # todo: all of this can be done with the symbol table and the call inliner
        # reductions
        reduction_mapping = {
            "sum": gtir.ReduceOperator.ADD,
            "product": gtir.ReduceOperator.MUL,
            "min": gtir.ReduceOperator.MIN,
            "max": gtir.ReduceOperator.MAX,
        }
        if node.func in reduction_mapping:
            if not len(node.args):
                raise ValueError(
                    "Invalid number of arguments specified for function {}. Expected 1, but {} were given.".format(
                        node.func, len(node.args)))
            if not isinstance(node.args[0], Generator) or len(node.args[0].generators) != 1:
                raise ValueError("Invalid argument to {}".format(node.func))

            op = reduction_mapping[node.func]
            neighbors = self.visit(node.args[0].generators[0], **kwargs)

            # operand gets new location stack
            new_location_stack = location_stack + [neighbors]

            operand = self.visit(node.args[0].elt, **{**kwargs, "location_stack": new_location_stack})

            return gtir.NeighborReduce(
                op=op,
                operand=operand,
                neighbors=neighbors,
                location_type=location_stack[-1].chain.elements[-1],
            )

        raise ValueError()

    def visit_Name(self, node: Name, location_stack):
        assert node.id in self.symbol_table
        if issubclass(self.symbol_table[node.id], Field):
            return gtir.FieldAccess(
                name=node.id,
                location_type=location_stack[-1].chain.elements[-1],
                subscript=[gtir.LocationRef(name=location_stack[0].name)],
            )  # todo: just visit the subscript symbol
        elif issubclass(self.symbol_table[node.id], Location):
            return gtir.LocationRef(name=node.id)

        raise ValueError()

    def visit_SubscriptSingle(self, node: SubscriptSingle, location_stack):
        # todo: do cannonicalization properly
        return self.visit(SubscriptMultiple(name=node.value, indices=Name(id=node.index)), location_stack)

    def visit_SubscriptMultiple(self, node: SubscriptMultiple, location_stack):
        assert node.value.id in self.symbol_table
        if issubclass(self.symbol_table[node.value.id], Field):
            assert all(issubclass(self.symbol_table[index.id], Location) for index in node.indices)
            # todo: just visit the index symbol
            return gtir.FieldAccess(
                name=node.value.id,
                subscript=[gtir.LocationRef(name=index.id) for index in node.indices],
                location_type=location_stack[-1].chain.elements[-1],
            )

        raise ValueError()

    def visit_Assign(self, node: Assign, **kwargs):
        return gtir.AssignStmt(
            left=self.visit(node.target, **kwargs), right=self.visit(node.value, **kwargs)
        )

    def visit_Stencil(self, node: Stencil, **kwargs):
        loop_order, primary_location = None, None
        for it_spec in node.iteration_spec:
            if isinstance(it_spec, IterationOrder):
                loop_order = self.visit(it_spec)
            elif isinstance(it_spec, LocationSpecification):
                primary_location = self.visit(it_spec)
            elif isinstance(it_spec, Interval):
                # todo: implement
                pass
            else:
                raise ValueError()
        assert loop_order != None
        assert primary_location != None

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
            ],
            declarations=[],
        )

    def visit_Computation(self, node: Computation):
        # parse arguments
        assert issubclass(self.symbol_table[node.arguments[0].name], Mesh)
        field_args = []
        for arg in node.arguments[1:]:
            arg_type = self.symbol_table[arg.name]
            assert issubclass(arg_type, Field)
            *location_types, vtype, = arg_type.args

            assert all(isinstance(loc_type, common.LocationType) for loc_type in location_types)
            assert isinstance(vtype, common.DataType)

            if len(location_types) == 1:
                horizontal_dim = gtir.HorizontalDimension(primary=location_types[0])
            elif len(location_types) == 2:
                horizontal_dim = gtir.HorizontalDimension(primary=location_types[0],
                                                          secondary=gtir.NeighborChain(elements=[location_types[1]]))
            else:
                raise ValueError()

            field_args.append(
                gtir.UField(
                    name=arg.name,
                    vtype=vtype,
                    dimensions=gtir.Dimensions(
                        horizontal=horizontal_dim
                    ),
                )
            )

        return gtir.Computation(
            name=node.name, params=field_args, stencils=self.visit(node.stencils)
        )
