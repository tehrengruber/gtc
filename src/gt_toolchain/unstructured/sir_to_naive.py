# -*- coding: utf-8 -*-
# Eve toolchain

import eve  # noqa: F401
from eve.core import NodeTranslator, Node
from gt_toolchain.unstructured import sir, common, naive


def sir2naiveLocationType(sir_loc: sir.LocationType):
    if sir_loc == sir.LocationType.Cell:
        return naive.LocationType.Face
    elif sir_loc == sir.LocationType.Edge:
        return naive.LocationType.Edge
    elif sir_loc == sir.LocationType.Vertex:
        return naive.LocationType.Node
    else:
        raise "unreachable"


class SirToNaive(NodeTranslator):
    def __init__(self, *, memo: dict = None, **kwargs):
        super().__init__(memo=memo)
        self.isControlFlow = None
        self.sir_stencil_params = {}  # TODO this is a dummy symbol table for stencil parameters

    def _get_field_location_type(self, field: sir.Field):
        # TODO handle sparse location types
        return sir2naiveLocationType(
            field.field_dimensions.horizontal_dimension.dense_location_type
        )

    def visit_Field(self, node: Node, **kwargs):
        return naive.UnstructuredField(
            name=node.name,
            location_type=sir2naiveLocationType(
                node.field_dimensions.horizontal_dimension.dense_location_type
            ),
            data_type=common.DataType.FLOAT64,
        )

    def visit_Stencil(self, node: Node, **kwargs):
        # Needs to run before the body/ast as it creates a dummy symbol table for stencil parameters
        params = []
        for f in node.params:
            self.sir_stencil_params[f.name] = f
            params.append(self.visit(f))

        k_loops = self.visit(node.ast)
        return naive.Computation(
            params=params, stencils=[naive.Stencil(name=node.name, k_loops=k_loops)]
        )

    def visit_VerticalRegion(self, node: Node, **kwargs):
        # TODO don't ignore interval
        horizontal_loops = [self.visit(node.ast)]  # TODO
        return [naive.ForK(loop_order=node.loop_order, horizontal_loops=horizontal_loops)]

    def visit_VerticalRegionDeclStmt(self, node: Node, **kwargs):
        return self.visit(node.vertical_region)

    def visit_FieldAccessExpr(self, node: Node, **kwargs):
        horizontal_offset = False  # TODO
        return naive.FieldAccessExpr(
            name=node.name,
            offset=(horizontal_offset, node.vertical_offset),
            location_type=self._get_field_location_type(self.sir_stencil_params[node.name]),
        )

    def visit_AssignmentExpr(self, node: Node, **kwargs):
        assert node.op == "="
        return naive.AssignmentExpr(left=self.visit(node.left), right=self.visit(node.right))

    def visit_ExprStmt(self, node: Node, **kwargs):
        return naive.ExprStmt(expr=self.visit(node.expr))

    def visit_BlockStmt(self, node: Node, **kwargs):
        if self.isControlFlow:
            for s in node.statements:
                assert isinstance(s, sir.VerticalRegionDeclStmt)
                return self.visit(s)
        else:
            statements = []
            for s in node.statements:
                print(s)
                res = self.visit(s)
                print(res)
                statements.append(res)
            return naive.BlockStmt(statements=statements)

    def visit_AST(self, node: Node, **kwargs):
        assert isinstance(node.root, sir.BlockStmt)  # TODO add check to IR
        if self.isControlFlow is None:
            self.isControlFlow = True
            return self.visit(node.root)
        elif self.isControlFlow is True:
            self.isControlFlow = False
            return naive.HorizontalLoop(ast=self.visit(node.root))
        else:
            raise "unreachable: there should not be an AST node in the stencil ast"
