# -*- coding: utf-8 -*-
# Eve toolchain

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from eve.core import Node, NodeTranslator
from gt_toolchain.unstructured import common, naive, sir


def sir2naiveLocationType(sir_loc: sir.LocationType):
    if sir_loc == sir.LocationType.Cell:
        return naive.LocationType.Face
    elif sir_loc == sir.LocationType.Edge:
        return naive.LocationType.Edge
    elif sir_loc == sir.LocationType.Vertex:
        return naive.LocationType.Node
    else:
        raise "unreachable"


def opstring2enum(op: str):
    if op == "+":
        return common.BinaryOperator.ADD
    elif op == "-":
        return common.BinaryOperator.SUB
    elif op == "*":
        return common.BinaryOperator.MUL
    elif op == "/":
        return common.BinaryOperator.DIV
    else:
        raise "unreachable: unknown op"


class SirToNaive(NodeTranslator):
    def __init__(self, *, memo: dict = None, **kwargs):
        super().__init__(memo=memo)
        self.isControlFlow = None
        self.sir_stencil_params = (
            {}
        )  # elements are sir.Field # TODO this is a dummy symbol table for stencil parameters
        self.current_loc_type_stack = []  # TODO experimental

    def _get_field_location_type(self, field: sir.Field):
        if field.field_dimensions.horizontal_dimension.sparse_part:
            return sir2naiveLocationType(field.field_dimensions.horizontal_dimension.sparse_part[0])
        return sir2naiveLocationType(
            field.field_dimensions.horizontal_dimension.dense_location_type
        )

    def visit_Field(self, node: Node, **kwargs):
        assert (not node.field_dimensions.horizontal_dimension.sparse_part) or (
            len(node.field_dimensions.horizontal_dimension.sparse_part) <= 1
        )
        sparse_location_type = None
        if node.field_dimensions.horizontal_dimension.sparse_part:
            sparse_location_type = sir2naiveLocationType(
                node.field_dimensions.horizontal_dimension.sparse_part[0]
            )
        return naive.UnstructuredField(
            name=node.name,
            location_type=sir2naiveLocationType(
                node.field_dimensions.horizontal_dimension.dense_location_type
            ),
            sparse_location_type=sparse_location_type,
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
        horizontal_loops = self.visit(node.ast)
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

    def visit_VarAccessExpr(self, node: Node, **kwargs):
        loctype = ""
        if node.location_type:
            loctype = sir2naiveLocationType(node.location_type)
        elif self.current_loc_type_stack:
            loctype = self.current_loc_type_stack[-1]
        else:
            raise ValueError("no location type")

        return naive.VarAccessExpr(name=node.name, location_type=loctype)

    def visit_BlockStmt(self, node: Node, **kwargs):
        if self.isControlFlow:
            for s in node.statements:
                assert isinstance(s, sir.VerticalRegionDeclStmt)
                return self.visit(s)
        else:
            horizontal_loops = []
            statements = []
            cur_loc_type = None
            # TODO this is a hack and violating the new parallel model!
            for s in node.statements:
                transformed = self.visit(s)
                if transformed.location_type != cur_loc_type:
                    if statements:  # we already have some statements -> make horizontal loop
                        horizontal_loops.append(
                            naive.HorizontalLoop(ast=naive.BlockStmt(statements=statements))
                        )
                        statements = []
                    cur_loc_type = transformed.location_type
                statements.append(transformed)
            if statements:
                horizontal_loops.append(
                    naive.HorizontalLoop(ast=naive.BlockStmt(statements=statements))
                )
            return horizontal_loops

    def visit_BinaryOperator(self, node: Node, **kwargs):
        return naive.BinaryOp(
            op=opstring2enum(node.op), left=self.visit(node.left), right=self.visit(node.right)
        )

    def visit_VarDeclStmt(self, node: Node, **kwargs):
        assert node.op == "="
        assert node.dimension == 0
        assert len(node.init_list) == 1
        assert isinstance(node.data_type.data_type, sir.BuiltinType)

        loctype = ""
        if node.location_type:
            loctype = sir2naiveLocationType(node.location_type)
        elif self.current_loc_type_stack:
            loctype = self.current_loc_type_stack[-1]
        else:
            raise ValueError("no location type")

        init = self.visit(node.init_list[0])
        return naive.VarDeclStmt(
            data_type=node.data_type.data_type.type_id,
            name=node.name,
            init=init,
            location_type=loctype,
        )

    def visit_LiteralAccessExpr(self, node: Node, **kwargs):
        loctype = ""
        if node.location_type:
            loctype = sir2naiveLocationType(node.location_type)
        elif self.current_loc_type_stack:
            loctype = self.current_loc_type_stack[-1]
        else:
            raise ValueError("no location type")

        return naive.LiteralExpr(
            value=node.value, data_type=node.data_type.type_id, location_type=loctype,
        )

    def visit_ReductionOverNeighborExpr(self, node: Node, **kwargs):
        self.current_loc_type_stack.append(sir2naiveLocationType(node.chain[-1]))
        right = self.visit(node.rhs)
        init = self.visit(node.init)
        self.current_loc_type_stack.pop()
        return naive.ReduceOverNeighbourExpr(
            operation=opstring2enum(node.op),
            right=right,
            init=init,
            location_type=sir2naiveLocationType(node.chain[0]),
        )

    def visit_AST(self, node: Node, **kwargs):
        assert isinstance(node.root, sir.BlockStmt)  # TODO add check to IR
        if self.isControlFlow is None:
            self.isControlFlow = True
            return self.visit(node.root)
        elif self.isControlFlow is True:
            self.isControlFlow = False
            return self.visit(node.root)
        else:
            raise "unreachable: there should not be an AST node in the stencil ast"
