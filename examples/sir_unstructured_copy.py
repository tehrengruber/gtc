# -*- coding: utf-8 -*-
# Eve toolchain

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gt_toolchain.unstructured import sir, common, naive, naive_codegen

from eve.core import NodeTranslator, Node, NOTHING

field_acc_a = sir.FieldAccessExpr(
    name="field_a", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
)
field_acc_b = sir.FieldAccessExpr(
    name="field_b", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
)

assign_expr = sir.AssignmentExpr(left=field_acc_a, op="=", right=field_acc_b)
assign_expr_stmt = sir.ExprStmt(expr=assign_expr)
root = sir.BlockStmt(statements=[assign_expr_stmt])
ast = sir.AST(root=root)

vert_decl_stmt = sir.VerticalRegionDeclStmt(
    vertical_region=sir.VerticalRegion(
        ast=ast, interval=sir.Interval(), loop_order=common.LoopOrder.FORWARD
    )
)
ctrl_flow_ast = sir.AST(root=sir.BlockStmt(statements=[vert_decl_stmt]))

field_a = sir.Field(
    name="field_a",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Cell)
    ),
)
field_b = sir.Field(
    name="field_b",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Cell)
    ),
)

stencil = sir.Stencil(name="copy", ast=ctrl_flow_ast, params=[field_a, field_b])

debug(stencil)


def sir2naiveLocationType(sir_loc: sir.LocationType):
    if sir_loc == sir.LocationType.Cell:
        return naive.LocationType.Face
    elif sir_loc == sir.LocationType.Edge:
        return naive.LocationType.Edge
    elif sir_loc == sir.LocationType.Vertex:
        return naive.LocationType.Node
    else:
        raise "unreachable"


class DummyPass(NodeTranslator):
    def __init__(self, *, memo: dict = None, **kwargs):
        super().__init__(memo=memo)
        self.isControlFlow = None

    def visit_Field(self, node: Node, **kwargs):
        return naive.UnstructuredField(
            name=node.name,
            location_type=sir2naiveLocationType(
                node.field_dimensions.horizontal_dimension.dense_location_type
            ),
            data_type=common.DataType.FLOAT64,
        )

    def visit_Stencil(self, node: Node, **kwargs):
        params = []
        for f in node.params:
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
            location_type=naive.LocationType.Face,
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
            # TODO check node.root is a BlockStmt
            return self.visit(node.root)
        elif self.isControlFlow is True:
            self.isControlFlow = False
            return naive.HorizontalLoop(ast=self.visit(node.root))
        else:
            raise "unreachable: there should not be a AST node in the stencil ast"


dpass = DummyPass()
# dpass.isControlFlow = True
tmp = dpass.visit(stencil)
debug(tmp)

print(naive_codegen.NaiveCodeGenerator.apply(tmp))
