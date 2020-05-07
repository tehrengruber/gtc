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


from typing import List

from devtools import debug

from eve import Node, Str


# from pydantic import root_validator


# Not part of Loop
class DummyNode(Node):
    desc: Str  # just describe what the node is supposed to do


class DummyAssign(Node):
    lhs: Str
    rhs: Node


# Some expression with vtype index (aka int)
class IndexExpr(Node):
    name: Str  # let's say the index is some variable with a name (or ssaid)


class AnyRegion(Node):
    arguments: List[Node]
    body: List[Node]
    pass


class SizedRegion1(AnyRegion):
    pass


class Condition(Node):
    expr: Node
    pass


# Loop


# let arguments = (ins Index:$lowerBound,
#                      Index:$upperBound,
#                      Index:$step,
#                      Variadic<AnyType>:$initArgs);
# let results = (outs Variadic<AnyType>:$results);
# let regions = (region SizedRegion<1>:$region);
class ForOp(Node):
    lowerBound: IndexExpr
    upperBound: IndexExpr
    step: IndexExpr
    initArgs: List[Node]
    region: SizedRegion1


# let arguments = (ins I1:$condition);
# let results = (outs Variadic<AnyType>:$results);
# let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
class IfOp(Node):
    I1: Condition
    thenRegion: SizedRegion1
    elseRegion: AnyRegion


# let arguments = (ins Variadic<Index>:$lowerBound,
#                      Variadic<Index>:$upperBound,
#                      Variadic<Index>:$step,
#                      Variadic<AnyType>:$initVals);
# let results = (outs Variadic<AnyType>:$results);
# let regions = (region SizedRegion<1>:$region);
class ParallelOp(Node):
    lowerBound: List[IndexExpr]
    upperBound: List[IndexExpr]
    step: List[IndexExpr]
    initArgs: List[Node]
    region: SizedRegion1


# let arguments = (ins AnyType:$operand);
# let regions = (region SizedRegion<1>:$reductionOperator);
class ReduceOp(Node):
    operand: Node
    reductionOperator: SizedRegion1
    pass


class ReduceReturnOp(Node):
    result: Node


class YieldOp(Node):
    results: List[Node]


# Examples

# ```mlir
# func @reduce(%buffer: memref<1024xf32>, %lb: index,
#              %ub: index, %step: index) -> (f32) {
#   // Initial sum set to 0.
#   %sum_0 = constant 0.0 : f32
#   // iter_args binds initial values to the loop's region arguments.
#   %sum = loop.for %iv = %lb to %ub step %step
#       iter_args(%sum_iter = %sum_0) -> (f32) {
#     %t = load %buffer[%iv] : memref<1024xf32>
#     %sum_next = addf %sum_iter, %t : f32
#     // Yield current iteration sum to next iteration %sum_iter or to %sum
#     // if final iteration.
#     loop.yield %sum_next : f32
#   }
#   return %sum : f32
# }
# ```
example_for = [
    DummyNode(desc="sum_0 = 0.0"),
    DummyAssign(
        lhs="sum",
        rhs=ForOp(
            lowerBound=IndexExpr(name="%iv = %lb"),
            upperBound=IndexExpr(name="ub"),
            step=IndexExpr(name="step"),
            initArgs=[DummyNode(desc="sum_iter = sum_0")],
            region=SizedRegion1(
                arguments=[],
                body=[
                    DummyNode(desc="%t = load %buffer[%iv]"),
                    DummyNode(desc=" %sum_next = addf %sum_iter, %t"),
                    YieldOp(results=[DummyNode(desc="sum_next")]),
                ],
            ),
        ),
    ),
    DummyNode(desc="return %sum : f32"),
]
debug(example_for)

# ```mlir
# %x, %y = loop.if %b -> (f32, f32) {
#   %x_true = ...
#   %y_true = ...
#   loop.yield %x_true, %y_true : f32, f32
# } else {
#   %x_false = ...
#   %y_false = ...
#   loop.yield %x_false, %y_false : f32, f32
# }
# ```
example_if = IfOp(
    I1=Condition(expr=Node()),
    thenRegion=SizedRegion1(
        arguments=[],
        body=[
            DummyNode(desc="x_true=..."),
            DummyNode(desc="y_true=..."),
            YieldOp(results=[DummyNode(desc="x_true"), DummyNode(desc="y_true")]),
        ],
    ),
    elseRegion=AnyRegion(
        arguments=[],
        body=[
            DummyNode(desc="x_false=..."),
            DummyNode(desc="y_false=..."),
            YieldOp(results=[DummyNode(desc="x_false"), DummyNode(desc="y_false")]),
        ],
    ),
)
debug(example_if)

# ```mlir
# loop.reduce(%operand) : f32 {
#   ^bb0(%lhs : f32, %rhs: f32):
#     %res = addf %lhs, %rhs : f32
#     loop.reduce.return %res : f32
# }
# ```
reduce_op = ReduceOp(
    operand=DummyNode(desc="elem_to_reduce"),
    reductionOperator=SizedRegion1(
        arguments=[DummyNode(desc="lhs"), DummyNode(desc="rhs")],
        body=[DummyNode(desc="res=lhs+rhs"), ReduceReturnOp(result=DummyNode(desc="res"))],
    ),
)

# ```mlir
# %init = constant 0.0 : f32
# loop.parallel (%iv) = (%lb) to (%ub) step (%step) init (%init) -> f32 {
#   %elem_to_reduce = load %buffer[%iv] : memref<100xf32>
#   loop.reduce(%elem_to_reduce) : f32 {
#     ^bb0(%lhs : f32, %rhs: f32):
#       %res = addf %lhs, %rhs : f32
#       loop.reduce.return %res : f32
#   }
# }
# ```
example_parallel = [
    DummyNode(desc="init = 0.0"),
    ParallelOp(
        lowerBound=[IndexExpr(name="lb")],
        upperBound=[IndexExpr(name="ub")],
        step=[IndexExpr(name="step")],
        initArgs=[DummyNode(desc="init")],
        region=SizedRegion1(arguments=[], body=[DummyNode(desc="elem_to_reduce"), reduce_op]),
    ),
]
debug(example_parallel)
