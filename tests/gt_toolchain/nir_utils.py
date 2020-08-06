# -*- coding: utf-8 -*-
from typing import List

from eve import Bool, Str
from gt_toolchain import common
from gt_toolchain.unstructured import nir


default_vtype = common.DataType.FLOAT32
default_location = common.LocationType.Vertex


def make_vertical_loop(horizontal_loops):
    return nir.VerticalLoop(horizontal_loops=horizontal_loops, loop_order=common.LoopOrder.FORWARD)


def make_block_stmt(stmts: List[nir.Stmt], declarations: List[nir.LocalVar]):
    return nir.BlockStmt(
        location_type=stmts[0].location_type if len(stmts) > 0 else common.LocationType.Vertex,
        statements=stmts,
        declarations=declarations,
    )


def make_horizontal_loop(block: nir.BlockStmt):
    return nir.HorizontalLoop(stmt=block, location_type=block.location_type)


def make_empty_block_stmt(location_type: common.LocationType):
    return nir.BlockStmt(location_type=location_type, declarations=[], statements=[])


def make_empty_horizontal_loop(location_type: common.LocationType):
    return make_horizontal_loop(make_empty_block_stmt(location_type))


# TODO share with dependency graph
# write = read
def make_horizontal_loop_with_copy(write: Str, read: Str, read_has_extent: Bool):
    loc_type = common.LocationType.Vertex
    write_access = nir.FieldAccess(name=write, extent=False, location_type=loc_type)
    read_access = nir.FieldAccess(name=read, extent=read_has_extent, location_type=loc_type)

    return (
        nir.HorizontalLoop(
            stmt=nir.BlockStmt(
                declarations=[], statements=[nir.AssignStmt(left=write_access, right=read_access)],
            ),
            location_type=loc_type,
        ),
        write_access,
        read_access,
    )


def make_local_var(name: Str):
    return nir.LocalVar(name=name, vtype=default_vtype, location_type=default_location)


def make_init(field: Str):
    write_access = nir.FieldAccess(name=field, extent=False, location_type=default_location)
    return (
        nir.AssignStmt(
            left=write_access,
            right=nir.Literal(
                value=common.BuiltInLiteral.ONE,
                vtype=default_vtype,
                location_type=default_location,
            ),
        ),
        write_access,
    )


def make_horizontal_loop_with_init(field: Str):
    write_access = nir.FieldAccess(name=field, extent=False, location_type=default_location)
    return (
        nir.HorizontalLoop(
            stmt=nir.BlockStmt(
                declarations=[],
                statements=[
                    nir.AssignStmt(
                        left=write_access,
                        right=nir.Literal(
                            value=common.BuiltInLiteral.ONE,
                            vtype=default_vtype,
                            location_type=default_location,
                        ),
                    )
                ],
            ),
            location_type=default_location,
        ),
        write_access,
    )
