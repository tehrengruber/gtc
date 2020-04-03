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


import enum
import random
import string
from typing import Collection, Dict, List, Mapping, Optional, Sequence, Set, Type, TypeVar

from pydantic import Field, validator  # noqa: F401

from eve.core import (  # type: ignore
    Bool,
    Bytes,
    Float,
    Int,
    Str,
    InmutableNode,
    MutableNode,
    SourceLocation,
    StrEnum,
)


T = TypeVar("T")
S = TypeVar("S")


class Factories:
    STR_LEN = 6

    @classmethod
    def make_bool(cls) -> bool:
        return True

    @classmethod
    def make_int(cls) -> int:
        return 1

    @classmethod
    def make_neg_int(cls) -> int:
        return -2

    @classmethod
    def make_pos_int(cls) -> int:
        return 2

    @classmethod
    def make_float(cls) -> float:
        return 1.1

    @classmethod
    def make_str(cls, length: Optional[int] = None) -> str:
        length = length or cls.STR_LEN
        return string.ascii_letters[:length]

    @classmethod
    def make_member(cls, values: Sequence[T]) -> T:
        return values[0]

    @classmethod
    def make_collection(
        cls,
        item_type: Type[T],
        collection_type: Type[Collection[T]] = list,
        length: Optional[int] = None,
    ) -> Collection[T]:
        length = length or cls.STR_LEN

        maker_attr_name = f"make_{item_type.__name__}"
        if hasattr(cls, maker_attr_name):
            maker = getattr(cls, maker_attr_name)
        else:

            def maker():
                return item_type()

        return collection_type([maker() for _ in range(length)])  # type: ignore

    @classmethod
    def make_mapping(
        cls,
        key_type: Type[S],
        value_type: Type[T],
        mapping_type: Type[Mapping[S, T]] = dict,
        length: Optional[int] = None,
    ) -> Mapping[S, T]:
        length = length or cls.STR_LEN

        key_maker_attr_name = f"make_{key_type.__name__}"
        if hasattr(cls, key_maker_attr_name):
            key_maker = getattr(cls, key_maker_attr_name)
        else:

            def key_maker():
                return key_type()

        value_maker_attr_name = f"make_{value_type.__name__}"
        if hasattr(cls, value_maker_attr_name):
            value_maker = getattr(cls, value_maker_attr_name)
        else:

            def value_maker():
                return value_type()

        return mapping_type({key_maker(): value_maker() for _ in range(length)})  # type: ignore


class RandomFactories(Factories):
    MIN_INT = -9999
    MAX_INT = 9999
    MIN_FLOAT = -999.0
    MAX_FLOAT = 999.09

    @classmethod
    def make_bool(cls) -> bool:
        return random.choice([True, False])

    @classmethod
    def make_int(cls) -> int:
        return random.randint(cls.MIN_INT, cls.MAX_INT)

    @classmethod
    def make_neg_int(cls) -> int:
        return random.randint(cls.MIN_INT, 1)

    @classmethod
    def make_pos_int(cls) -> int:
        return random.randint(1, cls.MAX_INT)

    @classmethod
    def make_float(cls) -> float:
        return cls.MIN_FLOAT + random.random() * (cls.MAX_FLOAT - cls.MIN_FLOAT)

    @classmethod
    def make_str(cls, length: Optional[int] = None) -> str:
        length = length or cls.STR_LEN
        return "".join(random.choice(string.ascii_letters) for _ in range(length))

    @classmethod
    def make_member(cls, values: Sequence[T]) -> T:
        return random.choice(values)


@enum.unique
class StrKind(StrEnum):
    """Sample string Enum."""

    FOO = "foo"
    BLA = "bla"
    FIZ = "fiz"
    FUZ = "fuz"


@enum.unique
class IntKind(enum.IntEnum):
    """Sample int Enum."""

    MINUS = -1
    ZERO = 0
    PLUS = 1


class EmptyNode(InmutableNode):
    pass


class LocationNode(InmutableNode):
    loc: SourceLocation


class SimpleNode(InmutableNode):
    bool_value: Bool
    int_value: Int
    float_value: Float
    str_value: Str
    bytes_value: Bytes
    int_kind: IntKind
    str_kind: StrKind


class SimpleNodeWithOptionals(InmutableNode):
    int_value: Optional[Int]
    float_value: Optional[Float]
    str_value: Optional[Str]


class SimpleNodeWithLoc(InmutableNode):
    int_value: Int
    float_value: Float
    str_value: Str
    loc: Optional[SourceLocation]


class SimpleNodeWithCollections(InmutableNode):
    int_list: List[Int]
    str_set: Set[Str]
    str_to_int_dict: Dict[Str, Int]
    loc: Optional[SourceLocation]


class SimpleNodeWithAbstractCollections(InmutableNode):
    int_sequence: Sequence[Int]
    str_set: Set[Str]
    str_to_int_mapping: Mapping[Str, Int]
    loc: Optional[SourceLocation]


class CompoundNode(InmutableNode):
    location: LocationNode
    simple: SimpleNode
    simple_loc: SimpleNodeWithLoc
    simple_opt: SimpleNodeWithOptionals
    other_simple_opt: Optional[SimpleNodeWithOptionals]


class CompoundNodeWithCollections(InmutableNode):
    simple_col: Optional[SimpleNodeWithCollections]
    simple_abscol: Optional[SimpleNodeWithAbstractCollections]
    compound_col: Sequence[CompoundNode]
    compound_all: Optional[Mapping[Str, CompoundNode]]


class MutableSimpleNode(MutableNode):
    bool_value: Bool
    int_value: Int
    float_value: Float
    str_value: Str
    int_kind: IntKind
    str_kind: StrKind


class MutableCompoundNode(MutableNode):
    location: LocationNode
    simple: SimpleNode
    simple_loc: SimpleNodeWithLoc
    simple_opt: SimpleNodeWithOptionals
    other_simple_opt: Optional[SimpleNodeWithOptionals]


# -- Maker functions --
def source_location_maker(randomize: bool = True) -> SourceLocation:
    factories = RandomFactories if randomize else Factories
    line = factories.make_pos_int()
    column = factories.make_pos_int()
    str_value = factories.make_str()
    source = f"file_{str_value}.py"

    return SourceLocation(line=line, column=column, source=source)


def location_node_maker(randomize: bool = True) -> LocationNode:
    return LocationNode(loc=source_location_maker(randomize))


def simple_node_maker(randomize: bool = True) -> SimpleNode:
    factories = RandomFactories if randomize else Factories
    bool_value = factories.make_bool()
    int_value = factories.make_int()
    float_value = factories.make_float()
    str_value = factories.make_str()
    bytes_value = factories.make_str().encode()
    int_kind = factories.make_member([*IntKind]) if randomize else IntKind.PLUS
    str_kind = factories.make_member([*StrKind]) if randomize else StrKind.BLA

    return SimpleNode(
        bool_value=bool_value,
        int_value=int_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def simple_node_with_optionals_maker(randomize: bool = True) -> SimpleNodeWithOptionals:
    factories = RandomFactories if randomize else Factories
    int_value = factories.make_int()
    float_value = factories.make_float()

    return SimpleNodeWithOptionals(int_value=int_value, float_value=float_value)


def simple_node_with_loc_maker(randomize: bool = True) -> SimpleNodeWithLoc:
    factories = RandomFactories if randomize else Factories
    int_value = factories.make_int()
    float_value = factories.make_float()
    str_value = factories.make_str()
    loc = source_location_maker(randomize)

    return SimpleNodeWithLoc(
        int_value=int_value, float_value=float_value, str_value=str_value, loc=loc
    )


def simple_node_with_collections_maker(randomize: bool = True) -> SimpleNodeWithCollections:
    factories = RandomFactories if randomize else Factories
    int_list = factories.make_collection(int, length=3)
    str_set = factories.make_collection(str, set, length=3)
    str_to_int_dict = factories.make_mapping(key_type=str, value_type=int, length=3)
    loc = source_location_maker(randomize)

    return SimpleNodeWithCollections(
        int_list=int_list, str_set=str_set, str_to_int_dict=str_to_int_dict, loc=loc
    )


def simple_node_with_abstractcollections_maker(
    randomize: bool = True,
) -> SimpleNodeWithAbstractCollections:
    factories = RandomFactories if randomize else Factories
    int_sequence = factories.make_collection(int, collection_type=tuple, length=3)
    str_set = factories.make_collection(str, set, length=3)
    str_to_int_mapping = factories.make_mapping(key_type=str, value_type=int, length=3)

    return SimpleNodeWithAbstractCollections(
        int_sequence=int_sequence, str_set=str_set, str_to_int_mapping=str_to_int_mapping
    )


def compound_node_maker(randomize: bool = True) -> CompoundNode:
    return CompoundNode(
        location=location_node_maker(),
        simple=simple_node_maker(),
        simple_loc=simple_node_with_loc_maker(),
        simple_opt=simple_node_with_optionals_maker(),
        other_simple_opt=None,
    )


def compound_node_with_collections_maker(randomize: bool = True) -> CompoundNodeWithCollections:
    compound_node_1 = compound_node_maker()
    compound_node_2 = compound_node_maker()
    compound_node_3 = compound_node_maker()
    compound_node_4 = compound_node_maker()

    return CompoundNodeWithCollections(
        simple_col=simple_node_with_collections_maker(),
        simple_abscol=simple_node_with_abstractcollections_maker(),
        compound_col=[compound_node_1, compound_node_2],
        compound_all={"node_3": compound_node_3, "node_4": compound_node_4},
    )


def invalid_location_node_maker(randomize: bool = True) -> LocationNode:
    return LocationNode(loc=SourceLocation(line=0, column=-1))


def invalid_at_int_simple_node_maker(randomize: bool = True) -> SimpleNode:
    factories = RandomFactories if randomize else Factories
    bool_value = factories.make_bool()
    int_value = factories.make_float()
    float_value = factories.make_float()
    bytes_value = factories.make_str().encode()
    str_value = factories.make_str()
    int_kind = factories.make_member([*IntKind]) if randomize else IntKind.PLUS
    str_kind = factories.make_member([*StrKind]) if randomize else StrKind.BLA

    return SimpleNode(
        bool_value=bool_value,
        int_value=int_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def invalid_at_float_simple_node_maker(randomize: bool = True) -> SimpleNode:
    factories = RandomFactories if randomize else Factories
    bool_value = factories.make_bool()
    int_value = factories.make_int()
    float_value = factories.make_int()
    str_value = factories.make_str()
    bytes_value = factories.make_str().encode()
    int_kind = factories.make_member([*IntKind]) if randomize else IntKind.PLUS
    str_kind = factories.make_member([*StrKind]) if randomize else StrKind.BLA

    return SimpleNode(
        bool_value=bool_value,
        int_value=int_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def invalid_at_str_simple_node_maker(randomize: bool = True) -> SimpleNode:
    factories = RandomFactories if randomize else Factories
    bool_value = factories.make_bool()
    int_value = factories.make_int()
    float_value = factories.make_float()
    str_value = factories.make_float()
    bytes_value = factories.make_str().encode()
    int_kind = factories.make_member([*IntKind]) if randomize else IntKind.PLUS
    str_kind = factories.make_member([*StrKind]) if randomize else StrKind.BLA

    return SimpleNode(
        bool_value=bool_value,
        int_value=int_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def invalid_at_bytes_simple_node_maker(randomize: bool = True) -> SimpleNode:
    factories = RandomFactories if randomize else Factories
    bool_value = factories.make_bool()
    int_value = factories.make_int()
    float_value = factories.make_float()
    str_value = factories.make_float()
    bytes_value = [1, "2", (3, 4)]
    int_kind = factories.make_member([*IntKind]) if randomize else IntKind.PLUS
    str_kind = factories.make_member([*StrKind]) if randomize else StrKind.BLA

    return SimpleNode(
        bool_value=bool_value,
        int_value=int_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def invalid_at_enum_simple_node_maker(randomize: bool = True) -> SimpleNode:
    factories = RandomFactories if randomize else Factories
    bool_value = factories.make_bool()
    int_value = factories.make_int()
    float_value = factories.make_float()
    str_value = factories.make_float()
    bytes_value = factories.make_str().encode()
    int_kind = factories.make_member([*StrKind]) if randomize else IntKind.PLUS
    str_kind = factories.make_member([*StrKind]) if randomize else StrKind.BLA

    return SimpleNode(
        bool_value=bool_value,
        int_value=int_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )
