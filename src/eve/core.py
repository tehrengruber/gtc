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

"""Definitions of basic infrastructure classes and functions."""


import copy
import itertools
import operator
from collections import abc as col_abc
from enum import Enum
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generator,
    Iterable,
    MutableSequence,
    MutableSet,
    Optional,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel, PositiveInt, validator


#: Marker value used to avoid confusion with `None`
#: (specially in contexts where `None` could be a valid value)
NOTHING = object()

#: Constantly increasing counter for generation of unique ids
__unique_counter = itertools.count(1)


def get_unique_id() -> int:
    """Generate a new globally unique `int` for the current session."""
    return next(__unique_counter)


class StrEnum(str, Enum):
    """Basic :class:`enum.Enum` subclass compatible with string operations."""

    def __str__(self) -> str:
        return self.value


class SourceLocation(BaseModel):
    """Source code location (line, column)."""

    line: PositiveInt
    column: PositiveInt


class BaseNode(BaseModel):
    """Base node class.

    Field values should be either:

        * builtin types: `str`, `int`, `float`, `bytes`?
        * other :class:`BaseNode` subclasses
        * other :class:`pydantic.BaseModel` subclasses
        * supported collections (:class:`collections.abc.Sequence`,
        :class:`collections.abc.Set`, :class:`collections.abc.Mapping`) of
        any of the previous items

    Using other classes as values would most likely work but it is not
    explicitly supported.

    Field names ending with "_"  are considered hidden fields and
    will not appear in the node iterators. Field names ending with
    "_attr" are considered attributes of the node, not children.
    """

    id_attr: Optional[int] = None
    kind_attr: Optional[str] = None
    dialect_attr: Optional[str] = None

    @validator("id_attr", pre=True, always=True)
    def _id_attr_validator(cls: Type[BaseModel], v: Optional[int]) -> int:  # type: ignore
        return v or get_unique_id()

    @validator("kind_attr", pre=True, always=True)
    def _kind_attr_validator(cls: Type[BaseModel], v: Optional[str]) -> str:  # type: ignore
        if v and v != cls.__name__:
            raise ValueError(f"kind_attr value '{v}' does not match cls.__name__ {cls.__name__}")

        return v or cls.__name__

    @validator("dialect_attr", pre=True, always=True)
    def _dialect_attr_validator(cls: Type[BaseModel], v: Optional[str]) -> str:  # type: ignore
        return v or cls.__module__.split(".")[-1]

    @property
    def attributes(self) -> Dict[str, Any]:
        return {name: value for name, value in self.iter_attributes()}

    @property
    def children(self) -> Dict[str, Any]:
        return {name: value for name, value in self.iter_children()}

    def iter_attributes(self) -> Generator[Tuple[str, Any], None, None]:
        for name, value in self.__fields__.items():
            if name.endswith("attr"):
                yield (name, getattr(self, name))

    def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
        for name, value in self.__fields__.items():
            if not (name.endswith("attr") or name.endswith("_")):
                yield (name, getattr(self, name))


class MutableNode(BaseNode):
    """Base mutable node class."""

    class Config:
        allow_mutation = True


class InmutableNode(BaseNode):
    """Base inmutable node class."""

    class Config:
        allow_mutation = False


Node = MutableNode


ValidLeafNodeType = Optional[Union[int, float, str, BaseNode, BaseModel]]
ValidNodeType = Optional[Union[ValidLeafNodeType, Collection[ValidLeafNodeType]]]


class NodeVisitor:
    """Simple node visitor class based on :class:`ast.NodeVisitor`.

    The base class walks the tree and calls a visitor function for every
    node found. This function may return a value which is forwarded by
    the `visit` method. This class is meant to be subclassed, with the
    subclass adding visitor methods.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node. So a `BinOpExpr` node visit function would
    be `visit_BinOpExpr`. If no visitor function exists for a node,
    it tries to get a visitor function for each of its parent classes
    in the order define by the class' `__mro__` attribute. Finally,
    if no visitor function exists for a node or its parents, the
    `generic_visit` visitor is used instead. This behavior can be changed
    by overriding the `visit` method.

    Don't use the `NodeVisitor` if you want to apply changes to nodes during
    traversing. For this a special visitor exists (`NodeTransformer`) that
    allows modifications.
    """

    def visit(self, node: ValidNodeType, **kwargs) -> Any:
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(node, **kwargs)

    def generic_visit(self, node: ValidNodeType, **kwargs) -> Any:
        items: Iterable[Tuple[Any, Any]] = []
        if isinstance(node, Node):
            items = node.iter_children()
        elif isinstance(node, (col_abc.Sequence, col_abc.Set)) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            items = enumerate(node)
        elif isinstance(node, col_abc.Mapping):
            items = node.items()

        # Process selected items (if any)
        for _, value in items:
            self.visit(value, **kwargs)


class NodeTransformer(NodeVisitor):
    """Simple :class:`NodeVisitor` subclass based on :class:`ast.NodeTransformer` to modify nodes in place.

    The `NodeTransformer` will walk the tree and use the return value of
    the visitor methods to replace or remove the old node. If the
    return value of the visitor method is :obj:`eve.core.NOTHING`,
    the node will be removed from its location, otherwise it is replaced
    with the return value. The return value may also be the original
    node, in which case no replacement takes place.

    Keep in mind that if the node you're operating on has child nodes
    you must either transform the child nodes yourself or call the
    :meth:`generic_visit` method for the node first.

    Usually you use the transformer like this::

       node = YourTransformer().visit(node)
    """

    def generic_visit(self, node: ValidNodeType, **kwargs) -> Any:
        result: Any = node
        if isinstance(node, (Node, col_abc.Collection)) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            items: Iterable[Tuple[Any, Any]] = []
            tmp_items: Collection[ValidNodeType] = []
            set_op: Union[Callable[[Any, str, Any], None], Callable[[Any, int, Any], None]]
            del_op: Union[Callable[[Any, str], None], Callable[[Any, int], None]]

            if isinstance(node, Node):
                items = node.iter_children()
                set_op = setattr
                del_op = delattr
            elif isinstance(node, col_abc.MutableSequence):
                items = enumerate(node)
                index_shift = 0

                def set_op(container: MutableSequence, idx: int, value: ValidNodeType) -> None:
                    container[idx - index_shift] = value

                def del_op(container: MutableSequence, idx: int) -> None:
                    nonlocal index_shift
                    del container[idx - index_shift]
                    index_shift += 1

            elif isinstance(node, col_abc.MutableSet):
                items = list(enumerate(node))

                def set_op(container: MutableSet, idx: Any, value: ValidNodeType) -> None:
                    container.add(value)

                def del_op(container: MutableSet, idx: int) -> None:
                    container.remove(items[idx])  # type: ignore

            elif isinstance(node, col_abc.MutableMapping):
                items = node.items()
                set_op = operator.setitem
                del_op = operator.delitem
            elif isinstance(node, (col_abc.Sequence, col_abc.Set)):
                # Inmutable sequence or set: create a new container instance with the new values
                tmp_items = [self.visit(value, **kwargs) for value in node]
                result = node.__class__(  # type: ignore
                    [value for value in tmp_items if value is not NOTHING]
                )
            elif isinstance(node, col_abc.Mapping):
                # Inmutable mapping: create a new mapping instance with the new values
                tmp_items = {key: self.visit(value, **kwargs) for key, value in node.items()}
                result = node.__class__(  # type: ignore
                    {key: value for key, value in tmp_items.items() if value is not NOTHING}
                )

            # Finally, in case current node object is mutable, process selected items (if any)
            for key, value in items:
                new_value = self.visit(value, **kwargs)
                if new_value is NOTHING:
                    del_op(result, key)
                elif new_value != value:
                    set_op(result, key, new_value)
        return result


class NodeTranslator(NodeVisitor):
    """`NodeVisitor` subclass to modify nodes NOT in place.

    The `NodeTranslator` will walk the tree and use the return value of
    the visitor methods to replace or remove the old node in a new copy
    of the tree. If the return value of the visitor method is
    `eve.core.NOTHING`, the node will be removed from its location in the
    result tree, otherwise it is replaced with the return value. In the
    default case, a `deepcopy` of the original node is returned.

    Keep in mind that if the node you're operating on has child nodes
    you must either transform the child nodes yourself or call the
    :meth:`generic_visit` method for the node first.

    Usually you use the transformer like this::

       node = YourTranslator().visit(node)
    """

    def __init__(self, *, memo: dict = None, **kwargs):
        assert memo is None or isinstance(memo, dict)
        self.memo = memo or {}

    def generic_visit(self, node: ValidNodeType, **kwargs) -> Any:
        result: Any
        if isinstance(node, (Node, col_abc.Collection)) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            tmp_items: Collection[ValidNodeType] = []
            if isinstance(node, Node):
                tmp_items = {key: self.visit(value, **kwargs) for key, value in node}
                result = node.__class__(  # type: ignore
                    node_id_=node.node_id_,
                    node_kind_=node.node_kind_,
                    **{key: value for key, value in tmp_items.items() if value is not NOTHING},
                )

            elif isinstance(node, (col_abc.Sequence, col_abc.Set)):
                # Sequence or set: create a new container instance with the new values
                tmp_items = [self.visit(value, **kwargs) for value in node]
                result = node.__class__(  # type: ignore
                    [value for value in tmp_items if value is not NOTHING]
                )

            elif isinstance(node, col_abc.Mapping):
                # Mapping: create a new mapping instance with the new values
                tmp_items = {key: self.visit(value, **kwargs) for key, value in node.items()}
                result = node.__class__(  # type: ignore
                    {key: value for key, value in tmp_items.items() if value is not NOTHING}
                )

        else:
            result = copy.deepcopy(node, memo=self.memo)

        return result
