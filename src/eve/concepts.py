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

"""Definitions of basic Eve concepts."""


import collections.abc
import copy
import itertools
import operator
import types
import typing
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    MutableSequence,
    MutableSet,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import TypedDict

from . import utils
from .types import IntEnum, PositiveInt, Str, StrEnum, modelclass, validator


if typing.TYPE_CHECKING:
    from .types import ModelclassType

    NodeclassT = TypeVar("NodeclassT", bound="ModelclassType")

    class NodeclassType(ModelclassType):
        def iter_attributes(self: "NodeclassT") -> Generator[Tuple[str, Any], None, None]:
            ...

        def iter_children(self: "NodeclassT") -> Generator[Tuple[str, Any], None, None]:
            ...


class _NOTHING_TYPE:
    pass


#: Marker value used to avoid confusion with `None`
#: (specially in contexts where `None` could be a valid value)
NOTHING = _NOTHING_TYPE()


#: Prefix for Eve metadata in dataclasses
EVE_METADATA_KEY = "__eve_meta"


_EVE_NODE_IMPL_SUFFIX = "_"

_EVE_NODE_ATTR_SUFFIX = "_attr_"

_EVE_NODECLASS_PYDANTIC_CONFIG: Dict[str, Any] = {
    "extra": "forbid",
    "arbitrary_types_allowed": True,
}


def nodeclass(
    class_: Optional[Type] = None,
    *,
    frozen: bool = False,
    validate_updates: bool = False,
    fields_schema: Optional[Mapping[str, Any]] = None,
    custom_schema: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any], Type], None]]] = None,
    **kwargs: Any,
) -> Union[Callable[[Type], "NodeclassType"], "NodeclassType"]:
    f"""Make a NodeClass (Modelclass/Dataclass with extra seatures).

    In Nodeclasses, field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * other :class:`Node` subclasses
        * other :class:`pydantic.BaseModel` subclasses
        * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
            of any of the previous items

    Field names may not start with "_". Field names ending with "{_EVE_NODE_IMPL_SUFFIX}" are
    considered implementation helpers and will not appear in the node iterators.
    Field names ending with "{_EVE_NODE_ATTR_SUFFIX}" are considered meta-attributes of the node,
    not children.
    """

    def _wrapper(cls_: Type) -> "NodeclassType":
        node_config: Dict[str, Any] = {
            **_EVE_NODECLASS_PYDANTIC_CONFIG,
            "validate_assignment": validate_updates,
        }
        if fields_schema:
            node_config["fields"] = fields_schema
        if fields_schema:
            node_config["schema_extra"] = custom_schema
        if kwargs:
            node_config.update(kwargs)

        def _iter_attributes_meth(self: "NodeclassT") -> Generator[Tuple[str, Any], None, None]:
            for name, _ in self.__dataclass_fields__.items():
                if name.endswith(_EVE_NODE_ATTR_SUFFIX):
                    yield name, getattr(self, name)

        cls_.iter_attributes = _iter_attributes_meth

        def _iter_children_meth(self: "NodeclassT") -> Generator[Tuple[str, Any], None, None]:
            for name, _ in self.__dataclass_fields__.items():
                if not name.endswith(_EVE_NODE_IMPL_SUFFIX):
                    yield name, getattr(self, name)

        cls_.iter_children = _iter_children_meth

        model_cls = modelclass(
            init=True,
            repr=True,
            eq=True,
            order=False,
            unsafe_hash=False,
            frozen=frozen,
            config=node_config,
        )(cls_)

        return typing.cast("NodeclassType", model_cls)

    return _wrapper(class_) if class_ else _wrapper


# def has_inputs(node_class: Type["Node"]) -> bool:
#     return any(node_class.is_input_field(field_name) for field_name in node_class.__fields__)


# def has_outputs(node_class: Type["Node"]) -> bool:
#     return any(node_class.is_output_field(field_name) for field_name in node_class.__fields__)


# def InputField(
#     default: Any = ...,
#     *,
#     vtype: Optional[Union[Type["VType"], Tuple[Type["VType"]]]] = None,
#     **kwargs: Any,
# ) -> pydantic.fields.FieldInfo:
#     kwargs[EVE_FIELD_ANNOTATIONS_MARKER] = _make_annotations(role="in", vtype=vtype)
#     return typing.cast(pydantic.fields.FieldInfo, pydantic.Field(default, **kwargs))


# def OutputField(
#     default: Any = ...,
#     *,
#     vtype: Optional[Union[Type["VType"], Tuple[Type["VType"]]]] = None,
#     **kwargs: Any,
# ) -> pydantic.fields.FieldInfo:
#     kwargs[EVE_FIELD_ANNOTATIONS_MARKER] = _make_annotations(role="out", vtype=vtype)
#     return typing.cast(pydantic.fields.FieldInfo, pydantic.Field(default, **kwargs))


# #: Valid definitions
# #:  role: "in", "out"
# #:  vtype: ...
# EveAnnotations = TypedDict(
#     "EveAnnotations", {"role": str, "vtype": Tuple[Type["VType"]]}, total=False
# )


# def _make_annotations(
#     role: Optional[str] = None, vtype: Optional[Union[Type["VType"], Tuple[Type["VType"]]]] = None
# ) -> EveAnnotations:
#     result: EveAnnotations = {}
#     if role is not None:
#         result["role"] = role
#     elif not isinstance(role, str):
#         raise TypeError(f"role parameter contains an invalid 'str' ({role})")

#     if vtype is not None:
#         if not isinstance(vtype, tuple):
#             vtype = (vtype,)
#         assert isinstance(vtype, tuple)
#         for v in vtype:
#             if not (isinstance(v, type) and issubclass(v, VType)):
#                 raise TypeError(f"vtype parameter contains an invalid VType class ({v})")
#         result["vtype"] = vtype

#     return result


# def _get_annotations(
#     definition: Union[Type[BaseModel], pydantic.fields.ModelField], field_name: Optional[str] = None
# ) -> EveAnnotations:
#     if isinstance(definition, type):
#         if field_name is not None:
#             return typing.cast(
#                 EveAnnotations,
#                 definition.__fields__[field_name].field_info.extra.get(
#                     EVE_FIELD_ANNOTATIONS_MARKER, {}
#                 ),
#             )
#         else:
#             raise ValueError(f"Invalid field_name: '{field_name}'")
#     elif isinstance(definition, pydantic.fields.ModelField):
#         return typing.cast(
#             EveAnnotations, definition.field_info.extra.get(EVE_FIELD_ANNOTATIONS_MARKER, {})
#         )
#     else:
#         raise TypeError(f"Invalid BaseModel or ModelField type: '{definition}'")


# NodeClassesSpec = Union[str, Type["Dialect"], Type["Node"]]


# def _collect_nodes(
#     spec: Union[NodeClassesSpec, Iterable[NodeClassesSpec]]
# ) -> Tuple[Tuple[str, ...], Tuple[Type["Node"], ...], str]:

#     if spec in (Ellipsis, Node):
#         return tuple(), tuple(), utils.shash("")
#     elif isinstance(spec, (str, type)):
#         spec = [spec]

#     # Collect qualified node names
#     collected_names = []
#     for item in spec:
#         if isinstance(item, str):
#             components = item.strip().split(".")
#             if (
#                 len(components) == 2
#                 and components[0] in registered_dialects
#                 and components[1] in registered_dialects[components[0]].nodes
#             ):
#                 collected_names.append(item)
#             else:
#                 raise ValueError(f"Invalid node name: '{item}'.")
#         elif isinstance(item, type):
#             if issubclass(item, Dialect):
#                 collected_names.extend(member.name_key() for member in item.nodes.values())
#             elif issubclass(item, Node):
#                 collected_names.append(item.name_key())
#             else:
#                 raise TypeError(f"Invalid Node class: {item}.")
#         else:
#             raise TypeError(f"Invalid Node class: {item}.")

#     # Sort and hash the collected names
#     collected_names = tuple(sorted(set(collected_names)))
#     collected_hash = utils.shash(*collected_names)

#     # Collect node types
#     collected_types = []
#     for name in collected_names:
#         dialect, node = name.split(".")
#         collected_types.append(registered_dialects[dialect].nodes[node])

#     collected_types = tuple(collected_types)

#     return collected_names, collected_types, collected_hash


# def _validate_dialect_reference(
#     dialect_member_class: Type[Any], expected_dialect: Optional[Type["Dialect"]] = None
# ) -> Type["Dialect"]:
#     dialect: Type["Dialect"] = dialect_member_class.__dict__.get("dialect", expected_dialect)
#     if expected_dialect is None:
#         if "dialect" not in dialect_member_class.__dict__:
#             raise TypeError(f"{dialect_member_class} does not contain a dialect reference")
#         if not issubclass(dialect, Dialect):
#             raise TypeError(f"Reference to an invalid dialect class ({dialect})")
#     elif issubclass(expected_dialect, Dialect):
#         if dialect is not expected_dialect:
#             raise TypeError(
#                 f"A conflictive 'dialect' class definition exists in {dialect_member_class} ({dialect})"
#             )
#     else:
#         raise TypeError(f"Invalid expected dialect class ({expected_dialect})")

#     return dialect


class UIDGenerator:
    """Simple unique id generator using a counter."""

    #: Constantly increasing counter for generation of unique ids
    __counter = itertools.count(1)

    @classmethod
    def get_unique_id(cls, *, prefix: Optional[str] = None) -> str:
        """Generate a new globally unique id (for the current session)."""
        count = next(cls.__counter)
        return f"{prefix}_{count}" if prefix else f"{count}"

    @classmethod
    def reset(cls, start: int = 1) -> None:
        """Reset generator."""
        cls.__counter = itertools.count(start)


@modelclass(frozen=True)
class SourceLocation:
    """Source code location (line, column, source)."""

    line: PositiveInt
    column: PositiveInt
    source: Str

    def __str__(self) -> str:
        return f"<'{self.source}': Line {self.line}, Col {self.column}>"


@modelclass
class VType:
    """Representation of an abstract value data type."""

    pass


# @nodeclass
# class Node:
#     f"""Base node class.

#     Field values should be either:

#         * builtin types: `bool`, `bytes`, `int`, `float`, `str`
#         * other :class:`Node` subclasses
#         * other :class:`pydantic.BaseModel` subclasses
#         * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
#           of any of the previous items

#     Field names may not start with "_". Field names ending with "{_NODE_IMPL_SUFFIX}" are
#     considered implementation helpers and will not appear in the node iterators.
#     Field names ending with "{_NODE_ATTR_SUFFIX}" are considered meta-attributes of the node,
#     not children.
#     """

#     # Node fields
#     #: Unique node id (meta-attribute)
#     id_attr_: Optional[Str] = None

#     @validator("id_attr_", pre=True, always=True)
#     def _id_attr_validator(cls: Type["Node"], v: Optional[str]) -> str:  # type: ignore
#         if v is None:
#             v = UIDGenerator.get_unique_id(prefix=cls.__qualname__)
#         if not isinstance(v, str):
#             raise TypeError(f"id_attr_ is not an 'str' instance ({type(v)})")
#         return v

# def iter_attributes(self) -> Generator[Tuple[str, Any], None, None]:
#     for name, _ in self.__dataclass_fields__.items():
#         if name.endswith(_NODE_ATTR_SUFFIX):
#             yield name, getattr(self, name)

# def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
#     for name, _ in self.__dataclass_fields__.items():
#         if not name.endswith(_NODE_IMPL_SUFFIX):
#             yield name, getattr(self, name)

# def iter_inputs(self) -> Generator[Tuple[str, Any], None, None]:
#     for name in self.__dataclass_fields__:
#         if self.is_input_field(name) and not name.endswith("__"):
#             yield name, getattr(self, name)

# def iter_outputs(self) -> Generator[Tuple[str, Any], None, None]:
#     for name in self.__dataclass_fields__:
#         if self.is_output_field(name) and not name.endswith("__"):
#             yield name, getattr(self, name)

# @classmethod
# def is_input_field(cls, field_name: str) -> bool:
#     assert field_name in cls.__dataclass_fields__
#     return bool(_get_annotations(cls, field_name).get("role", None) == "in")

# @classmethod
# def is_output_field(cls, field_name: str) -> bool:
#     assert field_name in cls.__dataclass_fields__
#     return bool(_get_annotations(cls, field_name).get("role", None) == "out")


# class _NodeFromClass:
#     def __getitem__(self, spec: Union[NodeClassesSpec, Iterable[NodeClassesSpec]]) -> TypeVar:  # type: ignore
#         _, classes, hashed_id = _collect_nodes(spec)
#         type_var_name = f"Node_{hashed_id}".replace("-", "_")
#         return TypeVar(type_var_name, *classes)  # type: ignore


# #: Syntax marker to create a TypeVar with restricted node types
# NodeFrom = _NodeFromClass()


# class Module:
#     pass
#     # symtable
#     # root


# class Program:
#     pass
#     # modules: List[Module]
#     # symtable
#     # dialects
