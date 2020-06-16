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


import collections.abc
import copy
import itertools
import operator
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
    MutableSequence,
    MutableSet,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pydantic
import pydantic.utils
from pydantic import BaseModel, PositiveInt, root_validator, validator
from typing_extensions import TypedDict

from . import utils
from .types import Bool, Bytes, Float, Int, Str


#: Marker value used to avoid confusion with `None`
#: (specially in contexts where `None` could be a valid value)
NOTHING = object()


#: Public register of dialects
registered_dialects: Dict[str, Type["Dialect"]] = {}

EVE_FIELD_ANNOTATIONS_MARKER = "__eve"


def has_inputs(node_class: Type["Node"]) -> bool:
    return any(node_class.is_input_field(field_name) for field_name in node_class.__fields__)


def has_regions(node_class: Type["Node"]) -> bool:
    return any(node_class.is_region_field(field_name) for field_name in node_class.__fields__)


def has_outputs(node_class: Type["Node"]) -> bool:
    return any(node_class.is_output_field(field_name) for field_name in node_class.__fields__)


def has_successors(node_class: Type["Node"]) -> bool:
    return any(node_class.is_successor_field(field_name) for field_name in node_class.__fields__)


def InputField(
    default: Any = ...,
    *,
    vtype: Optional[Union[Type["VType"], Tuple[Type["VType"]]]] = None,
    **kwargs: Any,
) -> pydantic.fields.FieldInfo:
    kwargs[EVE_FIELD_ANNOTATIONS_MARKER] = _make_annotations(role="in", vtype=vtype)
    return typing.cast(pydantic.fields.FieldInfo, pydantic.Field(default, **kwargs))


def OutputField(
    default: Any = ...,
    *,
    vtype: Optional[Union[Type["VType"], Tuple[Type["VType"]]]] = None,
    **kwargs: Any,
) -> pydantic.fields.FieldInfo:
    kwargs[EVE_FIELD_ANNOTATIONS_MARKER] = _make_annotations(role="out", vtype=vtype)
    return typing.cast(pydantic.fields.FieldInfo, pydantic.Field(default, **kwargs))


#: Valid definitions
#:  role: "in", "out"
#:  vtype: ...
EveAnnotations = TypedDict(
    "EveAnnotations", {"role": str, "vtype": Tuple[Type["VType"]]}, total=False
)


def _make_annotations(
    role: Optional[str] = None, vtype: Optional[Union[Type["VType"], Tuple[Type["VType"]]]] = None
) -> EveAnnotations:
    result: EveAnnotations = {}
    if role is not None:
        result["role"] = role
    elif not isinstance(role, str):
        raise TypeError(f"role parameter contains an invalid 'str' ({role})")

    if vtype is not None:
        if not isinstance(vtype, tuple):
            vtype = (vtype,)
        assert isinstance(vtype, tuple)
        for v in vtype:
            if not (isinstance(v, type) and issubclass(v, VType)):
                raise TypeError(f"vtype parameter contains an invalid VType class ({v})")
        result["vtype"] = vtype

    return result


def _get_annotations(
    definition: Union[Type[BaseModel], pydantic.fields.ModelField], field_name: Optional[str] = None
) -> EveAnnotations:
    if isinstance(definition, type):
        if field_name is not None:
            return typing.cast(
                EveAnnotations,
                definition.__fields__[field_name].field_info.extra.get(
                    EVE_FIELD_ANNOTATIONS_MARKER, {}
                ),
            )
        else:
            raise ValueError(f"Invalid field_name: '{field_name}'")
    elif isinstance(definition, pydantic.fields.ModelField):
        return typing.cast(
            EveAnnotations, definition.field_info.extra.get(EVE_FIELD_ANNOTATIONS_MARKER, {})
        )
    else:
        raise TypeError(f"Invalid BaseModel or ModelField type: '{definition}'")


NodeClassesSpec = Union[str, Type["Dialect"], Type["Node"]]


def _collect_nodes(
    spec: Union[NodeClassesSpec, Iterable[NodeClassesSpec]]
) -> Tuple[Tuple[str, ...], Tuple[Type["Node"], ...], str]:

    if spec in (Ellipsis, Node):
        return tuple(), tuple(), utils.shash("")
    elif isinstance(spec, (str, type)):
        spec = [spec]

    # Collect qualified node names
    collected_names = []
    for item in spec:
        if isinstance(item, str):
            components = item.strip().split(".")
            if (
                len(components) == 2
                and components[0] in registered_dialects
                and components[1] in registered_dialects[components[0]].nodes
            ):
                collected_names.append(item)
            else:
                raise ValueError(f"Invalid node name: '{item}'.")
        elif isinstance(item, type):
            if issubclass(item, Dialect):
                collected_names.extend(member.name_key() for member in item.nodes.values())
            elif issubclass(item, Node):
                collected_names.append(item.name_key())
            else:
                raise TypeError(f"Invalid Node class: {item}.")
        else:
            raise TypeError(f"Invalid Node class: {item}.")

    # Sort and hash the collected names
    collected_names = tuple(sorted(set(collected_names)))
    collected_hash = utils.shash(*collected_names)

    # Collect node types
    collected_types = []
    for name in collected_names:
        dialect, node = name.split(".")
        collected_types.append(registered_dialects[dialect].nodes[node])

    collected_types = tuple(collected_types)

    return collected_names, collected_types, collected_hash


def _register_dialect(dialect_class: Type["Dialect"]) -> Type["Dialect"]:
    global registered_dialects

    if not issubclass(dialect_class, Dialect):
        raise TypeError(f"Invalid dialect class {dialect_class}.")
    if "name" not in dialect_class.__dict__:
        raise TypeError("Dialect classes must define a unique 'name: str' attribute.")
    if (
        dialect_class.name in registered_dialects
        and dialect_class is not registered_dialects[dialect_class.name]
    ):
        raise ValueError(f"Dialect name ('{dialect_class.name}') is not unique.")

    if dialect_class.name not in registered_dialects:
        registered_dialects[dialect_class.name] = dialect_class
        dialect_class.nodes = {}
        dialect_class.vtypes = {}

    return dialect_class


def _validate_dialect_reference(
    dialect_member_class: Type[Any], expected_dialect: Optional[Type["Dialect"]] = None
) -> Type["Dialect"]:
    dialect: Type["Dialect"] = dialect_member_class.__dict__.get("dialect", expected_dialect)
    if expected_dialect is None:
        if "dialect" not in dialect_member_class.__dict__:
            raise TypeError(f"{dialect_member_class} does not contain a dialect reference.")
        if not issubclass(dialect, Dialect):
            raise TypeError(f"Reference to an invalid dialect class ({dialect})")
    elif issubclass(expected_dialect, Dialect):
        if dialect is not expected_dialect:
            raise TypeError(
                f"A conflictive 'dialect' class definition exists in {dialect_member_class} ({dialect})"
            )
    else:
        raise TypeError(f"Invalid expected dialect class ({expected_dialect})")

    return dialect


class UIDGenerator:
    """Simple unique id generator using a counter."""

    #: Constantly increasing counter for generation of unique ids
    __counter = itertools.count(1)

    @classmethod
    def get_unique_id(cls, *, prefix: Optional[str] = None) -> str:
        """Generate a new globally unique id (for the current session)."""
        count = next(cls.__counter)
        if prefix:
            return f"{prefix}_{count}"
        else:
            return f"{count}"

    @classmethod
    def reset(cls, start: int = 1) -> None:
        """Reset generator."""
        cls.__counter = itertools.count(start)


class BaseModelConfig:
    extra = "forbid"


class FrozenModelConfig(BaseModelConfig):
    allow_mutation = False


class SourceLocation(BaseModel):
    """Source code location (line, column, source)."""

    line: PositiveInt
    column: PositiveInt
    source: Str

    class Config(FrozenModelConfig):
        pass

    def __init__(self, line: int, column: int, source: str):
        super().__init__(line=line, column=column, source=source)

    def __str__(self) -> str:
        return f"<{self.source}: Line {self.line}, Col {self.column}>"


class CFTarget(BaseModel):
    """Control flow target class."""

    RETURN_TARGET: ClassVar[str] = "<RET>"

    # CFTarget
    #: Target block label
    block: str

    #: List of values passed to target block
    args: List["Node"]

    class Config(BaseModelConfig):
        pass

    @classmethod
    def make_return(cls, args: Optional[List["Node"]] = None) -> "CFTarget":
        return cls(block=cls.RETURN_TARGET, args=args if args else [])

    def __init__(self, block: str, args: List["Node"]):
        super().__init__(block=block, args=args)

    def is_return(self) -> bool:
        return self.block == self.RETURN_TARGET


SuccessorsList = List[CFTarget]


class Dialect:
    """Base dialect class."""

    # Placeholders for class members in concrete Dialect subclasses
    #: Unique name of the dialect
    name: ClassVar[str]

    #: Registered dialect nodes
    nodes: ClassVar[Dict[str, Type["Node"]]]

    #: Registered dialect types
    vtypes: ClassVar[Dict[str, Type["VType"]]]

    @classmethod
    def __init_subclass__(cls) -> None:
        _register_dialect(cls)

    @classmethod
    def register(cls, new_member_class: Type["DialectMember"]) -> Type["DialectMember"]:
        """Register a new member (Node, VType) of the dialect."""

        if issubclass(new_member_class, Node):
            member_type = "node"
        elif issubclass(new_member_class, VType):
            member_type = "vtype"
        else:
            raise TypeError(
                f"New member class ('{new_member_class}') is not a DialectMember subclass."
            )

        registered_members = getattr(cls, f"{member_type}s")

        member_name = new_member_class.__dict__.get("_name_", None)
        if not member_name:
            member_name = new_member_class.__name__

        if member_name.startswith("__") and member_name.endswith("__"):
            raise ValueError(f"Special node class ({new_member_class}) can not be registered.")

        if (
            member_name in registered_members
            and registered_members[member_name] is not new_member_class
        ):
            raise ValueError(
                f"{member_type.capitalize()} name '{member_name}' has been already "
                f"registered by {registered_members[member_name]}."
            )

        registered_members[member_name] = new_member_class
        new_member_class._dialect_ = _validate_dialect_reference(new_member_class, cls)

        return new_member_class


class DialectMember(BaseModel):
    # Placeholders for class members in concrete subclasses
    #: Custon name (if None, the name of the class will be used instead)
    _name_: ClassVar[Optional[str]] = None

    #: Reference to dialect (added automatically at registration)
    _dialect_: ClassVar[Optional[Type[Dialect]]] = None

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls._name_ is not None and not isinstance(cls._name_, str):
            raise TypeError(
                f"DialectMember class {cls} defines an invalid '_name_' (str) class attribute: {cls._name_}"
            )

        if cls._dialect_ is not None and not isinstance(cls._dialect_, Dialect):
            raise TypeError(
                f"Invalid dialect definition for class {cls}: {cls.__dict__['_dialect_']}."
            )

    @classmethod
    def name_key(cls) -> str:
        name = cls._name_ or cls.__name__
        if cls._dialect_:
            name = f"{cls._dialect_.name}.{name}"
        return name


class VType(DialectMember):
    """Representation of an abstract value data type."""

    class Config(FrozenModelConfig):
        pass


class Node(DialectMember):
    """Base node class.

    Field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * other :class:`Node` subclasses
        * other :class:`pydantic.BaseModel` subclasses
        * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
          of any of the previous items

    Field names ending with "_"  are considered hidden fields and
    will not appear in the node iterators. Field names ending with
    "__" are considered meta-attributes of the node, not children.
    """

    class Config(BaseModelConfig):
        pass

    # Node fields
    #: Unique node id (meta-attribute)
    id__: Optional[Str] = None

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        # Checka that the node field definitions are semantically sound
        successors_names: List[str] = []
        for name, model_field in cls.__fields__.items():
            vtype_definition = _get_annotations(model_field).get("vtype", None)
            if vtype_definition is not None and not issubclass(model_field.outer_type_, Node):
                raise TypeError(
                    f"Node class {cls.__name__} defines vtype constraints for field '{name}'"
                    f" which is not a Node ({model_field})"
                )
            if not name.endswith("__") and model_field.outer_type_ == List[CFTarget]:
                successors_names.append(name)

        if len(successors_names) > 1:
            raise TypeError(
                f"Node class {cls.__name__} defines multiple successors lists: {successors_names}"
            )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @validator("id__", pre=True, always=True)
    def _id_attr_validator(cls: Type["Node"], v: Optional[str]) -> str:  # type: ignore
        if v is None:
            v = UIDGenerator.get_unique_id(prefix=cls._name_)
        if not isinstance(v, str):
            raise TypeError(f"id__ is not an 'str' instance ({type(v)})")
        return v

    def iter_attributes(self) -> Generator[Tuple[str, Any], None, None]:
        for name, _ in self.__fields__.items():
            if name.endswith("__"):
                yield name, getattr(self, name)

    def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
        for name, _ in self.__fields__.items():
            if not name.endswith("__"):
                yield name, getattr(self, name)

    def iter_inputs(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__fields__:
            if self.is_input_field(name) and not name.endswith("__"):
                yield name, getattr(self, name)

    def iter_regions(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__fields__:
            if self.is_region_field(name) and not name.endswith("__"):
                yield name, getattr(self, name)

    def iter_outputs(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__fields__:
            if self.is_output_field(name) and not name.endswith("__"):
                yield name, getattr(self, name)

    def iter_successors(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__fields__:
            if self.is_successor_field(name) and not name.endswith("__"):
                for successor in getattr(self, name):
                    yield successor

                break

    @classmethod
    def is_input_field(cls, field_name: str) -> bool:
        assert field_name in cls.__fields__
        return bool(_get_annotations(cls, field_name).get("role", None) == "in")

    @classmethod
    def is_region_field(cls, field_name: str) -> bool:
        assert field_name in cls.__fields__
        return issubclass(cls.__fields__[field_name].type_, Region)

    @classmethod
    def is_output_field(cls, field_name: str) -> bool:
        assert field_name in cls.__fields__
        return bool(_get_annotations(cls, field_name).get("role", None) == "out")

    @classmethod
    def is_successor_field(cls, field_name: str) -> bool:
        assert field_name in cls.__fields__
        return bool(cls.__fields__[field_name].outer_type_ == List[CFTarget])


class _NodeFromClass:
    def __getitem__(self, spec: Union[NodeClassesSpec, Iterable[NodeClassesSpec]]) -> TypeVar:  # type: ignore
        _, classes, hashed_id = _collect_nodes(spec)
        type_var_name = f"Node_{hashed_id}".replace("-", "_")
        return TypeVar(type_var_name, *classes)  # type: ignore


#: Syntax marker to create a TypeVar with restricted node types
NodeFrom = _NodeFromClass()


class Block(BaseModel, collections.abc.MutableSequence):
    class Config(BaseModelConfig):
        orm_mode = True

    #: Accepted node types (it should be modified in custom subclasses)
    _restricted_nodes_: ClassVar[Tuple[Type[Node], ...]] = tuple()

    # Block fields
    #: Block label
    label: str = ""

    #: List of input arguments
    inputs: List[Node] = []

    #: List of nodes contained in the block
    nodes: List[Node] = []

    def __init__(self, label: str = "", **kwargs: Any):
        super().__init__(label=label, **kwargs)

    @root_validator(pre=True)
    def _nodes_from_object_validator(  # type: ignore
        cls: Type["Block"], data: Union[Dict[str, Any], pydantic.utils.GetterDict],
    ) -> Union[Dict[str, Any], pydantic.utils.GetterDict]:
        # This pre-root validator (combined with the orm_mode setting in Config) provides
        # a workaround to initialize fields of block subclasses inside other nodes using
        # superclass Block objects
        if isinstance(data, pydantic.utils.GetterDict) and isinstance(data._obj, Block):
            data = {"label": data._obj.label, "inputs": data._obj.inputs, "nodes": data._obj.nodes}

        return data

    @validator("nodes")
    def _node_list_validator(cls: Type["Block"], v: List[Any]) -> List[Node]:  # type: ignore
        if not isinstance(v, collections.abc.Sequence):
            raise TypeError(f"Invalid list of Node objects '{v}'")
        return cls._validate_nodes(v)

    @classmethod
    def _validate_nodes(cls, nodes: List[Node]) -> List[Node]:
        for node in nodes:
            if cls._restricted_nodes_ and not isinstance(node, cls._restricted_nodes_):
                raise TypeError(
                    f"'{node}'' has an invalid type ({type(node)}) for this block."
                    "Valid types are: {cls._restricted_nodes_}."
                )

        if nodes and not has_successors(nodes[-1].__class__):
            raise TypeError(f"Last node in the block ({nodes[-1]}) is not a terminator.")

        return nodes

    @typing.overload
    def __getitem__(self, idx: int) -> Node:
        ...

    @typing.overload
    def __getitem__(self, s: slice) -> List[Node]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Node, List[Node]]:
        return self.nodes.__getitem__(index)

    @typing.overload
    def __setitem__(self, idx: int, value: Node) -> None:
        ...

    @typing.overload
    def __setitem__(self, s: slice, value: Iterable[Node]) -> None:
        ...

    def __setitem__(self, index: Union[int, slice], value: Union[Node, Iterable[Node]]) -> None:
        if isinstance(value, Node):
            self._validate_nodes([value])
        else:
            self._validate_nodes(list(value))
        self.nodes.__setitem__(index, value)  # type:ignore

    def __delitem__(self, index: Union[int, slice]) -> None:
        return self.nodes.__delitem__(index)

    def __len__(self) -> int:
        return self.nodes.__len__()

    def insert(self, index: int, value: Node) -> None:
        self._validate_nodes([value])
        return self.nodes.insert(index, value)


class _BlockOfClass:

    __cached_subclasses: ClassVar[Dict[int, Type["Block"]]] = {}

    def __getitem__(self, spec: Union[NodeClassesSpec, Sequence[NodeClassesSpec]]) -> Type[Block]:
        spec = [spec] if isinstance(spec, (str, type)) else list(spec)
        class_name: Optional[str] = None

        if spec:
            for _i, item in enumerate(spec):
                if isinstance(item, str):
                    item = item.strip()
                    if item.startswith("__classname__:"):
                        class_name = item.split(":")[1]
                        continue

            if _i < len(spec):
                del spec[_i]

        restricted_names, restricted_classes, hashed_id = _collect_nodes(spec)

        # Find or create a (cached) custom Block subclass
        if hashed_id not in self.__cached_subclasses:
            class_name = class_name or f"Block_{hashed_id:x}".replace("-", "_")
            # Create a new pydantic model dynamically
            self.__cached_subclasses[hashed_id] = pydantic.create_model(  # type: ignore
                class_name,
                _restricted_nodes_=(ClassVar[Tuple[Type[Node]]], restricted_classes),
                nodes=(List[Union[restricted_classes]], []),  # type: ignore
                __base__=Block,
            )

        return self.__cached_subclasses[hashed_id]


#: Syntax marker to create constrained Block subclasses
BlockOf = _BlockOfClass()


class Region(BaseModel, collections.abc.MutableMapping):
    class Config(BaseModelConfig):
        orm_mode = True

    # CFGRegion fields
    #: Dict of blocks with labels
    blocks: Dict[str, Block] = {}

    def __init__(self, blocks: Optional[Dict[str, Block]] = None, *args, **kwargs) -> None:
        if blocks is None:
            blocks = {}
        super().__init__(blocks=blocks, *args, **kwargs)

    @root_validator(pre=True)
    def _blocks_from_object_validator(
        cls: Type[Block], data: Union[Dict[str, Any], pydantic.utils.GetterDict],
    ) -> Union[Dict[str, Any], pydantic.utils.GetterDict]:
        # This pre-root validator (combined with the orm_mode setting in Config) provides
        # a workaround to initialize CFGRegion fields inside other nodes using plain dicts
        if isinstance(data, pydantic.utils.GetterDict) and isinstance(
            data._obj, collections.abc.Mapping
        ):
            data = {"blocks": data._obj}

        return data

    def __getitem__(self, index):
        return self.blocks.__getitem__(index)

    def __setitem__(self, index, value):
        return self.blocks.__setitem__(index, value)

    def __delitem__(self, index):
        return self.blocks.__delitem__(index)

    def __len__(self):
        return self.blocks.__len__()

    def __iter__(self):
        return self.blocks.__iter__()


class Module:
    pass
    # symtable
    # dialects


class Program:
    pass
    # modules: List[Module]
    # symtable
    # dialects


ValidLeafNodeType = Union[
    bool, bytes, int, float, str, Bool, Bytes, Int, Float, Str, Node, BaseModel, None
]

ValidNodeType = Union[ValidLeafNodeType, Collection[ValidLeafNodeType]]


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

    ATOMIC_COLLECTION_TYPES = (str, bytes, bytearray)

    def visit(self, node: ValidNodeType, **kwargs) -> Any:
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

                if node_class is Node:
                    break

        return visitor(node, **kwargs)

    def generic_visit(self, node: ValidNodeType, **kwargs) -> Any:
        items: Iterable[Tuple[Any, Any]] = []
        if isinstance(node, Node):
            items = node.children()
        elif isinstance(node, (collections.abc.Sequence, collections.abc.Set)) and not isinstance(
            node, self.ATOMIC_COLLECTION_TYPES
        ):
            items = enumerate(node)
        elif isinstance(node, collections.abc.Mapping):
            items = node.items()

        # Process selected items (if any)
        for _, value in items:
            self.visit(value, **kwargs)


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
        if isinstance(node, (Node, collections.abc.Collection)) and not isinstance(
            node, self.ATOMIC_COLLECTION_TYPES
        ):
            tmp_items: Collection[ValidNodeType] = []
            if isinstance(node, Node):
                tmp_items = {key: self.visit(value, **kwargs) for key, value in node}
                result = node.__class__(  # type: ignore
                    **{key: value for key, value in node.attributes()},
                    **{key: value for key, value in tmp_items.items() if value is not NOTHING},
                )

            elif isinstance(node, (collections.abc.Sequence, collections.abc.Set)):
                # Sequence or set: create a new container instance with the new values
                tmp_items = [self.visit(value, **kwargs) for value in node]
                result = node.__class__(  # type: ignore
                    [value for value in tmp_items if value is not NOTHING]
                )

            elif isinstance(node, collections.abc.Mapping):
                # Mapping: create a new mapping instance with the new values
                tmp_items = {key: self.visit(value, **kwargs) for key, value in node.items()}
                result = node.__class__(  # type: ignore
                    {key: value for key, value in tmp_items.items() if value is not NOTHING}
                )

        else:
            result = copy.deepcopy(node, memo=self.memo)

        return result


class NodeModifier(NodeVisitor):
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
        if isinstance(node, (Node, collections.abc.Collection)) and not isinstance(
            node, self.ATOMIC_COLLECTION_TYPES
        ):
            items: Iterable[Tuple[Any, Any]] = []
            tmp_items: Collection[ValidNodeType] = []
            set_op: Union[Callable[[Any, str, Any], None], Callable[[Any, int, Any], None]]
            del_op: Union[Callable[[Any, str], None], Callable[[Any, int], None]]

            if isinstance(node, Node):
                items = node.children()
                set_op = setattr
                del_op = delattr
            elif isinstance(node, collections.abc.MutableSequence):
                items = enumerate(node)
                index_shift = 0

                def set_op(container: MutableSequence, idx: int, value: ValidNodeType) -> None:
                    container[idx - index_shift] = value

                def del_op(container: MutableSequence, idx: int) -> None:
                    nonlocal index_shift
                    del container[idx - index_shift]
                    index_shift += 1

            elif isinstance(node, collections.abc.MutableSet):
                items = list(enumerate(node))

                def set_op(container: MutableSet, idx: Any, value: ValidNodeType) -> None:
                    container.add(value)

                def del_op(container: MutableSet, idx: int) -> None:
                    container.remove(items[idx])  # type: ignore

            elif isinstance(node, collections.abc.MutableMapping):
                items = node.items()
                set_op = operator.setitem
                del_op = operator.delitem
            elif isinstance(node, (collections.abc.Sequence, collections.abc.Set)):
                # Inmutable sequence or set: create a new container instance with the new values
                tmp_items = [self.visit(value, **kwargs) for value in node]
                result = node.__class__(  # type: ignore
                    [value for value in tmp_items if value is not NOTHING]
                )
            elif isinstance(node, collections.abc.Mapping):
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
