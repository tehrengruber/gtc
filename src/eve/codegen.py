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

"""Tools for code generation."""


import collections.abc
import contextlib
import os
import string
import sys
import textwrap
from types import MappingProxyType
from typing import Any, ClassVar, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import jinja2
from mako import template as mako_tpl

from .core import BaseNode, NodeVisitor, StrEnum, ValidNodeType


ValidNameDefType = Union[str, Sequence[str]]


def make_canonical_cased(words: ValidNameDefType):
    words = [words] if isinstance(words, str) else words
    return (" ".join(words)).lower()


def make_concatcased(words: ValidNameDefType):
    words = [words] if isinstance(words, str) else words
    return "".join(words)


def make_camelCased(words: ValidNameDefType):
    words = [words] if isinstance(words, str) else words
    return words[0] + "".join(word.title() for word in words[1:])


def make_CamelCased(words: ValidNameDefType):
    words = [words] if isinstance(words, str) else words
    return "".join(word.title() for word in words)


def make_SNAKE_CASED(words: ValidNameDefType):
    words = [words] if isinstance(words, str) else words
    return "_".join(word.upper() for word in words)


def make_snake_cased(words: ValidNameDefType):
    words = [words] if isinstance(words, str) else words
    return "_".join(words)


class Identifier:
    """Text string representing a symbol name in a programming language."""

    # Based on code from :https://blog.kangz.net/posts/2016/08/31/code-generation-the-easier-way/

    def __init__(self, name: ValidNameDefType):
        if isinstance(name, collections.abc.Sequence):
            if not all(isinstance(item, str) for item in name):
                raise TypeError(
                    f"Identifier definition ('{name}') type is not 'Union[str, Sequence[str]]'"
                )
            self.words = name
        elif isinstance(name, str):
            self.words = [name]
        else:
            raise TypeError(
                f"Identifier definition ('{name}') type is not 'Union[str, Sequence[str]]'"
            )

    def as_canonical_cased(self):
        return self.make_canonical_cased(self.words)

    def as_concatcased(self):
        return self.make_concatcased(self.words)

    def as_camelCased(self):
        return self.make_camelCased(self.words)

    def as_CamelCased(self):
        return self.make_CamelCased(self.words)

    def as_SNAKE_CASE(self):
        return self.make_SNAKE_CASED(self.words)

    def as_snake_cased(self):
        return self.make_snake_cased(self.words)


TextSequenceType = Union[Sequence[str], "TextBlock"]


class TextBlock:
    """A block of source code represented as a sequence of text lines.

    This class also contains a context manager method (:meth:`indented`)
    for simple `indent - append - dedent` workflows.

    Args:
        indent_level: Initial indentation level
        indent_size: Number of characters per indentation level
        indent_char: Character used in the indentation
        end_line: Character or string used as new-line separator

    """

    def __init__(
        self,
        *,
        indent_level: int = 0,
        indent_size: int = 4,
        indent_char: str = " ",
        end_line: str = "\n",
    ):
        if not isinstance(indent_char, str) or len(indent_char) != 1:
            raise ValueError("'indent_char' must be a single-character string")
        if not isinstance(end_line, str):
            raise ValueError("'end_line' must be a string")

        self.indent_level = indent_level
        self.indent_size = indent_size
        self.indent_char = indent_char
        self.end_line = end_line
        self.lines: List[str] = []

    def append(self, new_line: str, *, update_indent: int = 0) -> "TextBlock":
        if update_indent > 0:
            self.indent(update_indent)
        elif update_indent < 0:
            self.dedent(-update_indent)

        self.lines.append(self.indent_str + new_line)

        return self

    def extend(self, new_lines: TextSequenceType, *, dedent: bool = False) -> "TextBlock":
        assert isinstance(new_lines, (collections.abc.Sequence, TextBlock))

        if dedent:
            if isinstance(new_lines, TextBlock):
                new_lines = textwrap.dedent(new_lines.text).splitlines()
            else:
                new_lines = textwrap.dedent("\n".join(new_lines)).splitlines()

        elif isinstance(new_lines, TextBlock):
            new_lines = new_lines.lines

        for line in new_lines:
            self.append(line)

        return self

    def empty_line(self, count: int = 1) -> "TextBlock":
        self.lines.extend([""] * count)
        return self

    def indent(self, steps: int = 1) -> "TextBlock":
        self.indent_level += steps
        return self

    def dedent(self, steps: int = 1) -> "TextBlock":
        assert self.indent_level >= steps
        self.indent_level -= steps
        return self

    @contextlib.contextmanager
    def indented(self, steps: int = 1):
        self.indent(steps)
        yield self
        self.dedent(steps)

    @property
    def text(self) -> str:
        """Single string with the whole block contents."""
        lines = ["".join([str(item) for item in line]) for line in self.lines]
        return self.end_line.join(lines)

    @property
    def indent_str(self):
        """Indentation string for new lines (in the current state)."""
        return self.indent_char * (self.indent_level * self.indent_size)

    def __iadd__(self, source_line: str):
        return self.append(source_line)

    def __len__(self):
        return len(self.lines)

    def __str__(self):
        return self.text


class TemplateKind(StrEnum):
    """Supported template kinds."""

    FMT = "fmt"
    JINJA = "jinja"
    MAKO = "mako"
    TPL = "tpl"


ValidTemplateDefType = Union[str, jinja2.Template, mako_tpl.Template, string.Template]


class TextTemplate:
    """A generic text template class abstracting different template engines.

    This class supports all the template types represented in the
    :class:`TemplateKind` class.

    Args:
        definition: Template definition (actual type depends on the kind)
        kind: Template engine. It not provided, is guessed based on the
            type of the ``definition`` value

    """

    KIND_TO_TYPES_MAPPING: ClassVar[Mapping[TemplateKind, ValidTemplateDefType]] = MappingProxyType(
        {
            TemplateKind.FMT: str,
            TemplateKind.JINJA: jinja2.Template,
            TemplateKind.MAKO: mako_tpl.Template,
            TemplateKind.TPL: string.Template,
        }
    )

    TYPES_TO_KIND_MAPPING: ClassVar[Mapping[ValidTemplateDefType, TemplateKind]] = MappingProxyType(
        {
            f"{type_.__module__}.{type_.__name__}": kind
            for kind, type_ in KIND_TO_TYPES_MAPPING.items()
        }
    )

    @classmethod
    def from_file(cls, file_path: Union[str, os.PathLike], kind: TemplateKind) -> "TextTemplate":
        with open(file_path, "r") as f:
            definition = f.read()
        return cls(definition, kind)

    @classmethod
    def from_fmt(cls, definition: str) -> "TextTemplate":
        return cls(definition, TemplateKind.FMT)

    @classmethod
    def from_jinja(cls, definition: Union[str, jinja2.Template]) -> "TextTemplate":
        if isinstance(definition, str):
            definition = jinja2.Template(definition)
        return cls(definition, TemplateKind.JINJA)

    @classmethod
    def from_mako(cls, definition: Union[str, mako_tpl.Template]) -> "TextTemplate":
        if isinstance(definition, str):
            definition = mako_tpl.Template(definition)
        return cls(definition, TemplateKind.MAKO)

    @classmethod
    def from_tpl(cls, definition: Union[str, string.Template]) -> "TextTemplate":
        if isinstance(definition, str):
            definition = string.Template(definition)
        return cls(definition, TemplateKind.TPL)

    def __init__(self, definition: ValidTemplateDefType, kind: Optional[TemplateKind] = None):
        if kind is None:
            kind = self.TYPES_TO_KIND_MAPPING.get(
                f"{type(definition).__module__}.{type(definition).__name__}", None
            )
        if not isinstance(kind, TemplateKind):
            raise TypeError(
                f"'kind' argument must be an instance of {TemplateKind} ({type(kind)} provided))"
            )
        if not isinstance(definition, self.KIND_TO_TYPES_MAPPING[kind]):
            raise TypeError(
                f"Invalid 'definition' type for '{kind}' template kind ({type(definition)})"
            )
        self.kind = kind
        self.definition = definition

    def render(self, mapping: Optional[Mapping[str, str]] = None, **kwargs) -> str:
        """Render the template.

        Args:
            mapping (optional): A `dict` whose keys match the template placeholders.
            **kwargs: placeholder values might be also provided as
                keyword arguments, and they take precedence over ``mapping``
                values for duplicated keys.

        """
        if isinstance(mapping, dict):
            mapping.update(kwargs)
        elif mapping is None:
            mapping = kwargs
        else:
            raise TypeError(
                f"'mapping' argument must be an instance of 'dict' ({type(mapping)} provided))"
            )

        return getattr(self, f"render_{self.kind}")(**mapping)

    def render_fmt(self, **kwargs) -> str:
        return self.definition.format(**kwargs)  # type: ignore

    def render_jinja(self, **kwargs) -> str:
        return self.definition.render(**kwargs)  # type: ignore

    def render_mako(self, **kwargs) -> str:
        return self.definition.render(**kwargs)  # type: ignore

    def render_tpl(self, **kwargs) -> str:
        return self.definition.substitute(**kwargs)  # type: ignore


TemplateType = Union[TextTemplate, ValidTemplateDefType]


class TemplatedGenerator(NodeVisitor):
    """A code generator visitor using :class:`TextTemplate` s."""

    class Templates:
        """Class-specific template definitions.

        This  class should be redefined in the subclasses with
        actual template definitions.

        """

        pass

    @classmethod
    def generic_dump(cls, node: ValidNodeType, **kwargs):
        """Class-specific ``dump()`` function for primitive types.

        This class could be redefined in the subclasses.
        """
        return str(node)

    @classmethod
    def apply(cls, root: ValidNodeType, **kwargs) -> str:
        """Public method to build a class instance and visit an IR node.

        The order followed to choose a `dump()` function for instances of
        :class:`eve.BaseNode` is the following:

            1. A `self.visit_NODE_TYPE_NAME()` method where `NODE_TYPE_NAME`
               matches `NODE_CLASS.__name__`, and `NODE_CLASS` is the
               actual type of the node or any of its superclasses
               following MRO order.
            2. A `Templates.NODE_TYPE_NAME` template where `NODE_TYPE_NAME`
               matches `NODE_CLASS.__name__`, and `NODE_CLASS` is the
               actual type of the node or any of its superclasses
               following MRO order.

        When a template is used, the following keys will be passed to the template
        instance:

            * `**node_fields`: all the node children and attributes by name.
            * `_attrs`: a `dict` instance with the results of visiting all
              the node attributes.
            * `_children`: a `dict` instance with the results of visiting all
              the node children.
            * `_this_node`: the actual node instance (before visiting children).
            * `_this_generator`: the current generator instance.
            * `_this_module`: the generator's module instance .
            * `**kwargs`: the keyword arguments received by the visiting method.

        For primitive types (not :class:`eve.BaseNode` subclasses),
        the :meth:`self.generic_dump()` method will be used.

        Args:
            root: An IR node.
            node_templates (optiona): see :class:`NodeDumper`.
            dump_function (optiona): see :class:`NodeDumper`.
            **kwargs (optional): custom extra parameters forwarded to
                `visit_NODE_TYPE_NAME()`.

        Returns:
            String (or collection of strings) with the dumped version of the root IR node.

        """
        return cls().visit(root, **kwargs)

    def __init__(self):
        cls = type(self)
        if not hasattr(cls, "Templates"):
            raise AttributeError(f"Missing required 'Templates' attribute in class {cls}")
        if not isinstance(cls.Templates, type):
            raise TypeError(f"Required 'Templates' attribute is not class ('{cls.Templates})")

        self._templates = {
            key: value if isinstance(value, TextTemplate) else TextTemplate(value)
            for key, value in cls.Templates.__dict__.items()
            if not key.startswith("__")
        }

    def generic_visit(self, node: ValidNodeType, **kwargs) -> Union[str, Collection[str]]:
        result = ""
        if isinstance(node, BaseNode):
            template, _ = self.get_template(node)
            if template:
                result = self.render_template(
                    template,
                    node,
                    self.transform_children(node, **kwargs),
                    self.transform_attrs(node, **kwargs),
                    **kwargs,
                )
        elif isinstance(node, (collections.abc.Sequence, collections.abc.Set)) and not isinstance(
            node, self.ATOMIC_COLLECTION_TYPES
        ):
            result = [self.visit(value, **kwargs) for value in node]
        elif isinstance(node, collections.abc.Mapping):
            result = {key: self.visit(value, **kwargs) for key, value in node.items()}
        else:
            result = self.generic_dump(node, **kwargs)

        return result

    def get_template(self, node: ValidNodeType) -> Tuple[Optional[TextTemplate], Optional[str]]:
        """Get a template for a node instance (see :meth:`apply`)."""
        template: Optional[TextTemplate] = None
        template_key: Optional[str] = None
        if isinstance(node, BaseNode):
            for node_class in node.__class__.__mro__:
                template_key = node_class.__name__
                template = self._templates.get(template_key, None)
                if template is not None or node_class is BaseNode:
                    break

        return template, None if template is None else template_key

    def render_template(
        self,
        template: TextTemplate,
        node: ValidTemplateDefType,
        transformed_children: Mapping[str, Any],
        transformed_attrs: Mapping[str, Any],
        **kwargs,
    ) -> str:
        """Render a template using node instance data (see :meth:`apply`)."""

        return template.render(
            **transformed_children,
            **transformed_attrs,
            _children=transformed_children,  # type: ignore
            _attrs=transformed_attrs,  # type: ignore
            _this_node=node,
            _this_generator=self,
            _this_module=sys.modules[type(self).__module__],
            **kwargs,
        )

    def transform_children(self, node: ValidNodeType, **kwargs) -> Dict[str, Any]:
        return {key: self.visit(value, **kwargs) for key, value in node.children()}

    def transform_attrs(self, node: ValidNodeType, **kwargs) -> Dict[str, Any]:
        return {key: self.visit(value, **kwargs) for key, value in node.attributes()}
