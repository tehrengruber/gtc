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


import contextlib
import os
import string
import textwrap
from collections import abc as col_abc
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Dict, Optional, Sequence, Union

import jinja2

from .core import Node, NodeVisitor, StrEnum


TextLineType = Union[str, Sequence[Any]]
TextSequenceType = Union[TextLineType, Sequence[TextLineType], "TextBlock"]


class TextBlock:
    """A block of source code represented as a sequence of text lines."""

    def __init__(
        self,
        *,
        indent_level: int = 0,
        indent_size: int = 4,
        indent_char: str = " ",
        end_line: str = "\n",
    ):
        self.indent_level = indent_level
        self.indent_size = indent_size
        self.indent_char = indent_char
        self.end_line = end_line
        self.lines = []

    def append(self, new_line: TextLineType, *, update_indent: int = 0) -> "TextBlock":
        if update_indent > 0:
            self.indent(update_indent)
        elif update_indent < 0:
            self.dedent(-update_indent)

        if isinstance(new_line, str):
            new_line = [new_line]

        built_line = [self.indent_char * (self.indent_level * self.indent_size)]
        for item in new_line:
            if isinstance(item, str) and isinstance(built_line[-1], str):
                built_line[-1] += item
            else:
                built_line.append(item)

        self.lines.append(built_line)

        return self

    def extend(self, new_lines: TextSequenceType, *, dedent: bool = False) -> "TextBlock":
        assert isinstance(new_lines, (str, col_abc.Sequence, TextBlock))

        if dedent:
            if isinstance(new_lines, TextBlock):
                new_lines = new_lines.text
            elif not isinstance(new_lines, str):
                new_lines = "\n".join(new_lines)
            new_lines = textwrap.dedent(new_lines)

        if isinstance(new_lines, str):
            new_lines = new_lines.splitlines()
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
        lines = ["".join([str(item) for item in line]) for line in self.lines]
        return self.end_line.join(lines)

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
    TPL = "tpl"


ValidTemplateType = Union[str, jinja2.Template, string.Template]


class TextTemplate:
    """A generic text template class abstracting different template engines."""

    _TEMPLATE_TYPES = MappingProxyType(
        {
            TemplateKind.FMT: str,
            TemplateKind.JINJA: jinja2.Template,
            TemplateKind.TPL: string.Template,
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
    def from_tpl(cls, definition: Union[str, string.Template]) -> "TextTemplate":
        if isinstance(definition, str):
            definition = string.Template(definition)
        return cls(definition, TemplateKind.TPL)

    def __init__(self, definition: ValidTemplateType, kind: TemplateKind):
        if not isinstance(kind, TemplateKind):
            raise TypeError(
                f"'kind' argument must be an instance of {TemplateKind} ({type(kind)} provided))"
            )
        if not isinstance(definition, self._TEMPLATE_TYPES[kind]):
            raise TypeError(
                f"Invalid 'definition' type for '{kind}' template kind ({type(definition)})"
            )
        self.kind = kind
        self.definition = definition

    def render(self, mapping: Optional[Dict[str, str]] = None, **kwargs) -> str:
        """Render the template.

        `mapping` is a `dict` object with keys matching the template placeholders.
        Alternatively, placeholder values might be provided using keyword arguments
        (kwargs take precedence over mapping values for duplicated keys).

        """
        if isinstance(mapping, dict):
            mapping.update(kwargs)
        elif mapping is None:
            mapping = kwargs
        else:
            raise TypeError(
                f"'mappint' argument must be an instance of 'dict' ({type(mapping)} provided))"
            )

        return getattr(self, f"render_{self.kind}")(**mapping)

    def render_fmt(self, **kwargs) -> str:
        return self.definition.format(**kwargs)

    def render_jinja(self, **kwargs) -> str:
        return self.definition.render(**kwargs)

    def render_tpl(self, **kwargs) -> str:
        return self.definition.substitute(**kwargs)


class NodeDumper(NodeVisitor):
    @classmethod
    def apply(
        cls,
        root: Node,
        *,
        node_templates: Optional[Dict[str, TextTemplate]] = None,
        dump_func: Optional[Callable[[Node], str]] = None,
        **kwargs,
    ) -> str:
        return cls(node_templates, dump_func).visit(root, **kwargs)

    def __init__(
        self,
        node_templates: Optional[Dict[str, TextTemplate]] = None,
        dump_func: Optional[Callable[[Node], str]] = None,
    ):
        self.node_templates = node_templates or {}
        self.dump_func = dump_func if dump_func is not None else str

    def generic_visit(self, node: Node, **kwargs):
        attrs = {}
        this = node
        template = self.node_templates.get(node.__class__.__name__, None)
        template_kwargs = {}

        if isinstance(node, (Node, col_abc.Collection)) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            if isinstance(node, Node):
                attrs = {key: self.visit(value, **kwargs) for key, value in node.iter_attributes()}
                this = {key: self.visit(value, **kwargs) for key, value in node.iter_children()}

            elif isinstance(node, (col_abc.Sequence, col_abc.Set)):
                this = {f"_{i}": self.visit(value, **kwargs) for i, value in enumerate(node)}

            elif isinstance(node, col_abc.Mapping):
                this = {key: self.visit(value, **kwargs) for key, value in node.items()}

        if template:
            template_kwargs.update(attrs)
            if isinstance(this, dict):
                template_kwargs.update(this)

            return template.render(
                _node_instance=node,
                _this=SimpleNamespace(**this),
                _attrs=SimpleNamespace(**attrs),
                **template_kwargs,
            )
        else:
            return self.dump_func(this)


class TemplatedGenerator(NodeDumper):

    NODE_TEMPLATES = None
    DUMP_FUNCTION = None

    @classmethod
    def apply(cls, root: Node, **kwargs) -> str:
        return super().apply(
            root, node_templates=cls.NODE_TEMPLATES, dump_func=cls.DUMP_FUNCTION, **kwargs
        )
