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
import collections.abc
import os
import string
import textwrap
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Sequence, Union

import jinja2

from .core import BaseNode, NodeVisitor, StrEnum, ValidNodeType


TextSequenceType = Union[Sequence[str], "TextBlock"]


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
        self.lines: List[str] = []

    def append(self, new_line: str, *, update_indent: int = 0) -> "TextBlock":
        if update_indent > 0:
            self.indent(update_indent)
        elif update_indent < 0:
            self.dedent(-update_indent)

        self.lines.append(self.indent_str + new_line)

        return self

    def extend(
        self, new_lines: Union[Sequence[str], "TextBlock"], *, dedent: bool = False
    ) -> "TextBlock":
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
        lines = ["".join([str(item) for item in line]) for line in self.lines]
        return self.end_line.join(lines)

    @property
    def indent_str(self):
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
    TPL = "tpl"


ValidTemplateType = Union[str, jinja2.Template, string.Template]


class TextTemplate:
    """A generic text template class abstracting different template engines."""

    KIND_TO_TYPES_MAPPING: ClassVar[Mapping[TemplateKind, ValidTemplateType]] = MappingProxyType(
        {
            TemplateKind.FMT: str,
            TemplateKind.JINJA: jinja2.Template,
            TemplateKind.TPL: string.Template,
        }
    )

    TYPES_TO_KIND_MAPPING: ClassVar[Mapping[ValidTemplateType, TemplateKind]] = MappingProxyType(
        {type_.__name__: kind for kind, type_ in KIND_TO_TYPES_MAPPING.items()}
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

    def __init__(self, definition: ValidTemplateType, kind: Optional[TemplateKind] = None):
        if kind is None:
            kind = self.TYPES_TO_KIND_MAPPING.get(type(definition).__name__, None)
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
        return self.definition.format(**kwargs)  # type: ignore

    def render_jinja(self, **kwargs) -> str:
        return self.definition.render(**kwargs)  # type: ignore

    def render_tpl(self, **kwargs) -> str:
        return self.definition.substitute(**kwargs)  # type: ignore


class NodeDumper(NodeVisitor):
    @classmethod
    def apply(
        cls,
        root: ValidNodeType,
        *,
        node_templates: Optional[Dict[str, Union[TextTemplate, ValidTemplateType]]] = None,
        dump_func: Optional[Callable[[ValidNodeType], str]] = None,
        **kwargs,
    ) -> str:
        return cls(node_templates, dump_func).visit(root, **kwargs)

    def __init__(
        self,
        node_templates: Optional[Dict[str, Union[TextTemplate, ValidTemplateType]]] = None,
        dump_func: Optional[Callable[[ValidNodeType], str]] = None,
    ):
        node_templates = node_templates or {}
        self.node_templates = {
            key: value if isinstance(value, TextTemplate) else TextTemplate(value)
            for key, value in node_templates.items()
        }
        self.dump_func = dump_func

    def generic_visit(self, node: ValidNodeType, **kwargs) -> str:
        template: TextTemplate = self.node_templates.get(node.__class__.__name__, None)
        attrs_strings: Dict[str, Any] = {}
        child_strings: Dict[str, Any] = {}

        if isinstance(node, (BaseNode, collections.abc.Collection)) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            if isinstance(node, BaseNode):
                attrs_strings = {
                    key: self.visit(value, **kwargs) for key, value in node.attributes()
                }
                child_strings = {key: self.visit(value, **kwargs) for key, value in node.children()}

            elif isinstance(node, (collections.abc.Sequence, collections.abc.Set)):
                child_strings = {
                    f"_{i}": self.visit(value, **kwargs) for i, value in enumerate(node)
                }

            elif isinstance(node, collections.abc.Mapping):
                child_strings = {key: self.visit(value, **kwargs) for key, value in node.items()}

        if template:
            return template.render(
                _instance=node,
                _this=SimpleNamespace(**child_strings),  # type: ignore
                _attrs=SimpleNamespace(**attrs_strings),  # type: ignore
                **attrs_strings,
                **child_strings,
            )
        elif self.dump_func:
            return self.dump_func(child_strings or node)  # type: ignore
        else:
            return ""


class TemplatedGenerator(NodeDumper):

    NODE_TEMPLATES = None
    DUMP_FUNCTION = None

    @classmethod
    def apply(cls, root: ValidNodeType, **kwargs) -> str:
        return super().apply(
            root, node_templates=cls.NODE_TEMPLATES, dump_func=cls.DUMP_FUNCTION, **kwargs
        )
