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


from typing import Any, ClassVar, Dict, List, Type

import pydantic

from .concepts import _EVE_NODE_ATTR_SUFFIX, Node, Trait


class SymbolTableTrait(Trait):

    name: ClassVar[str] = "symbol_table"

    @classmethod
    def process_namespace(
        cls, namespace: Dict[str, Any], trait_names: List[str], meta_kwargs: Dict[str, Any]
    ) -> None:
        attr_name = f"symtable{_EVE_NODE_ATTR_SUFFIX}"
        namespace["__annotations__"][attr_name] = Dict[str, Node]
        namespace[attr_name] = pydantic.Field(default_factory=dict)

    @classmethod
    def process_class(
        cls, node_class: Type[Node], trait_names: List[str], meta_kwargs: Dict[str, Any]
    ) -> None:
        pass
