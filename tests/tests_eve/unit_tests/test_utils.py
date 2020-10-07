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

import string

import numpy as np
import pytest

import eve
import eve.utils


@pytest.mark.parametrize("nfuncs", [0, 1, 2, 3, 10])
def test_call_all(nfuncs):
    fs = []
    locals_dict = {}
    for i in range(nfuncs):
        exec(
            f"""
def func_{i}_definition(a):
    a.append({i})
""",
            {},
            locals_dict,
        )
    fs.extend([value for key, value in locals_dict.items() if key.startswith("func_")])
    composite_f = eve.utils.call_all(fs)
    result = []
    composite_f(result)
    assert result == list(range(nfuncs))


# -- CaseStyleConverter --
@pytest.fixture
def name_with_cases():
    from eve.utils import CaseStyleConverter

    cases = {
        "words": ["first", "second", "UPPER", "Title"],
        CaseStyleConverter.CASE_STYLE.CONCATENATED: "firstseconduppertitle",
        CaseStyleConverter.CASE_STYLE.CANONICAL: "first second upper title",
        CaseStyleConverter.CASE_STYLE.CAMEL: "firstSecondUpperTitle",
        CaseStyleConverter.CASE_STYLE.PASCAL: "FirstSecondUpperTitle",
        CaseStyleConverter.CASE_STYLE.SNAKE: "first_second_upper_title",
        CaseStyleConverter.CASE_STYLE.KEBAB: "first-second-upper-title",
    }

    yield cases


def test_case_style_converter(name_with_cases):
    from eve.utils import CaseStyleConverter

    words = name_with_cases.pop("words")
    for case, cased_string in name_with_cases.items():
        assert CaseStyleConverter.join(words, case) == cased_string
        if case == CaseStyleConverter.CASE_STYLE.CONCATENATED:
            with pytest.raises(ValueError, match="Impossible to split"):
                CaseStyleConverter.split(cased_string, case)
        else:
            assert [w.lower() for w in CaseStyleConverter.split(cased_string, case)] == [
                w.lower() for w in words
            ]


# -- FStringFormatter tests --
@pytest.fixture(params=[(), (0.12345, -1.12345, 2.12345, -3.12345, 4.12345), np.random.rand(5)])
def data_collection(request):
    yield request.param


# #f"abc {a['x']} def"
# f'result={foo()}'
# f'''{x
# ... +1}'''
# f'{fn(lst,2)} {fn(lst,3)}'
# f'{fn(lst,2)} {fn(lst,3)}'


@pytest.fixture(params=[(), (0.12345, -1.12345, 2.12345, -3.12345, 4.12345), np.random.rand(5)])
def fstr_definitions(request):
    context = dict()
    yield (request.param, context)


class TestXStringFormatter:
    def test_without_extras(self, data_collection):
        fmt = eve.utils.XStringFormatter()
        std_fmt = string.Formatter()

        assert fmt.format("aA") == std_fmt.format("aA")
        assert fmt.format("a{}A", 0) == std_fmt.format("a{}A", 0)
        assert fmt.format("a{:*^5}A", 0) == std_fmt.format("a{:*^5}A", 0)
        assert fmt.format("a{:*^5.3}A", 0.12345) == std_fmt.format("a{:*^5.3}A", 0.12345)
        assert fmt.format("{data}", data=data_collection) == std_fmt.format(
            "{data}", data=data_collection
        )
        assert fmt.format("{data:}", data=data_collection) == std_fmt.format(
            "{data:}", data=data_collection
        )

    def test_with_expressions(self):
        fmt = eve.utils.XStringFormatter()
        std_fmt = string.Formatter()

        assert fmt.format("aA") == std_fmt.format("aA")
        assert fmt.format("a{}A", 0) == std_fmt.format("a{}A", 0)
        assert fmt.format("a{:*^5}A", 0) == std_fmt.format("a{:*^5}A", 0)
        assert fmt.format("a{:*^5.3}A", 0.12345) == std_fmt.format("a{:*^5.3}A", 0.12345)
        assert fmt.format("{data}", data=data_collection) == std_fmt.format(
            "{data}", data=data_collection
        )
        assert fmt.format("{data:}", data=data_collection) == std_fmt.format(
            "{data:}", data=data_collection
        )

    # def test_collection(self, collection, joiner, item_template, collection_fmt):
    #     fmt = eve.codegen.StringFormatter()
    #     std_fmt = string.Formatter()

    #     std_joiner = joiner.replace("^^", "^").replace("::", ":")
    #     if item_template is None:
    #         spec = "{data:" + joiner + ":" + collection_fmt + "}"
    #         std_item_template = "{}"
    #     else:
    #         spec = "{data:" + joiner + "^[" + item_template + "]:" + collection_fmt + "}"
    #         std_item_template = item_template.replace("{{", "{").replace("}}", "}")

    #     std_spec = "{data:" + collection_fmt + "}"

    #     assert fmt.format(spec, data=collection) == std_fmt.format(
    #         std_spec, data=std_joiner.join(std_item_template.format(d) for d in collection)
    #     )


#     def test_wrong_collection(self):
#         fmt = eve.codegen.StringFormatter()
#         with pytest.raises(ValueError, match="scalar value"):
#             fmt.format("{data::}", data=1.34)
