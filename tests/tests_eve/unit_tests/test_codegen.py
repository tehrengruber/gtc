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
import eve.codegen


# -- Name tests --
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


def test_name(name_with_cases):
    name = eve.codegen.Name(name_with_cases.pop("words"))
    for case, cased_string in name_with_cases.items():
        assert name.as_case(case) == cased_string
        for other_case, other_cased_string in name_with_cases.items():
            if other_case == eve.utils.CaseStyleConverter.CASE_STYLE.CONCATENATED:
                with pytest.raises(ValueError, match="split a simply concatenated"):
                    other_name = eve.codegen.Name.from_string(other_cased_string, other_case)
                    assert other_name.as_case(case) == cased_string
            else:
                other_name = eve.codegen.Name.from_string(other_cased_string, other_case)
                assert other_name.as_case(case) == cased_string


# -- StringFormatter tests --
@pytest.fixture(params=[(), (0.12345, -1.12345, 2.12345, -3.12345, 4.12345), np.random.rand(5)])
def collection(request):
    yield request.param


@pytest.fixture(params=["", ",", " ", "[", "]", "::", "^^", "[[", "]]"])
def joiner(request):
    yield request.param


@pytest.fixture(params=[None, "", "{{}}", "{{:.5}}", "{{:*^5.3}}"])
def item_template(request):
    yield request.param


@pytest.fixture(params=["", "*^5", "*^5", "*^50"])
def collection_fmt(request):
    yield request.param


class TestStringFormatter:
    def test_without_extras(self, collection):
        fmt = eve.codegen.StringFormatter()
        std_fmt = string.Formatter()

        assert fmt.format("aA") == std_fmt.format("aA")
        assert fmt.format("a{}A", 0) == std_fmt.format("a{}A", 0)
        assert fmt.format("a{:*^5}A", 0) == std_fmt.format("a{:*^5}A", 0)
        assert fmt.format("a{:*^5.3}A", 0.12345) == std_fmt.format("a{:*^5.3}A", 0.12345)
        assert fmt.format("{data}", data=collection) == std_fmt.format("{data}", data=collection)
        assert fmt.format("{data:}", data=collection) == std_fmt.format("{data:}", data=collection)

    def test_collection(self, collection, joiner, item_template, collection_fmt):
        fmt = eve.codegen.StringFormatter()
        std_fmt = string.Formatter()

        std_joiner = joiner.replace("^^", "^").replace("::", ":")
        if item_template is None:
            spec = "{data:" + joiner + ":" + collection_fmt + "}"
            std_item_template = "{}"
        else:
            spec = "{data:" + joiner + "^[" + item_template + "]:" + collection_fmt + "}"
            std_item_template = item_template.replace("{{", "{").replace("}}", "}")

        std_spec = "{data:" + collection_fmt + "}"

        assert fmt.format(spec, data=collection) == std_fmt.format(
            std_spec, data=std_joiner.join(std_item_template.format(d) for d in collection)
        )

    def test_wrong_collection(self):
        fmt = eve.codegen.StringFormatter()
        with pytest.raises(ValueError, match="scalar value"):
            fmt.format("{data::}", data=1.34)


# -- Template tests --
def fmt_tpl_maker(skeleton, keys):
    transformed_keys = {k: "{{{}}}".format(k) for k in keys}
    return eve.codegen.StrFormatTemplate(skeleton.format(**transformed_keys))


def string_tpl_maker(skeleton, keys):
    transformed_keys = {k: "${}".format(k) for k in keys}
    return eve.codegen.StringTemplate(skeleton.format(**transformed_keys))


def jinja_tpl_maker(skeleton, keys):
    transformed_keys = {k: "{{{{ {} }}}}".format(k) for k in keys}
    return eve.codegen.JinjaTemplate(skeleton.format(**transformed_keys))


def mako_tpl_maker(skeleton, keys):
    transformed_keys = {k: "${{{}}}".format(k) for k in keys}
    return eve.codegen.MakoTemplate(skeleton.format(**transformed_keys))


@pytest.fixture(params=[fmt_tpl_maker, string_tpl_maker, jinja_tpl_maker, mako_tpl_maker])
def template_maker(request):
    yield request.param


def test_render_template(template_maker):
    skeleton = "aaa {s} bbbb {i} cccc"
    data = {"s": "STRING", "i": 1}
    template = template_maker(skeleton, data.keys())
    assert template.render(**data) == "aaa STRING bbbb 1 cccc"
    assert template.render(data, i=2) == "aaa STRING bbbb 2 cccc"


# -- TemplatedGenerator tests --
class _BaseTestGenerator(eve.codegen.TemplatedGenerator):
    KEYWORDS = ("BASE", "ONE")

    def visit_IntKind(self, node, **kwargs):
        return f"ONE INTKIND({node.value})"

    def visit_SourceLocation(self, node, **kwargs):
        return f"SourceLocation<line:{node.line}, column:{node.column}, source: {node.source}>"

    LocationNode = eve.codegen.StrFormatTemplate("LocationNode {{{loc}}}")

    SimpleNode = eve.codegen.JinjaTemplate(
        "|{{ bool_value }}, {{ int_value }}, {{ float_value }}, {{ str_value }}, {{ bytes_value }}, "
        "{{ int_kind }}, {{ _this_node.str_kind.__class__.__name__ }}|"
    )

    def visit_SimpleNode(self, node, **kwargs):
        return f"SimpleNode {{{self.generic_visit(node, **kwargs)}}}"

    CompoundNode = eve.codegen.MakoTemplate(
        """
----CompoundNode [BASE]----
    - location: ${location}
    - simple: ${simple}
    - simple_opt: <has_optionals ? (${_this_node.simple_opt.int_value is not None}, ${_this_node.simple_opt.float_value is not None}, ${_this_node.simple_opt.str_value is not None})>
    - other_simple_opt: <is_present ? ${_this_node.other_simple_opt is not None}>
"""
    )

    def visit_CompoundNode(self, node, **kwargs):
        return "TemplatedGenerator result:\n" + self.generic_visit(node, **kwargs)


class _InheritedTestGenerator(_BaseTestGenerator):
    KEYWORDS = ("INHERITED", "OTHER")

    def visit_IntKind(self, node, **kwargs):
        return f"OTHER INTKIND({node.value})"

    CompoundNode = eve.codegen.MakoTemplate(
        """
----CompoundNode [INHERITED]----
    - location: ${location}
    - simple: ${simple}
    - simple_opt: <has_optionals ? (${_this_node.simple_opt.int_value is not None}, ${_this_node.simple_opt.float_value is not None}, ${_this_node.simple_opt.str_value is not None})>
    - other_simple_opt: <is_present ? ${_this_node.other_simple_opt is not None}>
"""
    )


@pytest.fixture(params=[_BaseTestGenerator, _InheritedTestGenerator])
def templated_generator(request):
    yield request.param


def test_templated_generator(templated_generator, fixed_compound_node):
    rendered_code = templated_generator.apply(fixed_compound_node)
    assert rendered_code.find("TemplatedGenerator result:\n") >= 0
    assert rendered_code.find("----CompoundNode [") >= 0
    assert rendered_code.find("LocationNode {") >= 0
    assert rendered_code.find("SimpleNode {|") >= 0
    assert rendered_code.find("<has_optionals ? (True, True, False)>") >= 0
    assert rendered_code.find("<is_present ? False>") >= 0

    for keyword in templated_generator.KEYWORDS:
        assert rendered_code.find(keyword) >= 0
