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

"""General utility functions."""


from typing import Sequence, Union

from boltons.strutils import (  # noqa: F401
    a10n,
    args2cmd,
    args2sh,
    asciify,
    bytes2human,
    camel2under,
    cardinalize,
    escape_shell_args,
    find_hashtags,
    format_int_list,
    indent,
    is_ascii,
    is_uuid,
    iter_splitlines,
    ordinalize,
    parse_int_list,
    pluralize,
    singularize,
    slugify,
    split_punct_ws,
    strip_ansi,
    under2camel,
    unit_len,
    unwrap_text,
)


WordsSequenceType = Union[str, Sequence[str]]


def join_canonical_cased(words: WordsSequenceType):
    words = [words] if isinstance(words, str) else words
    return (" ".join(words)).lower()


def join_concatcased(words: WordsSequenceType):
    words = [words] if isinstance(words, str) else words
    return "".join(words)


def join_camelCased(words: WordsSequenceType):
    words = [words] if isinstance(words, str) else words
    return words[0] + "".join(word.title() for word in words[1:])


def join_CamelCased(words: WordsSequenceType):
    words = [words] if isinstance(words, str) else words
    return "".join(word.title() for word in words)


def join_SNAKE_CASED(words: WordsSequenceType):
    words = [words] if isinstance(words, str) else words
    return "_".join(word for word in words).upper()


def join_snake_cased(words: WordsSequenceType):
    words = [words] if isinstance(words, str) else words
    return "_".join(words)
