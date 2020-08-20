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
class BuiltInTypeMeta(type):
    def __new__(cls, class_name, bases, namespace, args=None):
        assert bases == () or (len(bases)==1 and issubclass(bases[0], BuiltInType))
        assert all(attr[0:2] == "__" for attr in namespace.keys()) # no custom attributes
        # tülülü
        instance = type.__new__(cls, class_name, bases, namespace)
        instance.class_name = class_name
        instance.namespace = namespace
        instance.args = args
        return instance

    def __getitem__(self, args): # todo: evaluate __class_getitem__
        if not isinstance(args, tuple):
            args = (args,)
        return BuiltInTypeMeta(self.class_name, (), self.namespace, args=args)

    def __instancecheck__(self, instance):
        raise RuntimeError()

    def __subclasscheck__(self, other):
        # todo: enhance
        if isinstance(other, BuiltInTypeMeta) and self.namespace == other.namespace and self.class_name == other.class_name:
            if self.args == None or self.args == other.args:
                return True
        return False

class BuiltInType(metaclass=BuiltInTypeMeta):
    pass

class Mesh(BuiltInType):
    pass

class Field(BuiltInType):
    pass

class TemporaryField(BuiltInType): # todo: make this a subtype of Field
    pass

class Location(BuiltInType):
    pass

# LocalDimension
class Local(BuiltInType):
    pass
