import sys
import ast
import inspect
import typing
import typing_inspect
from . import gtscript
from . import ast_node_matcher as anm
from .ast_node_matcher import Capture
from eve import UIDGenerator
import enum
import gtc.common

class PyToGTScript:
    @staticmethod
    def _all_subclasses(typ, *, module=None):
        """
            Return all subclasses of a given type. The type must be one of
             - GT4PyAstNode (returns all subclasses of the given class)
             - Union (return the subclasses of the united)
             - ForwardRef (resolve the reference given the specified module and return its subclasses)
             - built-in python type: str, int, type(None) (return as is)
        """
        if inspect.isclass(typ) and issubclass(typ, gtscript.GTScriptAstNode):
            result = set()
            result.add(typ)
            result.update(typ.__subclasses__())
            result.update([s for c in typ.__subclasses__() for s in PyToGTScript._all_subclasses(c) if
                           not inspect.isabstract(c)])
            return result
        elif inspect.isclass(typ) and typ in [gtc.common.AssignmentKind, gtc.common.UnaryOperator, gtc.common.BinaryOperator]:
            # note: other types in gtc.common, e.g. gtc.common.DataType are not valid leaf nodes here as they
            #  map to symbols in the gtscript ast and are resolved there
            assert issubclass(typ, enum.Enum)
            return set([typ])
        elif typing_inspect.is_union_type(typ):
            result = set()
            for el_cls in typing_inspect.get_args(typ):
                result.update(PyToGTScript._all_subclasses(el_cls, module=module))
            return result
        elif isinstance(typ, typing.ForwardRef):
            type_name = typing_inspect.get_forward_arg(typ)
            if not hasattr(module, type_name):
                raise ValueError(
                    "Reference to type `{}` in `ForwardRef` not found in module {}".format(type_name, module.__name__))
            return PyToGTScript._all_subclasses(getattr(module, type_name), module=module)
        elif typ in [str, int, float, type(None)]:  # todo: enhance
            return set([typ])

        raise ValueError("Invalid field type {}".format(typ))

    class Templates:
        Name = ast.Name(id=Capture("id"))

        IterationOrder = ast.withitem(
            context_expr=ast.Call(func=ast.Name(id="computation"), args=[ast.Name(id=Capture("order"))]))

        Constant = ast.Constant(value=Capture("value"))

        Interval = ast.withitem(context_expr=ast.Call(func=ast.Name(id="interval"),
                                                      args=[Capture("start"), Capture("stop")]))

        LocationSpecification = ast.withitem(
            context_expr=ast.Call(func=ast.Name(id="location"), args=[ast.Name(id=Capture("location_type"))]),
            optional_vars=Capture("name", default=ast.Name(id=UIDGenerator.get_unique_id(prefix="location"))))

        SubscriptSingle = ast.Subscript(value=Capture("value"),
                                        slice=ast.Index(value=ast.Name(id=Capture("index"))))

        SubscriptMultiple = ast.Subscript(value=Capture("value"),
                                          slice=ast.Index(value=ast.Tuple(elts=Capture("indices"))))

        BinaryOp = ast.BinOp(op=Capture("op"), left=Capture("left"), right=Capture("right"))

        Call = ast.Call(args=Capture("args"), func=ast.Name(id=Capture("func")))

        LocationComprehension = ast.comprehension(target=Capture("target"), iter=Capture("iter"))

        Generator = ast.GeneratorExp(generators=Capture("generators"), elt=Capture("elt"))

        Assign = ast.Assign(targets=[Capture("target")], value=Capture("value"))

        Stencil = ast.With(items=Capture("iteration_spec"), body=Capture("body"))

        Pass = ast.Pass()

        Argument = ast.arg(arg=Capture("name"), annotation=Capture("type"))

        Computation = ast.FunctionDef(args=ast.arguments(args=Capture("arguments")), body=Capture("stencils"),
                                      name=Capture("name"))

    leaf_map = {
        ast.Mult : gtc.common.BinaryOperator.MUL,
        ast.Add : gtc.common.BinaryOperator.ADD,
        ast.Div : gtc.common.BinaryOperator.DIV
    }

    def transform(self, node, eligible_node_types=[gtscript.Computation]):
        """
        Transform python ast into GTScript ast recursively
        """
        if isinstance(node, ast.AST):
            is_leaf_node = len(list(ast.iter_fields(node)))==0
            if is_leaf_node:
                if not type(node) in self.leaf_map:
                    raise ValueError("Leaf node of type {} found in the python ast can not be mapped.")
                return self.leaf_map[type(node)]
            else:
                # visit node fields and transform
                # todo: check if multiple nodes match and throw an error in that case
                # disadvantage: templates can be ambiguous
                for node_type in eligible_node_types:
                    if not hasattr(self.Templates, node_type.__name__):
                        continue
                    captures = {}
                    if not anm.match(node, getattr(self.Templates, node_type.__name__), captures=captures):
                        continue
                    module = sys.modules[node_type.__module__]
                    transformed_captures = {}
                    for name, capture in captures.items():
                        field_type = node_type.__annotations__[name]
                        if typing_inspect.get_origin(field_type) == list:
                            # determine eligible capture types
                            el_type = typing_inspect.get_args(field_type)[0]
                            eligible_capture_types = self._all_subclasses(el_type, module=module)

                            # transform captures recursively
                            transformed_captures[name] = []
                            for child_capture in capture:
                                transformed_captures[name].append(self.transform(child_capture, eligible_capture_types))
                        else:
                            # determine eligible capture types
                            eligible_capture_types = self._all_subclasses(field_type, module=module)
                            # transform captures recursively
                            transformed_captures[name] = self.transform(capture, eligible_capture_types)
                    return node_type(**transformed_captures)
                raise ValueError("Expected a node of type {}".format(
                    ", ".join([ent.__name__ for ent in eligible_node_types])))
        elif type(node) in eligible_node_types:
            return node

        raise ValueError("Expected a node of type {}, but got {}".format(eligible_node_types + [ast.AST], type(node)))
