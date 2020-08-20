from typing import Tuple, Dict, Any, List
import ast

class Capture:
    name: str
    default: Any

    def __init__(self, name, default=None):
        self.name = name
        self.default = default

def check_optional(pattern_node, captures={}):
    """
    Check if the given pattern node is optional and populate the `captures` dict with the default values stored
    in the `Capture` nodes
    """
    if isinstance(pattern_node, Capture) and pattern_node.default != None:
        captures[pattern_node.name] = pattern_node.default
        return True, captures
    elif isinstance(pattern_node, ast.AST):
        is_optional = True
        for fieldname, child_node in ast.iter_fields(pattern_node):
            is_optional &= check_optional(getattr(child_node, fieldname), captures)
        return is_optional, captures
    return False, captures

def match(concrete_node, pattern_node, captures=None) -> Tuple[bool, Dict[str, ast.AST]]:
    if captures == None:
        captures = {}

    if isinstance(pattern_node, Capture):
        captures[pattern_node.name] = concrete_node
        return True
    elif type(concrete_node) != type(pattern_node):
        return False
    elif isinstance(pattern_node, ast.AST):
        # iterate over the fields of the concrete- and pattern-node side by side and check if they match
        for fieldname, pattern_val in ast.iter_fields(pattern_node):
            if not hasattr(concrete_node, fieldname):
                is_opt, opt_captures = check_optional(pattern_val)
                if is_opt:
                    # if the node is optional populate captures from the default values stored in the pattern node
                    captures.update(opt_captures)
                else:
                    return False
            else:
                if not match(getattr(concrete_node, fieldname), pattern_val, captures=captures):
                    return False
        return True
    elif isinstance(concrete_node, List):
        return all([match(ccn, cpn, captures=captures) for ccn, cpn in zip(concrete_node, pattern_node)])
    elif concrete_node == pattern_node:
        return True

    return False

# simple example
#concrete_node = ast.With(items=["dummy"], body=["123"])
#pattern_node = ast.With(items=Capture("items"), body=Capture("body"))

#concrete_node = ast.With(items=["dummy"])
#pattern_node = ast.With(items=Capture("items"), body=Capture("body", default=["abc"]))

#matches, captures = match(concrete_node, pattern_node)

#pass

# todo: pattern node ast.Name(bla=123) matches ast.Name(id="123") since bla is not an attribute
#  this can lead to errors which are hard to track
