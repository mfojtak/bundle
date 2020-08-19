from inspect import signature, Parameter
import inspect
from typing import List
import importlib.machinery
import importlib.util

class Param:
    name = None
    default = None
    annotation = None
    type_name = None

class Signature:
    params: List[Param]
    return_annotation = None
    return_type_name = None
    doc = None
    def __init__(self):
        self.params = []

def get_module(name, path):
    spec = importlib.machinery.PathFinder().find_spec(name, [path])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def introspect(f,mod):
    f = getattr(mod, f)
    sig = signature(f)
    res = Signature()
    for param in sig.parameters.values():
        res_param = Param()
        res_param.name = param.name
        if param.default is not Parameter.empty:
            res_param.default = param.default
        if param.annotation is not Parameter.empty:
            res_param.annotation = param.annotation
            if hasattr(param.annotation, '__name__'):
                res_param.type_name = ""
                if param.annotation.__module__ is not "builtins":
                    res_param.type_name = param.annotation.__module__ + "."
                res_param.type_name = res_param.type_name + param.annotation.__name__
            else:
                res_param.type_name = str(param.annotation)

        res.params.append(res_param)
    if sig.return_annotation is not inspect.Signature.empty:
        res.return_annotation = sig.return_annotation
        res.return_type_name = str(sig.return_annotation)
    res.doc = f.__doc__
    return res
        
