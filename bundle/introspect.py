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
    params: List[Param] = list()
    return_annotation = None
    return_type_name = None
    doc = None

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
            res_param.type_name = param.annotation.__name__

        res.params.append(res_param)
    if sig.return_annotation is not inspect.Signature.empty:
        res.return_annotation = sig.return_annotation
        res.return_type_name = sig.return_annotation.__name__
    res.doc = f.__doc__
    return res
        
