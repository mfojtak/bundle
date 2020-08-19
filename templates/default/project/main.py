{% if functions is defined -%}
from fastapi import FastAPI, UploadFile, File
import uvicorn
import typing
from pydantic import BaseModel
app = FastAPI()
api = FastAPI(openapi_prefix="{{path}}")

{% for item in functions -%}
{% for key, value in item.items() -%}
from {{key}} import {{value}}
{% set mod = get_module(key, proj_dest) -%}
{% set sig = introspect(value, mod) -%}
{% if sig.params|length > 1 -%}
class {{value}}_Item(BaseModel):
    {% for param in sig.params -%}
    {{param.name}}{% if param.type_name is not none -%}: {{param.type_name}}{% endif -%}{% if param.default is not none -%} = {{param.default}}{% endif %}
    {% endfor %}
@api.post('/{{value}}'{% if sig.return_type_name is not none %}, response_model={{sig.return_type_name}}{% endif -%})
def {{value}}_wrapper(item: {{value}}_Item){% if sig.return_type_name is not none %} -> {{sig.return_type_name}}{% endif -%}:
    {% if sig.doc is not none -%}
    """{{ sig.doc }}"""
    {% endif -%}
    res = {{value}}({% for param in sig.params -%}item.{{param.name}},{% endfor %})
    return res
{% endif -%}
{% if sig.params|length == 1 -%}
{% set type_name = "UploadFile" if sig.params[0].type_name == "typing.BinaryIO" else sig.params[0].type_name -%}
{% set default = "File(...)" if sig.params[0].type_name == "typing.BinaryIO" else sig.params[0].default -%}
{% set call = sig.params[0].name + ".file" if sig.params[0].type_name == "typing.BinaryIO" else sig.params[0].name -%}

@api.post('/{{value}}'{% if sig.return_type_name is not none %}, response_model={{sig.return_type_name}}{% endif -%})
def {{value}}_wrapper({{sig.params[0].name}}{% if type_name is not none -%}: {{type_name}}{% endif -%}{% if default is not none -%} = {{default}}{% endif %}):
    {% if sig.doc is not none -%}
    """{{ sig.doc }}"""
    {% endif -%}
    res = {{value}}({{call}})
    return res
{% endif -%}
{% if sig.params|length == 0 -%}
@api.get('/{{value}}'{% if sig.return_type_name is not none %}, response_model={{sig.return_type_name}}{% endif -%})
def {{value}}_wrapper():
    {% if sig.doc is not none -%}
    """{{ sig.doc }}"""
    {% endif -%}
    res = {{value}}()
    return res
{% endif -%}
{% endfor %}
{% endfor -%}

app.mount("{{path}}", api)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port={{port|default(80)}})
{% endif -%}