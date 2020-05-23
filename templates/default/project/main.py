{% if functions is defined -%}
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
app = FastAPI()
api = FastAPI(openapi_prefix="{{path}}")

{% for item in functions -%}
{% for key, value in item.items() -%}
from {{key}} import {{value}}
{% set mod = get_module(key, proj_dest) -%}
{% set sig = introspect(value, mod) -%}
{% if sig.params -%}
class {{value}}_Item(BaseModel):
    {% for param in sig.params -%}
    {{param.name}}{% if param.type_name is not none -%}: {{param.type_name}}{% endif -%}{% if param.default is not none -%} = {{param.default}}{% endif %}
    {% endfor %}
{% endif -%}
@api.{{"post" if sig.params else "get"}}('/{{value}}')
def {{value}}_wrapper({% if sig.params -%}item: {{value}}_Item{% endif -%}):
    res = {{value}}({% for param in sig.params -%}item.{{param.name}},{% endfor %})
    return res
{% endfor %}
{% endfor -%}

app.mount("{{path}}", api)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port={{port|default(80)}})
{% endif -%}