{% if functions is defined -%}
from flask import Flask
from flask import request
from flask import jsonify
app = Flask(__name__)

{% for item in functions -%}
{% for key, value in item.items() -%}
from {{key}} import {{value}}
@app.route('/{{value}}')
def {{value}}_wrapper():
    data = request.json
    if data:
        res = {{value}}(**data)
    else:
        res = {{value}}()
    return jsonify(res)
{% endfor -%}
{% endfor %}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port={{port|default(80)}})
{% endif -%}