FROM mfojtak/python-docker:latest
ADD requirements.txt requirements.txt
ADD project project
RUN pip install -r requirements.txt && pip install python-multipart
{% for item in extra_req_files -%}
RUN pip install -r {{ item }}
{% endfor -%}
{% if apt_packages is defined -%}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install -y {{ apt_packages }}
{% endif -%}