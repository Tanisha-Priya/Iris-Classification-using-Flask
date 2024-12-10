FROM python:3.6-slim
COPY requirements.txt requirements.txt
RUN pip install --proxy http://proxy365.sacombank.com:1985 -rrequirements.txt
COPY . /opt/
WORKDIR /opt
EXPOSE 8080 8000
ENTRYPOINT ["sh", "-c", "python app.py & prometheus_client.start_http_server(8000) && flask run --host=0.0.0.0 --port=8080"]
