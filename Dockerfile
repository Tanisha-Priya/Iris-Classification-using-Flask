FROM python:3.6-slim
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /opt/
WORKDIR /opt
# Expose ports
EXPOSE 8080 8000
ENTRYPOINT ["sh", "-c", "python app.py & prometheus_client.start_http_server(8000) && flask run --host=0.0.0.0 --port=8080"]
