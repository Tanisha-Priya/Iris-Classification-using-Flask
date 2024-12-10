FROM python:3.6-slim
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /opt/
WORKDIR /opt
EXPOSE 8080
EXPOSE 8090  # Expose the Prometheus metrics port
ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0 --port=8080
