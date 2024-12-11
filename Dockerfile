FROM python:3.6-slim
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /opt/
WORKDIR /opt
EXPOSE 8000
EXPOSE 8080
CMD  ["python","app.py"]
