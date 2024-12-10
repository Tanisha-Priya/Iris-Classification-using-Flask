FROM python:3.6-slim
COPY requirements.txt requirements.txt
RUN pip install --proxy http://proxy365.sacombank.com:1985 -r -requirements.txt
COPY . /opt/
WORKDIR /opt
EXPOSE 8080 8000
ENV FLASK_APP=app.py
CMD ['python', './app.py','--host=0.0.0.0']
