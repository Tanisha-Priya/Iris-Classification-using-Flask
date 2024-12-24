FROM python:3.6
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --proxy http://proxy365.sacombank.com:1985 -r requirements.txt
COPY . /opt/
WORKDIR /opt
EXPOSE 8000
#EXPOSE 8000
CMD  ["python","model_metrics.py"]
