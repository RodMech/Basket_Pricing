FROM python:3.8-slim

RUN mkdir /opt/project

COPY requirements.txt requirements.txt

RUN python3.8 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/project

CMD ["jupyter-lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
