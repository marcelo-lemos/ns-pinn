FROM python:3.10-bookworm

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

COPY . /app/

ENTRYPOINT ["python", "ns_pinn/train.py"]
