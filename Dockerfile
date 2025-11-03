FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gfortran \
      libblas-dev \
      liblapack-dev \
      libhdf5-dev \
      libnetcdf-dev \
      pkg-config \
      wget \
      git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt


ENTRYPOINT ["/bin/bash", "-lc"]
