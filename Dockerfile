FROM python:3.11-slim

WORKDIR /app

RUN mkdir -p /data/INPUT/BATHYMETRY /data/INPUT/CLIMATOLOGY

COPY blue-cloud-dataspace/MEI/INGV/INPUT/BATHYMETRY/gebco_2019_mask_1_8_edited_final.nc /data/INPUT/BATHYMETRY/
COPY blue-cloud-dataspace/MEI/INGV/INPUT/CLIMATOLOGY/Temperature_sliding_climatology_WP.nc /data/INPUT/CLIMATOLOGY/

COPY ohc_runner.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python","ohc_runner.py"]
