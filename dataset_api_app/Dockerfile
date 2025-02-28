FROM python:3.13-slim

# Change into a new directory /digital-twin-dataset
WORKDIR /digital-twin-dataset

# Copy requirements and app code
COPY utils/ utils/
COPY requirements.txt requirements.txt
COPY dataset_api_app/src/ dataset_api_app/src/
COPY dataset_api_app/templates/ dataset_api_app/templates/
COPY dataset_api_app/requirements.txt dataset_api_app/requirements.txt
COPY dataset_api_app/gunicorn.conf.py dataset_api_app/gunicorn.conf.py

# Install Python requirements
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r dataset_api_app/requirements.txt

# Change to the dataset_api_app directory
WORKDIR /digital-twin-dataset/dataset_api_app

# Run gunicorn (using dataset_api_app/gunicorn.conf.py)
ENTRYPOINT [ "gunicorn" ]
