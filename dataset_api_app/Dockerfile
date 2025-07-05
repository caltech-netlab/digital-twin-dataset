FROM python:3.13

# Get URL from https://www.sqlite.org/download.html
ARG SQLITE_AUTOCONF_URL=https://www.sqlite.org/2025/sqlite-autoconf-3500200.tar.gz

# Install SQLite from source. (Version 3.42.0 or later is required for the `subsec`
# option used in rate limiting.)
ENV LD_LIBRARY_PATH=/usr/local/lib
RUN mkdir /tmp/sqlite && cd /tmp/sqlite \
    && curl ${SQLITE_AUTOCONF_URL} | tar xz --strip-components=1 \
    && ./configure \
    && make && make install \
    && cd / && rm -rf /tmp/sqlite

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
