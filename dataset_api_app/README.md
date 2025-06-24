# Dataset API App

This directory defines the API app that runs on the server.

## Initial Setup

1. Clone this repository to the server.

2. The server must be run on a machine with a static IP address and publically
   accessible ports 80 (for HTTP) and 443 (for HTTPS).

3. Obtain one or more domain names pointing to the server's IP address, and update the
   two `server_name` directives in [nginx.prod.conf](nginx.prod.conf) (one for port 80
   and the other for port 443).

4. Obtain a TLS/SSL certificate and place the certificate and key files a subdirectory
   of this directory called `certs`. Make sure the file names match the `ssl_certificate`
   and `ssl_certificate_key` directives in [nginx.prod.conf](nginx.prod.conf), or update
   the directives accordingly.

5. If the private key file requires a password (set when creating the initial CSR), add
   that password to a file. Make sure the file name matches the `ssl_password_file`
   directive in [nginx.prod.conf](nginx.prod.conf), or update the directives accordingly.

6. Create a file in this directory called `replacement_lookup.json`, containing a
   dictionary from real network element names to anonymized ones. This will be used by
   the server to return data with anonymized element names.

7. Create a file in this directory called `.env` containing the following:

   ```env
   APP_ENV=prod
   MAGNITUDES_DIR=<path to magnitude data directory>
   PHASORS_DIR=<path to phasor data directory>
   WAVEFORMS_DIR=<path to waveform data directory>
   WAVEFORMS_2024_10_DIR=<path to pre-11/01/2024 waveform data directory>
   ```

8. Change into this directory (e.g. `cd dataset_api_app`).

9. Run the following commands to create a Python virtual environment and use it to
   create the initial users.db file:

   ```
   python -m venv venv
   venv/bin/pip install -r requirements.txt
   venv/bin/python src/users.py
   ```

10. Docker is required to run the production server. Make sure that Docker is installed
    and that the Docker daemon is running by running `docker ps`.

11. Run `docker compose build` to build the Docker image, and then `docker compose up -d`
    to start the production server in detached mode. (To restart, run
    `docker compose restart`, and to stop, run `docker compose stop`. Run
    `docker compose --help` to see all options.)

12. Visit your domain in the browser. You should see a page called "Digital Twin Dataset
    API".

## Adding and Removing Users

To add a user to the database, run the following from within this directory on the
server, using the desired user's actual GitHub username. This assumes that the Python
virtual environment was created in [Setup](#setup).

```
venv/bin/python -c "from src.users import User; User.add('<github username>')"
```

To remove a user, run the following:

```
venv/bin/python -c "from src.users import User; User.remove('<github ID or username>')"
```

To confirm which users are authorized, run the following:

```
venv/bin/python -c "from src.users import User; User.print_all()"
```

## Server Maintenance, Updates

If the code is updated, change in to this directory (e.g. `cd dataset_api_app`) and run
`docker compose build`, followed by `docker compose up -d`.

You can also run `docker system prune -a` to remove all containers and images that are
not currently in use.

To see what images are running, run `docker ps`.

Run `docker logs gunicorn` to view the Gunicorn server logs and run `docker logs nginx`
to view the nginx logs.

### API Usage Logs

Each time a request is made, a line will be added to the file
`dataset_api_app/logs/api_usage.log` logging information including username, the data
being request, and any error that occurred. These logs can help spot potential abuse,
high-demand users, or bugs.

Logs are rotated monthly (although the rotation will only happen when the first log
comes in during a new month). Months with no requests will not have a log file.

There are also some helper functions defined in
[`dataset_api_app/src/log_helpers.py`](src/log_helpers.py) to help with analyzing the
logs. See file and corresponding docstrings for more details.

## Running in Development Mode

To run the server in development mode, follow these steps, similar to those in
[Setup](#setup):

1. Clone this repository to the server.

2. Create a file in this directory called `replacement_lookup.json`, containing a
   dictionary from real network element names to anonymized ones. This will be used by
   the server to return data with anonymized element names.

3. Create a file in this directory called `.env` containing the following (same as in
   [Setup](#setup), but with `APP_ENV` set to `dev` and local paths):

   ```env
   APP_ENV=dev
   MAGNITUDES_DIR=<path to magnitude data directory>
   PHASORS_DIR=<path to phasor data directory>
   WAVEFORMS_DIR=<path to waveform data directory>
   WAVEFORMS_2024_10_DIR=<path to pre-11/01/2024 waveform data directory>
   ```

4. Change into this directory (e.g. `cd dataset_api_app`).

5. Run the following commands to create a Python virtual environment and use it to
   create the initial users.db file:

   ```
   python -m venv venv
   venv/bin/pip install -r requirements.txt
   venv/bin/python src/users.py
   ```

Then, to run the app in development mode, run:

```
venv/bin/python src/app.py
```

The development server has the address http://localhost:5050/api and will be updated
when the code changes. You can pass this address to `DatasetApiClient`:

```python
data_api_client = DatasetApiClient(base_url="http://localhost:5050/api")
```

To test the Docker production server in development, follow the same instructions from
the end of [Setup](#setup) (`docker compose build` and `docker compose up -d`). The
API will then have the address http://localhost/api.
