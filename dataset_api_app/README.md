# Dataset API App

This directory defines the API app that runs on the server.

## Setup

1. Clone this repository to the server.

2. The server must be run on a machine with a static IP address and publically
   accessible ports 80 (for HTTP) and 443 (for HTTPS).

3. Obtain one or more domain names pointing to the server's IP address, and update the
   two `server_name` directives in [nginx.prod.conf](nginx.prod.conf) (one for port 80
   and the other for port 443).

4. Obtain a TLS/SSL certificate and place the certificate and key files in this
   directory. Update the `ssl_certificate` and `ssl_certificate_key` directives in
   [nginx.prod.conf](nginx.prod.conf) accordingly.

5. Create a file in this directory called `replacement_lookup.json`, containing a
   dictionary from real network element names to anonymized ones. This will be used by
   the server to return data with anonymized element names.

6. Create a file in this directory called `.env` containing the following:

```env
APP_ENV=prod
MAGNITUDES_DIR=<path to magnitude data directory>
PHASORS_DIR=<path to phasor data directory>
WAVEFORMS_DIR=<path to waveform data directory>
WAVEFORMS_2024_10_DIR=<path to pre-11/01/2024 waveform data directory>
```

6. Docker is required to run the production server. Make sure that Docker is installed
   and that the Docker daemon is running by running `docker ps`.

7. Change into this directory (e.g. `cd dataset_api_app`).

8. Run `docker compose build`, and then `docker compose up`. This will start the
   production server.

9. Visit your domain in the browser. You should see a page called "Digital Twin Dataset
   API".

## Adding and Removing Users

To add a user to the database, run the following from within this directory on the
server, using the desired user's actual GitHub username. Note that Python must be
installed to run this.

```
python -c "from src.users import User; User.add('<github username>')"
```

To remove a user, run the following:

```
python -c "from src.users import User; User.remove('<github ID or username>')"
```

To confirm which users are authorized, run the following:

```
python -c "from src.users import User; User.print_all()"
```
