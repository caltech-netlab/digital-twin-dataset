---
name: dataset_api_app:restart-server
description: Restart the Dataset API production server after a host reboot - remounts the NAS data drives and brings the Docker containers back up. Use when the server host has rebooted, the NAS mounts are missing, or `docker ps` shows the containers are down.
---

# Restart Dataset API server after reboot

Run from anywhere (mount commands use `sudo`, no need to `cd` first).

## 1. Check and remount NAS shares

Check whether each mount is already present before mounting (mounting an
already-mounted path is a no-op but `mount` will error if the mountpoint dir
doesn't exist yet, so always `mkdir -p` first):

```bash
mountpoint -q /mnt/dt_nas_2024_10 || \
  (sudo mkdir -p /mnt/dt_nas_2024_10 && sudo mount -t nfs 192.168.11.100:/caltech_twin_data /mnt/dt_nas_2024_10)

mountpoint -q /mnt/dt_nas_2025_03 || \
  (sudo mkdir -p /mnt/dt_nas_2025_03 && sudo mount -t nfs 192.168.11.100:/caltech_twin_data3 /mnt/dt_nas_2025_03)

mountpoint -q /mnt/dt_nas || \
  (sudo mkdir -p /mnt/dt_nas && sudo mount -t nfs 192.168.11.100:/caltech_twin_data4 /mnt/dt_nas)
```

These three mounts correspond to the `WAVEFORMS_2024_10_DIR`, `WAVEFORMS_2025_03_DIR`,
and `MAGNITUDES_DIR`/`PHASORS_DIR`/`WAVEFORMS_DIR` paths configured in `.env` (see
`dataset_api_app/README.md`).

## 2. Make sure nothing else is holding port 80/443

This host also has the stock Ubuntu `nginx` package installed, and its systemd
service (`nginx.service`, serving the generic default page) can auto-start on
boot and grab ports 80/443 before the Docker nginx container does, causing
`docker compose up -d` to fail with `address already in use`. Confirm it's
still disabled (it was intentionally stopped/disabled once already):

```bash
systemctl is-active nginx    # should print "inactive"
systemctl is-enabled nginx   # should print "disabled"
```

If it shows `active`/`enabled`, stop and disable it: `sudo systemctl stop nginx && sudo systemctl disable nginx`.

## 3. Start Docker and the server

```bash
docker ps                    # confirm the Docker daemon is running
cd dataset_api_app
docker compose build
docker compose up -d
```

Other useful commands:

- Restart: `docker compose restart`
- Stop: `docker compose stop`
- Check running containers: `docker ps`
- View logs: `docker logs gunicorn` / `docker logs nginx`

Note: `requirements.txt` must include `pytz` (used by `utils/utils.py`) — it was
missing at one point and only worked because an old cached image happened to
have it as a transitive dependency; a fresh `docker compose build` will
crash-loop gunicorn with `ModuleNotFoundError: No module named 'pytz'` if it's
ever removed again.

## 4. Verify

```bash
curl -sk -o /dev/null -w "HTTP %{http_code}\n" http://localhost/
curl -sk -o /dev/null -w "HTTPS %{http_code}\n" https://localhost/
```

Expect `HTTP 301` (redirect to HTTPS) and `HTTPS 200`. Also visit the
configured domain in a browser — you should see a page titled "Digital Twin
Dataset API".
