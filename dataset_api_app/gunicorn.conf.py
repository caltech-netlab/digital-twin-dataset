import multiprocessing

# Use the Flask "app" object in src/app.py as the WSGI app
wsgi_app = "src.app:app"

# Bind to port 8000
bind = "0.0.0.0:8000"

# See https://docs.gunicorn.org/en/latest/design.html#how-many-workers
workers = multiprocessing.cpu_count() * 2 + 1

# Disable timeout to allow for long running zip streaming (default is 30 seconds)
timeout = 0
