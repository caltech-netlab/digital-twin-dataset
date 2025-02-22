# Third-party imports
import sys
import pathlib
from werkzeug import run_simple
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from flask import Flask, render_template, request

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
from paths import FLASK_TEMPLATES_DIR
from api import api

base = Flask(__name__, template_folder=FLASK_TEMPLATES_DIR)
"""
Flask app to be mounted on the base path. This allows us to display a more user-friendly
homepage if someone visits the domain in a browser.
"""


@base.route("/")
def index():
    return render_template("index.html", api_url=f"{request.base_url}api")


app = DispatcherMiddleware(app=base, mounts={"/api": api})
"""WSGI app for the Dataset API."""

if __name__ == "__main__":
    # Run the app in debug mode.
    run_simple(
        hostname="localhost",
        port=5050,
        application=app,
        use_reloader=True,
        use_debugger=True,
    )
