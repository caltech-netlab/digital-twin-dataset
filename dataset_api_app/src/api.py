# Third-party imports
import sys
import time
import pathlib
from functools import wraps
from dataclasses import asdict
from datetime import datetime
from pydantic import ValidationError
from http import HTTPStatus
from werkzeug.exceptions import HTTPException
from flask import Flask, abort, stream_with_context, request, g
from stream_zip import stream_zip

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
from logs import api_usage_logger
from github import request_github_authenticated_user
from users import User
from data import DataRequest, generate_files

api = Flask(__name__)
"""
Flask app to be mounted on the `/api` path. This allows us to have a more friendly
homepage and reserve other subpaths for future use (such as documentation).
"""

# Stop Flask from sorting JSON keys alphabetically.
api.json.sort_keys = False

# Output JSON in a non-compact format (i.e. with newlines and indentation).
api.json.compact = False


def get_authenticated_user() -> User:
    """
    Request the currently authenticated user based on the authorization header.

    If a valid GitHub access token is not present or if the given GitHub user is not in
    the users database, raise a 401 Unauthorized HTTP error.
    """
    authorization_header = request.headers.get("Authorization")
    if authorization_header:
        response = request_github_authenticated_user(authorization_header)
        if response.ok:
            user_info = response.json()
            github_id = user_info["id"]
            github_username = user_info["login"]
            g.github_id = github_id
            g.github_username = github_username
            user = User.get(github_id)
            if user is not None:
                return user
    abort(HTTPStatus.UNAUTHORIZED)


def protected(route):
    """Decorator that marks a protected route."""

    @wraps(route)
    def wrapper(*args, **kwargs):
        get_authenticated_user()
        return route(*args, **kwargs)

    return wrapper


@api.errorhandler(HTTPException)
def handle_exception(exception: HTTPException):
    """Return HTTP exceptions as JSON."""
    api_usage_logger.error("")
    return (
        {
            "code": exception.code,
            "name": exception.name,
            "description": exception.description,
        },
        exception.code,
    )


@api.get("/user")
def user():
    """Return the currently authenticated user."""
    return asdict(get_authenticated_user())


@api.post("/data")
@protected
def data():
    """
    Stream a ZIP file containing the requested data. If the request body cannot be
    parsed as a ``DataRequest``, an ``HTTPException`` will be raised.
    """
    g.request_started = time.perf_counter()
    if not request.is_json:
        raise request.on_json_loading_failed(e=None)
    try:
        data_request = DataRequest.model_validate_json(request.data)
        g.data_request = data_request
    except ValidationError as validation_error:
        abort(
            HTTPStatus.BAD_REQUEST,
            description=validation_error.errors(
                include_url=False, include_context=False, include_input=False
            ),
        )
    zip_root_dir = datetime.now().strftime("data_%Y-%m-%d_%H-%M-%S")
    files = generate_files(zip_root_dir, data_request)

    def stream_zip_and_log():
        try:
            yield from stream_zip(files)
        except BaseException as exc:
            message = str(exc)
            if not message and isinstance(exc, GeneratorExit):
                message = "connection was interrupted"
            api_usage_logger.error("")
            raise exc
        else:
            api_usage_logger.info("")

    return stream_with_context(stream_zip_and_log()), {
        "Content-Type": "application/zip",
        "Content-Disposition": f'filename="{zip_root_dir}.zip"',
    }
