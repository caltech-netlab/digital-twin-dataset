# Third-party imports
import requests


def github_api_request(
    url_path: str, authorization_header: str | None = None
) -> requests.Response:
    """
    Make a request via the GitHub API.

    :param url_path: A ``GET`` request will be made to
        ``https://api.github.com/{url_path}``.
    :param authorization_header: Optional, contents of an authorization header to send
        with the request.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if authorization_header:
        headers |= {"Authorization": authorization_header}
    return requests.get(f"https://api.github.com/{url_path}", headers=headers)


def request_github_authenticated_user(authorization_header: str) -> requests.Response:
    """
    Get the currently authenticated GitHub user.

    See
    https://docs.github.com/en/rest/users/users?apiVersion=2022-11-28#get-the-authenticated-user
    for more information.

    :param authorization_header: Contents of the authorization header to send with the
        request.
    :returns: A ``Response`` object.
    """
    return github_api_request("user", authorization_header)


def request_github_user(github_username: str) -> requests.Response:
    """
    Get public information for the given GitHub user.

    See https://docs.github.com/en/rest/users/users?apiVersion=2022-11-28#get-a-user
    for more information.

    :param github_username: Username of a GitHub user.
    :returns: A ``Response`` object.
    """
    return github_api_request(f"users/{github_username}")
