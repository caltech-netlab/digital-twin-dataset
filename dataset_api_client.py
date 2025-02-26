# Third-party imports
from typing import Self, Any, overload
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import time
import json
import webbrowser
from datetime import datetime, timedelta
import pyperclip
from tqdm import tqdm
import requests
from stream_unzip import stream_unzip

# Assume verification codes or tokens are expired if they are within this many seconds
# of expiring in order to avoid using invalid tokens.
_EXPIRATION_TOLERANCE = 60


def raise_response_error(response: requests.Response) -> None:
    """
    Raise a descriptive ``RuntimeError`` if the given response does not have a
    successful HTTP status code.

    :param response: the response to check.
    """
    if response.ok:
        return
    response_json = response.json()
    code = response_json["code"]
    name = response_json["name"]
    description = response_json["description"]
    # Format Pydantic validation errors nicely.
    if isinstance(description, list):
        description = "\n" + "\n".join(f"  {row!r}" for row in description)
    raise RuntimeError(f"{code} {name}: {description}")


@overload
def github_login_request(
    url_path: str, args: dict[str, Any], expected_error: str
) -> dict[str, Any] | None: ...
@overload
def github_login_request(url_path: str, args: dict[str, Any]) -> dict[str, Any]: ...
def github_login_request(
    url_path: str, args: dict[str, Any], expected_error: str | None = None
) -> dict[str, Any] | None:
    """
    Make a GitHub login request.

    :param url_path: A ``POST`` request will be made to
        ``https://github.com/login/{url_path}``.
    :param args: JSON data to include in the body of the ``POST`` request.
    :param expected_error: If given, return ``None`` if the request returns that error.
        For all other errors returned, raise a ``RuntimeError``.
    :returns: The JSON response,
    """
    issued_at = time.time()
    response = requests.post(
        f"https://github.com/login/{url_path}",
        json=args,
        headers={"Accept": "application/json"},
    )
    response_json: dict[str, Any] = response.json()
    error = response_json.get("error")
    if expected_error is not None and error == expected_error:
        return None
    if error is not None:
        error_message = error
        error_description = response_json.get("error_description")
        if error_description is not None:
            error_message += f" ({error_description})"
        raise RuntimeError(error_message)
    return response_json | {"issued_at": issued_at}


@dataclass
class GitHubVerification:
    """
    This class makes the request and stores the result from
    https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#step-1-app-requests-the-device-and-user-verification-codes-from-github.
    """

    device_code: str
    """40-character code used to retrieve the authentication status and tokens."""

    user_code: str
    """
    8-character code with a hyphen in the middle, which the user must enter in a browser
    at ``verification_uri``.
    """

    verification_uri: str
    """Verification URL where users need to enter ``user_code``."""

    expires_in: int
    """Number of seconds before ``device_code`` and ``user_code`` expire."""

    interval: int
    """
    Minimum number of seconds to wait between polling for tokens to avoid the rate
    limit.
    """

    issued_at: float
    """UTC timestamp of when the initial request was sent."""

    @property
    def expired(self) -> bool:
        """Whether ``device_code`` and ``user_code`` have expired."""
        return time.time() >= self.issued_at + self.expires_in - _EXPIRATION_TOLERANCE

    @classmethod
    def request(cls, client_id: str) -> Self:
        """
        Initiate device flow verification.

        See
        https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#step-1-app-requests-the-device-and-user-verification-codes-from-github
        for more information.

        :returns: A ``GitHubVerification`` object.
        """
        response_data = github_login_request(
            "device/code", args={"client_id": client_id}
        )
        return cls(**response_data)


@dataclass
class GitHubTokens:
    """
    This class makes the request and stores the result from
    https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#step-3-app-polls-github-to-check-if-the-user-authorized-the-device.
    """

    access_token: str
    """Token used to authorize requests on behalf of the user."""

    expires_in: int
    """Number of seconds before ``access_token`` expires."""

    refresh_token: str
    """Token used to request new tokens."""

    refresh_token_expires_in: int
    """Number of seconds before ``refresh_token`` expires."""

    token_type: str
    """Type of ``access_token``."""

    scope: str
    """
    Scope of ``access_token``. See
    https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps#available-scopes
    for more information.
    """

    issued_at: float
    """UTC timestamp of when the initial request was sent."""

    @property
    def expired(self) -> bool:
        """Whether ``access_token`` has expired."""
        return time.time() >= self.issued_at + self.expires_in - _EXPIRATION_TOLERANCE

    @property
    def refresh_token_expired(self) -> bool:
        """Whether ``refresh_token`` has expired."""
        return (
            time.time()
            >= self.issued_at + self.refresh_token_expires_in - _EXPIRATION_TOLERANCE
        )

    def save(self, credentials_file: str) -> None:
        """
        Save tokens to the credentials file.

        :param credentials_file: Path to the JSON file to save credentials to.
        """
        with open(credentials_file, "w") as f:
            print(json.dumps(asdict(self), indent=2), file=f)

    @classmethod
    def load(cls, credentials_file: str) -> Self:
        """
        Load tokens from the credentials file.

        :param credentials_file: Path to the JSON file to load credentials from.
        """
        with open(credentials_file) as f:
            return cls(**json.load(f))

    @classmethod
    def request(
        cls, client_id: str, device_code: str, credentials_file: str
    ) -> Self | None:
        """
        Request access tokens after initiating device flow verification.

        See
        https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#step-3-app-polls-github-to-check-if-the-user-authorized-the-device
        for more information.

        :param client_id: The client ID of a GitHub App to authenticate with.
        :param device_code: The device code returned by ``GitHubVerification.request()``.
        :param credentials_file: Path to the JSON file to save credentials to.
        :returns: A ``GitHubTokens`` object if successful, or ``None`` if authorization is
            pending.
        """
        response_data = github_login_request(
            "oauth/access_token",
            args={
                "client_id": client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            expected_error="authorization_pending",
        )
        if response_data is None:
            return None
        tokens = cls(**response_data)
        tokens.save(credentials_file)
        return tokens

    @classmethod
    def poll(
        cls, client_id: str, verification: GitHubVerification, credentials_file: str
    ) -> Self:
        """
        Poll GitHub to request access tokens until the user has entered ``user_code``.

        See
        https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#step-3-app-polls-github-to-check-if-the-user-authorized-the-device)
        for more information.

        :param client_id: The client ID of the GitHub App being used to authenticate.
        :param verification: A ``GitHubVerification`` object.
        :param credentials_file: Path to the JSON file to save credentials to.
        """
        while not verification.expired:
            tokens = cls.request(
                client_id=client_id,
                device_code=verification.device_code,
                credentials_file=credentials_file,
            )
            if tokens is not None:
                return tokens
            time.sleep(verification.interval)
        raise RuntimeError("code expired")

    def refresh(self, client_id: str, credentials_file: str) -> Self:
        """
        Request new tokens using the ``refresh_token``.

        See
        https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/refreshing-user-access-tokens#refreshing-a-user-access-token-with-a-refresh-token
        for more information.

        :param client_id:
        """
        response_data = github_login_request(
            "oauth/access_token",
            args={
                "client_id": client_id,
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            },
        )
        tokens = type(self)(**response_data)
        tokens.save(credentials_file)
        return tokens


@dataclass
class DatasetApiClient:
    base_url: str = "https://socal28bus.caltech.edu/api"
    """Base URL for the dataset API."""

    github_client_id: str = "Iv23liatKIFoXYtBW2eU"
    """
    Client ID for the GitHub app to sign in to. The default is for
    https://github.com/apps/digital-twin-dataset.
    """

    credentials_file: str = "dataset_api_credentials.json"
    """Default path to a file to store credentials in and retrieve them from."""

    chunk_size: int = 65536
    """
    Chunk size for streaming responses, chosen based on
    https://stream-unzip.docs.trade.gov.uk/get-started/#usage.
    """

    @property
    def tokens(self) -> GitHubTokens:
        """
        Return tokens from ``credentials_file`` if it exists. Otherwise, request new
        tokens via the GitHub https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#device-flow.
        """
        if os.path.exists(self.credentials_file):
            tokens = GitHubTokens.load(self.credentials_file)
            if tokens.expired:
                if tokens.refresh_token_expired:
                    print(
                        f"Refresh token in '{self.credentials_file}' is expired,"
                        " requesting new tokens.\n"
                    )
                    return self.request_new_tokens()
                return tokens.refresh(
                    client_id=self.github_client_id,
                    credentials_file=self.credentials_file,
                )
            return tokens
        print(
            f"Credentials file '{self.credentials_file}' does not exist, requesting"
            " new tokens.\n"
        )
        return self.request_new_tokens()

    @property
    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.tokens.access_token}"}

    def request_new_tokens(self) -> GitHubTokens:
        """
        Request new tokens via the GitHub
        https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#device-flow.
        """
        verification = GitHubVerification.request(client_id=self.github_client_id)
        try:
            # If possible, automatically copy user_code to the clipboard.
            pyperclip.copy(verification.user_code)
            copied_message = " (automatically copied to clipboard)"
        except pyperclip.PyperclipException:
            copied_message = ""
        print(f"Verification code: {verification.user_code}{copied_message}")
        print(f"To continue, enter this code at {verification.verification_uri}\n")
        webbrowser.open_new_tab(verification.verification_uri)
        tokens = GitHubTokens.poll(
            client_id=self.github_client_id,
            verification=verification,
            credentials_file=self.credentials_file,
        )
        print(
            f"Success! Credentials saved in {os.path.abspath(self.credentials_file)}."
        )
        print(
            "WARNING: Keep this file a secret (e.g. do not share this file or commit it"
            " to a git repository)."
        )
        return tokens

    def get_user(self) -> str:
        """Request the currently authenticated user."""
        response = requests.get(f"{self.base_url}/user", headers=self.auth_header)
        raise_response_error(response)
        return response.json()

    def download_data(
        self,
        *,
        magnitudes_for: list[str] | None = None,
        phasors_for: list[str] | None = None,
        waveforms_for: list[str] | None = None,
        time_range: tuple[datetime | str, datetime | str],
        resolution: timedelta | float | str | None = None,
    ) -> None:
        """
        Download data files. Any data that does not exist for the given elements within
        the given time range will be silently omitted.

        :param magnitudes_for: Optional, metwork elements to download magnitude data
            for.
        :param phasors_for: Optional, network elements to download phasor data for.
        :param waveforms_for: Optional, network elements to download waveform data for.
        :param time_range: Time range to retrieve data for.
        :param resolution: Optional, interval to sample data by.
        """
        with requests.post(
            f"{self.base_url}/data",
            headers=self.auth_header,
            json={
                "magnitudes_for": magnitudes_for or [],
                "phasors_for": phasors_for or [],
                "waveforms_for": waveforms_for or [],
                "time_range": [
                    d.isoformat() if isinstance(d, datetime) else d for d in time_range
                ],
                "resolution": resolution,
            },
            stream=True,
        ) as response:
            raise_response_error(response)
            zipped_chunks = response.iter_content(chunk_size=self.chunk_size)
            progress_in_bytes = tqdm(
                unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading"
            )
            num_files = 0
            progress_in_bytes.set_postfix_str(f"{num_files:,} files", refresh=False)
            first = True
            for file_name, _, unzipped_chunks in stream_unzip(zipped_chunks):
                file_path = Path(file_name.decode())
                if first:
                    root_dir = datetime.now().strftime("data_%Y-%m-%d_%H-%M-%S")
                    unique_root_dir = root_dir
                    suffix_num = 1
                    while os.path.exists(unique_root_dir):
                        unique_root_dir = f"{root_dir}_{suffix_num}"
                        suffix_num += 1
                    first = False
                unique_file_path = os.path.join(unique_root_dir, *file_path.parts[1:])
                os.makedirs(os.path.dirname(unique_file_path), exist_ok=True)
                with open(unique_file_path, "wb") as f:
                    for chunk in unzipped_chunks:
                        bytes_written = f.write(chunk)
                        progress_in_bytes.update(bytes_written)
                num_files += 1
                progress_in_bytes.set_postfix_str(f"{num_files:,} files", refresh=False)
            progress_in_bytes.close()
