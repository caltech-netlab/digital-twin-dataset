# Third-party imports
import sys
import pathlib
import json
from http import HTTPStatus
from flask import abort

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
from paths import REPLACEMENT_LOOKUP_FILE

replacement_lookup: dict[str, str]
"""Dictionary from real to anonymous element names."""

reverse_replacement_lookup: dict[str, str]
"""Dictionary from anonymous to real element names."""

with open(REPLACEMENT_LOOKUP_FILE) as f:
    replacement_lookup = json.load(f)
    reverse_replacement_lookup = {v: k for k, v in replacement_lookup.items()}


def anonymize_elements(real_elements: list[str]) -> list[str]:
    """
    Anonymize the given real elements.

    :param real_elements: List of real element names.
    :returns: List of anonymized element names.
    """
    return [replacement_lookup[real_element] for real_element in real_elements]


def deanonymize_elements(anon_elements: list[str] | None) -> list[str]:
    """
    Deanonymize the given anonymized elements. If any are not in the lookup dictionary,
    raise an ``HTTPException``.

    :param anon_elements: List of anonymized element names.
    :returns: List of real element names.
    """
    if anon_elements is None:
        return []
    try:
        return [
            reverse_replacement_lookup[anon_element] for anon_element in anon_elements
        ]
    except KeyError as e:
        abort(HTTPStatus.BAD_REQUEST, description=f"element {e} does not exist")
