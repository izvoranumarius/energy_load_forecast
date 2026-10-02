"""Configuration helpers for loading the ENTSO-E API key.

The project reads the API key from the ``ENTSOE_API_KEY`` environment
variable, optionally populated from a local ``.env`` file.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_entsoe_api_key() -> str:
    """Read the ENTSO-E API key from the environment.
    
        The project loads environment variables from a local ``.env`` file before
        reading ``ENTSOE_API_KEY``. The key is intentionally not stored in source
        code.
    
        Returns:
            The ENTSO-E API key stored in ``ENTSOE_API_KEY``.
    
        Raises:
            RuntimeError: If ``ENTSOE_API_KEY`` is not set.
    """
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError(
            "ENTSOE_API_KEY is not set. Copy .env.example to .env and fill "
            "in your key (register at https://transparency.entsoe.eu/)."
        )
    return key