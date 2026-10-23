"""Login check: read the Supabase token and find out which customer is asking.

The frontend logs in with Supabase and sends us the token (the "ticket").
We check the ticket is real, then read the customer ID written inside it."""

import os

import jwt
from dotenv import load_dotenv

from assistant.tools import is_valid_id


class AuthError(Exception):
    """The token is missing, fake, expired or has no customer ID."""


def supabase_url_from_env() -> str:
    """Read SUPABASE_URL from .env, e.g. https://abc.supabase.co"""
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_URL is not set. Add it to your .env file.")
    return url


class SupabaseAuth:
    """Checks Supabase login tokens with the project's public keys."""

    def __init__(self, url: str, key_client=None) -> None:
        # Tokens from our project say they were issued by this address.
        self.issuer = f"{url}/auth/v1"
        # Supabase publishes its public keys here. PyJWKClient downloads them
        # once and remembers them. Tests pass a fake key_client instead.
        self.key_client = key_client or jwt.PyJWKClient(
            f"{self.issuer}/.well-known/jwks.json"
        )

    @classmethod
    def from_env(cls) -> "SupabaseAuth":
        return cls(supabase_url_from_env())

    def customer_id(self, token: str) -> str:
        """Check the token and return the customer ID inside it, e.g. "U001"."""
        try:
            key = self.key_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["ES256"],
                audience="authenticated",  # only logged-in users' tokens
                issuer=self.issuer,  # only tokens from OUR project
            )
        except jwt.PyJWTError:
            # Fake signature, expired, wrong project, or not a token at all.
            raise AuthError("Invalid or expired login.") from None

        # app_metadata can only be changed by us (the admin), never by the user,
        # so nobody can make their own token say someone else's customer ID.
        customer_id = claims.get("app_metadata", {}).get("customer_id")
        if not isinstance(customer_id, str) or not is_valid_id(customer_id, "U"):
            raise AuthError("This login is not linked to a customer.")
        return customer_id
