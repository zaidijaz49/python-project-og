"""Tests for the login check in auth.py. We make our own keys and tokens,
so there is no Supabase and no network."""

import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from assistant.auth import AuthError, SupabaseAuth

URL = "https://test-project.supabase.co"

# Our own "Supabase" key pair. The private key signs tokens (Supabase's job),
# the public key checks them (our backend's job).
PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())  # an attacker's key


class FakeKeyClient:
    """Fake PyJWKClient: always gives our public key, no download."""

    def get_signing_key_from_jwt(self, token):
        return SimpleNamespace(key=PRIVATE_KEY.public_key())


def make_token(key=PRIVATE_KEY, **changes):
    """A token like the one Supabase gives a logged-in user. `changes`
    replaces any field, e.g. make_token(exp=0) for an expired one."""
    claims = {
        "iss": f"{URL}/auth/v1",
        "aud": "authenticated",
        "sub": "6c1f0e2a-0000-0000-0000-000000000000",
        "exp": int(time.time()) + 3600,  # valid for 1 hour
        "role": "authenticated",
        "app_metadata": {"provider": "email", "customer_id": "U001"},
    }
    claims.update(changes)
    return jwt.encode(claims, key, algorithm="ES256")


auth = SupabaseAuth(URL, key_client=FakeKeyClient())


def test_good_token_gives_customer_id():
    assert auth.customer_id(make_token()) == "U001"


def test_not_a_token_is_rejected():
    with pytest.raises(AuthError):
        auth.customer_id("hello")


def test_token_signed_by_someone_else_is_rejected():
    with pytest.raises(AuthError):
        auth.customer_id(make_token(key=OTHER_PRIVATE_KEY))


def test_expired_token_is_rejected():
    with pytest.raises(AuthError):
        auth.customer_id(make_token(exp=int(time.time()) - 10))


def test_token_from_another_project_is_rejected():
    with pytest.raises(AuthError):
        auth.customer_id(make_token(iss="https://other.supabase.co/auth/v1"))


def test_anon_token_is_rejected():
    # The public "anon" key is a token too, but it is not a logged-in user.
    with pytest.raises(AuthError):
        auth.customer_id(make_token(aud="anon"))


def test_token_without_customer_id_is_rejected():
    with pytest.raises(AuthError):
        auth.customer_id(make_token(app_metadata={"provider": "email"}))


def test_bad_customer_id_is_rejected():
    with pytest.raises(AuthError):
        auth.customer_id(make_token(app_metadata={"customer_id": "U0012"}))


def test_customer_id_in_user_metadata_is_ignored():
    # Users can edit their own user_metadata, so we never trust it.
    token = make_token(
        app_metadata={"provider": "email"},
        user_metadata={"customer_id": "U001"},
    )
    with pytest.raises(AuthError):
        auth.customer_id(token)
