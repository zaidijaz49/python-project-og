"""Tests for the helper functions in tools.py. No AI involved."""

from assistant.tools import (
    ORDERS,
    TOOL_SCHEMAS,
    TOOLS,
    check_order_status,
    get_user_account,
    is_valid_id,
)


def test_valid_ids():
    assert is_valid_id("U001", "U")
    assert is_valid_id("O105", "O")


def test_invalid_ids():
    assert not is_valid_id("U01", "U")  # too short
    assert not is_valid_id("X001", "U")  # wrong letter
    assert not is_valid_id("UABC", "U")  # not digits
    assert not is_valid_id("U001; drop table", "U")  # tricky input


def test_get_user_account_found():
    result = get_user_account("U001")

    assert result == {
        "user_id": "U001",
        "name": "Ali",
        "plan": "Pro",
        "status": "active",
    }


def test_get_user_account_accepts_lowercase_and_spaces():
    assert get_user_account("  u001 ")["name"] == "Ali"


def test_get_user_account_not_found():
    assert get_user_account("U999") == {"error": "User U999 not found."}


def test_get_user_account_invalid_id():
    assert "error" in get_user_account("hello")


def test_check_order_status_found():
    result = check_order_status("O105")

    assert result["item"] == "Wireless headphones"
    assert result["status"] == "shipped"


def test_check_order_status_not_found():
    assert check_order_status("O999") == {"error": "Order O999 not found."}


def test_check_order_status_invalid_id():
    assert "error" in check_order_status("105")


def test_tools_do_not_change_the_database():
    result = check_order_status("O105")
    result["status"] = "hacked"

    assert ORDERS["O105"]["status"] == "shipped"


def test_allowlist_has_only_our_tools():
    assert set(TOOLS) == {"get_user_account", "check_order_status"}


def test_every_tool_has_a_schema():
    # Groq only knows about tools listed in TOOL_SCHEMAS, so the two must match.
    schema_names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert schema_names == list(TOOLS)
