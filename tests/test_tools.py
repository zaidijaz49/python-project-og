"""Tests for the helper functions in tools.py. No AI involved."""

from assistant.tools import (
    ORDERS,
    TOOL_SCHEMAS,
    check_order_status,
    get_my_account,
    is_valid_id,
    tools_for,
)


def test_valid_ids():
    assert is_valid_id("U001", "U")
    assert is_valid_id("O105", "O")


def test_invalid_ids():
    assert not is_valid_id("U01", "U")  # too short
    assert not is_valid_id("X001", "U")  # wrong letter
    assert not is_valid_id("UABC", "U")  # not digits
    assert not is_valid_id("U001; drop table", "U")  # tricky input


def test_get_my_account_found():
    result = get_my_account("U001")

    assert result == {
        "user_id": "U001",
        "name": "Ali",
        "plan": "Pro",
        "status": "active",
    }


def test_get_my_account_not_found():
    assert get_my_account("U999") == {"error": "Account not found."}


def test_check_own_order():
    result = check_order_status("O105", "U001")

    assert result["item"] == "Wireless headphones"
    assert result["status"] == "shipped"


def test_order_id_accepts_lowercase_and_spaces():
    assert check_order_status("  o105 ", "U001")["status"] == "shipped"


def test_someone_elses_order_looks_not_found():
    # O106 belongs to U002. U001 must not see it, or even learn that it exists.
    assert check_order_status("O106", "U001") == {"error": "Order O106 not found."}


def test_check_order_status_not_found():
    assert check_order_status("O999", "U001") == {"error": "Order O999 not found."}


def test_check_order_status_invalid_id():
    assert "error" in check_order_status("105", "U001")


def test_tools_do_not_change_the_database():
    result = check_order_status("O105", "U001")
    result["status"] = "hacked"

    assert ORDERS["O105"]["status"] == "shipped"


def test_tools_for_is_locked_to_one_customer():
    tools = tools_for("U002")

    assert tools["get_my_account"]()["user_id"] == "U002"
    assert tools["check_order_status"](order_id="O106")["status"] == "pending"
    assert "error" in tools["check_order_status"](order_id="O105")  # U001's order


def test_allowlist_has_only_our_tools():
    assert set(tools_for("U001")) == {"get_my_account", "check_order_status"}


def test_every_tool_has_a_schema():
    # Groq only knows about tools listed in TOOL_SCHEMAS, so the two must match.
    # search_help_docs is added by the agent, because it needs the doc index.
    schema_names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert schema_names == [*tools_for("U001"), "search_help_docs"]


def test_schemas_never_ask_the_ai_for_a_customer_id():
    for schema in TOOL_SCHEMAS:
        assert "customer_id" not in schema["function"]["parameters"]["properties"]
        assert "user_id" not in schema["function"]["parameters"]["properties"]
