"""Helper functions (tools) the AI can ask us to run, plus a fake database."""

# Fake database. In a real app this data would come from MongoDB / PostgreSQL.
USERS = {
    "U001": {"name": "Ali", "plan": "Pro", "status": "active"},
    "U002": {"name": "Sara", "plan": "Free", "status": "suspended"},
    "U003": {"name": "Ahmed", "plan": "Business", "status": "active"},
}

ORDERS = {
    "O105": {
        "user_id": "U001",
        "item": "Wireless headphones",
        "status": "shipped",
        "expected_delivery": "2026-10-23",
    },
    "O106": {
        "user_id": "U002",
        "item": "Phone case",
        "status": "pending",
        "expected_delivery": "2026-10-28",
    },
    "O107": {
        "user_id": "U001",
        "item": "Laptop stand",
        "status": "delivered",
        "expected_delivery": "2026-10-15",
    },
}


def is_valid_id(value: str, prefix: str) -> bool:
    """True if value looks like prefix + 3 digits, e.g. U001 or O105."""
    return len(value) == 4 and value[0] == prefix and value[1:].isdigit()


def get_my_account(customer_id: str) -> dict:
    """Look up the logged-in customer's own account: name, plan and status."""
    user = USERS.get(customer_id)
    if user is None:
        return {"error": "Account not found."}

    return {"user_id": customer_id, **user}


def check_order_status(order_id: str, customer_id: str) -> dict:
    """Check one of the customer's own orders by order ID, e.g. O105: item,
    status and expected delivery date."""
    order_id = order_id.strip().upper()

    if not is_valid_id(order_id, "O"):
        return {"error": "Invalid order ID. It should look like O105."}

    order = ORDERS.get(order_id)
    # Someone else's order gets the same answer as a missing one, so nobody
    # can find out which order IDs exist.
    if order is None or order["user_id"] != customer_id:
        return {"error": f"Order {order_id} not found."}

    return {"order_id": order_id, **order}


def tools_for(customer_id: str) -> dict:
    """The tools for ONE logged-in customer (the allowlist).

    The customer ID comes from their login, never from the AI. The AI only
    gives the order ID; if it tries to send a customer ID too, the call fails.
    """
    return {
        "get_my_account": lambda: get_my_account(customer_id),
        "check_order_status": lambda order_id: check_order_status(
            order_id, customer_id
        ),
    }


# What we tell the AI about each tool: its name, what it does, and its inputs.
# Groq reads this (JSON schema format), it can't read our Python code.
# There is no customer ID here: the AI never chooses whose data to read.
# search_help_docs is not in tools_for: it needs the doc index, so the agent adds it.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_my_account",
            "description": get_my_account.__doc__,
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": check_order_status.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "e.g. O105"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_help_docs",
            "description": (
                "Search QuickCart's help docs (orders, payments, refunds and "
                "returns, shipping, account and login, contact). Use it for any "
                "question about how the store works."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The customer's question, in English",
                    }
                },
                "required": ["question"],
            },
        },
    },
]

# Tools that read one customer's own data. Answers that used them are personal,
# so they must never be cached and shown to someone else.
PERSONAL_TOOLS = {"get_my_account", "check_order_status"}
