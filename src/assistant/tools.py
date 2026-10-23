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


def get_user_account(user_id: str) -> dict:
    """Look up a customer's account (name, plan, status) by user ID, e.g. U001."""
    user_id = user_id.strip().upper()

    if not is_valid_id(user_id, "U"):
        return {"error": "Invalid user ID. It should look like U001."}

    user = USERS.get(user_id)
    if user is None:
        return {"error": f"User {user_id} not found."}

    return {"user_id": user_id, **user}


def check_order_status(order_id: str) -> dict:
    """Check an order's item, status and expected delivery date by ID, e.g. O105."""
    order_id = order_id.strip().upper()

    if not is_valid_id(order_id, "O"):
        return {"error": "Invalid order ID. It should look like O105."}

    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Order {order_id} not found."}

    return {"order_id": order_id, **order}


# Allowlist: the ONLY functions the AI is allowed to run.
TOOLS = {
    "get_user_account": get_user_account,
    "check_order_status": check_order_status,
}

# What we tell the AI about each tool: its name, what it does, and its inputs.
# Groq reads this (JSON schema format), it can't read our Python code.
# search_help_docs is not in TOOLS: it needs the doc index, so the agent adds it.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_account",
            "description": get_user_account.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "e.g. U001"}
                },
                "required": ["user_id"],
            },
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
PERSONAL_TOOLS = {"get_user_account", "check_order_status"}
