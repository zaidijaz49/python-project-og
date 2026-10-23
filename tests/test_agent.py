"""Tests for the tool-calling agent. Groq is faked: no key, no network."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx
import pytest
from tenacity import wait_none

from assistant.agent import AgentError, GroqAgent, MockAgent, read_arguments, run_tool
from assistant.tools import TOOL_SCHEMAS


def reply(message):
    """Wrap a message the way Groq does: response.choices[0].message."""
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def text_answer(text):
    """Fake Groq response: a final answer, no tool requested."""
    return reply(SimpleNamespace(content=text, tool_calls=None))


def tool_request(name, **args):
    """Fake Groq response: 'please run this tool with these arguments'."""
    call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )
    return reply(SimpleNamespace(content=None, tool_calls=[call]))


def status_error(code):
    """A Groq error with this HTTP status code, e.g. 429 or 503."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return groq.APIStatusError(f"HTTP {code}", response=response, body=None)


class FakeDocs:
    """Fake help docs: no index, no model. Always finds the refund section."""

    def __init__(self):
        self.questions = []

    def search_help_docs(self, question):
        self.questions.append(question)
        return {
            "sections": [{"id": "refunds.md#3", "text": "Refunds take 7 working days."}]
        }


def make_agent(*outcomes, models=("test-model",), docs=None):
    """GroqAgent with a fake SDK. Each outcome is one Groq call's result."""
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = list(outcomes)
    agent = GroqAgent(sdk, docs=docs, models=list(models), wait=wait_none())
    return agent, sdk.chat.completions.create


def last_tool_result(create):
    """The tool result we sent back to Groq in the last call."""
    last_message = create.call_args.kwargs["messages"][-1]
    assert last_message["role"] == "tool"
    return json.loads(last_message["content"])


# ---------- run_tool: the allowlist ----------


def test_run_tool_runs_allowed_tool():
    assert run_tool("check_order_status", {"order_id": "O105"})["status"] == "shipped"


def test_run_tool_blocks_unknown_tool():
    assert run_tool("delete_user", {"user_id": "U001"}) == {
        "error": "Unknown tool: delete_user"
    }


def test_run_tool_handles_wrong_argument_name():
    assert "error" in run_tool("get_user_account", {"wrong_name": "U001"})


def test_run_tool_handles_wrong_argument_type():
    assert "error" in run_tool("check_order_status", {"order_id": 105})


def test_read_arguments():
    assert read_arguments('{"order_id": "O105"}') == {"order_id": "O105"}
    assert read_arguments("not json") == {}
    assert read_arguments(None) == {}
    assert read_arguments('["a list"]') == {}


# ---------- GroqAgent: the loop ----------


def test_direct_answer_without_tools():
    agent, create = make_agent(text_answer("Hello! How can I help?"))

    result = agent.chat("Hi")

    assert result.answer == "Hello! How can I help?"
    assert result.tools_used == []
    assert create.call_count == 1


def test_message_is_wrapped_in_tags():
    agent, create = make_agent(text_answer("Hi"))

    agent.chat("Hello")

    user_message = create.call_args.kwargs["messages"][1]
    assert user_message["content"] == "<user_text>\nHello\n</user_text>"


def tool_names(create):
    return [schema["function"]["name"] for schema in create.call_args.kwargs["tools"]]


def test_without_docs_only_order_and_account_tools_are_offered():
    agent, create = make_agent(text_answer("Hi"))

    agent.chat("Hello")

    assert tool_names(create) == ["get_user_account", "check_order_status"]
    assert create.call_args.kwargs["tool_choice"] == "auto"


def test_with_docs_all_three_tools_are_offered():
    agent, create = make_agent(text_answer("Hi"), docs=FakeDocs())

    agent.chat("Hello")

    assert create.call_args.kwargs["tools"] == TOOL_SCHEMAS


def test_tool_call_then_answer():
    agent, create = make_agent(
        tool_request("check_order_status", order_id="O105"),
        text_answer("Your headphones have been shipped."),
    )

    result = agent.chat("Where is my order O105?")

    assert result.answer == "Your headphones have been shipped."
    assert result.tools_used == ["check_order_status"]
    assert create.call_count == 2
    # Groq got the real data from our tool in the second call.
    assert last_tool_result(create)["status"] == "shipped"
    # The tool result is linked to the AI's request by its id.
    assert create.call_args.kwargs["messages"][-1]["tool_call_id"] == "call_1"


def test_help_docs_tool_gives_sources():
    docs = FakeDocs()
    agent, create = make_agent(
        tool_request("search_help_docs", question="How long does a refund take?"),
        text_answer("Refunds take 7 working days."),
        docs=docs,
    )

    result = agent.chat("Refund time?")

    assert docs.questions == ["How long does a refund take?"]
    assert result.tools_used == ["search_help_docs"]
    assert result.sources == ["refunds.md#3"]
    assert last_tool_result(create)["sections"][0]["id"] == "refunds.md#3"


def test_order_and_docs_in_one_message():
    # "Where is O105, and how do returns work?" needs both kinds of tool.
    agent, _ = make_agent(
        tool_request("check_order_status", order_id="O105"),
        tool_request("search_help_docs", question="How do returns work?"),
        text_answer("O105 has shipped. You can return it within 14 days."),
        docs=FakeDocs(),
    )

    result = agent.chat("Where is O105, and how do returns work?")

    assert result.tools_used == ["check_order_status", "search_help_docs"]
    assert result.sources == ["refunds.md#3"]


def test_help_docs_tool_is_blocked_without_docs():
    agent, create = make_agent(
        tool_request("search_help_docs", question="Refund time?"),
        text_answer("Sorry, I can't check that right now."),
    )

    result = agent.chat("Refund time?")

    assert result.tools_used == []
    assert last_tool_result(create) == {"error": "Unknown tool: search_help_docs"}


def test_order_not_found_is_sent_back_to_groq():
    agent, create = make_agent(
        tool_request("check_order_status", order_id="O999"),
        text_answer("I could not find order O999."),
    )

    agent.chat("Where is O999?")

    assert last_tool_result(create) == {"error": "Order O999 not found."}


def test_unknown_tool_is_not_run():
    agent, create = make_agent(
        tool_request("delete_user", user_id="U001"),
        text_answer("Sorry, I can't do that."),
    )

    result = agent.chat("Delete user U001")

    assert result.tools_used == []
    assert last_tool_result(create) == {"error": "Unknown tool: delete_user"}


def test_stops_after_max_rounds():
    agent, create = make_agent(*[tool_request("get_user_account", user_id="U001")] * 5)

    with pytest.raises(AgentError):
        agent.chat("Loop forever")

    assert create.call_count == 5


def test_empty_answer_raises_error():
    agent, _ = make_agent(text_answer(None))

    with pytest.raises(AgentError):
        agent.chat("Hi")


def test_retry_when_groq_is_busy():
    agent, create = make_agent(status_error(503), text_answer("Hello!"))

    result = agent.chat("Hi")

    assert result.answer == "Hello!"
    assert create.call_count == 2


def test_no_retry_on_401():
    agent, create = make_agent(status_error(401))

    with pytest.raises(groq.APIStatusError):
        agent.chat("Hi")

    assert create.call_count == 1


# ---------- fallback to the next model ----------


def test_answer_says_which_model_was_used():
    agent, _ = make_agent(text_answer("Hi"), models=["model-a", "model-b"])

    assert agent.chat("Hello").model_used == "model-a"


def test_fallback_restarts_chat_on_next_model():
    agent, create = make_agent(
        tool_request("check_order_status", order_id="O105"),  # model-a round 1
        status_error(429),  # model-a round 2: rate limit used up
        tool_request("check_order_status", order_id="O105"),  # model-b round 1
        text_answer("Your order has shipped."),  # model-b round 2
        models=["model-a", "model-b"],
    )

    result = agent.chat("Where is O105?")

    calls = create.call_args_list
    assert [c.kwargs["model"] for c in calls] == [
        "model-a",
        "model-a",
        "model-b",
        "model-b",
    ]
    # model-b started fresh: a new conversation, not model-a's one continued.
    model_a_chat = calls[0].kwargs["messages"]
    model_b_chat = calls[2].kwargs["messages"]
    assert model_b_chat is not model_a_chat
    assert len(model_b_chat) == 4  # system, user, tool request, tool result
    assert result.model_used == "model-b"
    assert result.tools_used == ["check_order_status"]  # not counted twice


def test_agent_loop_error_does_not_fall_back():
    # Too many tool rounds is not a quota/server problem, so no next model.
    agent, create = make_agent(
        *[tool_request("get_user_account", user_id="U001")] * 5,
        models=["model-a", "model-b"],
    )

    with pytest.raises(AgentError):
        agent.chat("Loop forever")

    assert create.call_count == 5


# ---------- MockAgent ----------


def test_mock_agent_returns_fixed_answer():
    mock = MockAgent()

    result = mock.chat("anything")

    assert result.answer == "This is a mock answer."
    assert mock.calls == ["anything"]
