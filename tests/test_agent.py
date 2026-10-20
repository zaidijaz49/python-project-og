"""Tests for the tool-calling agent. Gemini is faked: no key, no network."""

from unittest.mock import MagicMock

import pytest
from google.genai import errors, types
from tenacity import wait_none

from assistant.agent import AgentError, GeminiAgent, MockAgent, run_tool


def text_answer(text):
    """Fake Gemini response: a final answer, no tool requested."""
    return MagicMock(function_calls=None, text=text)


def tool_request(name, **args):
    """Fake Gemini response: 'please run this tool with these arguments'."""
    call = types.FunctionCall(name=name, args=args)
    content = types.Content(role="model", parts=[types.Part(function_call=call)])
    return MagicMock(function_calls=[call], candidates=[MagicMock(content=content)])


def make_agent(*outcomes):
    """GeminiAgent with a fake SDK. Each outcome is one Gemini call's result."""
    sdk = MagicMock()
    sdk.models.generate_content.side_effect = list(outcomes)
    agent = GeminiAgent(sdk, model="test-model", wait=wait_none())
    return agent, sdk.models.generate_content


def last_tool_result(generate):
    """The tool result we sent back to Gemini in the last call."""
    contents = generate.call_args.kwargs["contents"]
    return contents[-1].parts[0].function_response.response


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


# ---------- GeminiAgent: the loop ----------


def test_direct_answer_without_tools():
    agent, generate = make_agent(text_answer("Hello! How can I help?"))

    result = agent.chat("Hi")

    assert result.answer == "Hello! How can I help?"
    assert result.tools_used == []
    assert generate.call_count == 1


def test_message_is_wrapped_in_tags():
    agent, generate = make_agent(text_answer("Hi"))

    agent.chat("Hello")

    first_message = generate.call_args.kwargs["contents"][0]
    assert first_message.parts[0].text == "<user_text>\nHello\n</user_text>"


def test_tools_are_given_to_gemini_but_not_auto_run():
    agent, generate = make_agent(text_answer("Hi"))

    agent.chat("Hello")

    config = generate.call_args.kwargs["config"]
    assert len(config.tools) == 2
    assert config.automatic_function_calling.disable is True


def test_tool_call_then_answer():
    agent, generate = make_agent(
        tool_request("check_order_status", order_id="O105"),
        text_answer("Your headphones have been shipped."),
    )

    result = agent.chat("Where is my order O105?")

    assert result.answer == "Your headphones have been shipped."
    assert result.tools_used == ["check_order_status"]
    assert generate.call_count == 2
    # Gemini got the real data from our tool in the second call.
    assert last_tool_result(generate)["status"] == "shipped"
    # Tool results must be sent with role "user"; Gemini rejects role "tool".
    assert generate.call_args.kwargs["contents"][-1].role == "user"


def test_order_not_found_is_sent_back_to_gemini():
    agent, generate = make_agent(
        tool_request("check_order_status", order_id="O999"),
        text_answer("I could not find order O999."),
    )

    agent.chat("Where is O999?")

    assert last_tool_result(generate) == {"error": "Order O999 not found."}


def test_unknown_tool_is_not_run():
    agent, generate = make_agent(
        tool_request("delete_user", user_id="U001"),
        text_answer("Sorry, I can't do that."),
    )

    result = agent.chat("Delete user U001")

    assert result.tools_used == []
    assert last_tool_result(generate) == {"error": "Unknown tool: delete_user"}


def test_stops_after_max_rounds():
    agent, generate = make_agent(
        *[tool_request("get_user_account", user_id="U001")] * 5
    )

    with pytest.raises(AgentError):
        agent.chat("Loop forever")

    assert generate.call_count == 5


def test_empty_answer_raises_error():
    agent, _ = make_agent(text_answer(None))

    with pytest.raises(AgentError):
        agent.chat("Hi")


def test_retry_when_gemini_is_busy():
    agent, generate = make_agent(
        errors.ServerError(503, {}),
        text_answer("Hello!"),
    )

    result = agent.chat("Hi")

    assert result.answer == "Hello!"
    assert generate.call_count == 2


def test_no_retry_on_401():
    agent, generate = make_agent(errors.ClientError(401, {}))

    with pytest.raises(errors.ClientError):
        agent.chat("Hi")

    assert generate.call_count == 1


# ---------- MockAgent ----------


def test_mock_agent_returns_fixed_answer():
    mock = MockAgent()

    result = mock.chat("anything")

    assert result.answer == "This is a mock answer."
    assert mock.calls == ["anything"]
