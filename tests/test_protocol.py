"""Validate the supported text-only Chat Completions subset before dispatch."""

import pytest

from muse_proxy.errors import ProxyError
from muse_proxy.protocol import parse_chat


def payload(content="hello", **options):
    return {"messages": [{"role": "user", "content": content}], **options}


def parse(value):
    return parse_chat(value, default_model="muse")


def test_defaults_select_latest_user_without_replaying_history():
    result = parse(
        payload(
            messages=[
                {"role": "system", "content": "Local instructions"},
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "  latest\n🌏  "},
            ],
            temperature=0.1,
            max_tokens=10,
        )
    )
    assert result.text == "  latest\n🌏  "
    assert result.model == "muse"
    assert result.stream is False
    assert result.include_usage is False
    assert result.session_id is None


def test_text_parts_join_in_order_and_session_id_targets_side_chat():
    result = parse(
        payload(
            [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
            model="muse",
            session_id="side-chat-1",
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert result.text == "first\nsecond"
    assert result.stream is True
    assert result.include_usage is True
    assert result.session_id == "side-chat-1"


@pytest.mark.parametrize("value", [None, [], {}, 123, True, "text"])
def test_request_must_be_an_object_with_a_user_message(value):
    with pytest.raises(ProxyError) as failure:
        parse(value)
    assert failure.value.status_code == 400


@pytest.mark.parametrize(
    "messages", [None, [], {}, "text", [None], [{"role": "assistant", "content": "text"}]]
)
def test_missing_or_malformed_message_array_is_rejected(messages):
    with pytest.raises(ProxyError) as failure:
        parse({"messages": messages})
    assert failure.value.status_code == 400


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        " \n\t",
        123,
        True,
        {},
        [],
        ["text"],
        [{"type": "text", "text": None}],
        [{"type": "text", "text": 1}],
        [{"type": "image_url", "image_url": {"url": "https://example.test/image"}}],
        [{"type": "input_audio", "input_audio": {"data": "test", "format": "wav"}}],
        [{"type": [], "text": "bad type"}],
    ],
)
def test_invalid_latest_user_never_falls_back_to_an_earlier_message(content):
    with pytest.raises(ProxyError) as failure:
        parse(
            {
                "messages": [
                    {"role": "user", "content": "Never reuse this text"},
                    {"role": "user", "content": content},
                ]
            }
        )
    assert failure.value.status_code == 400
    assert "Never reuse this text" not in str(failure.value)


@pytest.mark.parametrize("content", ["\ud800", "bad\udfff", [{"type": "text", "text": "\ud800"}]])
def test_lone_unicode_surrogates_are_rejected(content):
    with pytest.raises(ProxyError) as failure:
        parse(payload(content))
    assert failure.value.status_code == 400


def test_character_limit_applies_to_joined_text_and_accepts_exact_boundary():
    assert parse(payload("あ" * 16000)).text == "あ" * 16000
    parts = [{"type": "text", "text": "x" * 8000}, {"type": "text", "text": "y" * 7999}]
    assert len(parse(payload(parts)).text) == 16000
    parts[1]["text"] += "y"
    with pytest.raises(ProxyError):
        parse(payload(parts))
    with pytest.raises(ProxyError):
        parse(payload("x" * 16001))


@pytest.mark.parametrize("stream", ["true", 1, 0, [], {}, None])
def test_stream_requires_a_boolean(stream):
    with pytest.raises(ProxyError):
        parse(payload(stream=stream))


@pytest.mark.parametrize("n", [0, 2, -1, "1", True, None])
def test_only_one_completion_is_supported(n):
    with pytest.raises(ProxyError):
        parse(payload(n=n))


@pytest.mark.parametrize(
    "options",
    [
        {"tools": [{"type": "function", "function": {"name": "test"}}]},
        {"tool_choice": "auto"},
        {"functions": [{"name": "test"}]},
        {"function_call": "auto"},
        {"response_format": {"type": "json_object"}},
        {"response_format": {"type": "json_schema", "json_schema": {}}},
        {"modalities": ["audio"]},
        {"audio": {"voice": "alloy", "format": "wav"}},
    ],
)
def test_unsupported_features_fail_explicitly(options):
    with pytest.raises(ProxyError) as failure:
        parse(payload(**options))
    assert failure.value.status_code == 400


@pytest.mark.parametrize(
    "session_id", ["", "a" * 65, "with_space", "../escape", "こんにちは", 1, []]
)
def test_side_chat_ids_have_a_bounded_safe_syntax(session_id):
    with pytest.raises(ProxyError):
        parse(payload(session_id=session_id))


def test_side_chat_id_boundary_and_main_chat_none():
    assert parse(payload(session_id="a" * 64)).session_id == "a" * 64
    assert parse(payload(session_id=None)).session_id is None


@pytest.mark.parametrize("options", [[], True, {"include_usage": "true"}, {"include_usage": 1}])
def test_stream_options_require_object_and_boolean_include_usage(options):
    with pytest.raises(ProxyError):
        parse(payload(stream=True, stream_options=options))
