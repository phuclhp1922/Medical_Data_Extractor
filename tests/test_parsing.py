"""Tests for ``curator.parsing``: every way a model call fails maps to the right typed error.

**No network, and no stub of our own making.** The client is a real ``OpenAI`` client whose HTTP
transport is replaced by ``httpx2.MockTransport``, which answers every request with a
hand-written JSON body. So the SDK's own request-building and response-handling run for real --
including the post-parser D18 exists to avoid -- and only the wire is fake. ``httpx2`` is the
HTTP library ``openai==3.3.1`` is built on; if an upgrade changes that, the import below breaks,
which is the right place to find out.

**Note on authorship.** ``parsing.py`` was written by the agent, so the agent has written no
assertion here -- the helpers below are the agent's, every ``assert`` and ``pytest.raises`` is
the test author's (learning record 0001). The work order marked these tests **[agent]**; the
authorship rule overrides it, because the agent also wrote the code under test.

Imports you will need that the harness does not use yet: ``CurationError``, ``RefusalError``,
``TruncationError``, ``SchemaError``, ``StoppedBy`` from ``curator.schemas``, and
``ValidationError`` from ``pydantic`` for the ``__cause__`` check.
"""

import json

import httpx2
import pytest
from openai import OpenAI
from pydantic import ValidationError

from curator.parsing import parse_structured
from curator.schemas import (
    EditorVerdict,
    RefusalError,
    Role,
    SchemaError,
    StoppedBy,
    TruncationError,
)

CASE_ID = "case_28"
ROLE = Role.EDITOR

# --------------------------------------------------------------------------------------
# Harness  [agent]
# --------------------------------------------------------------------------------------


def _text(text: str) -> dict:
    """An ``output_text`` content item -- what the model wrote."""
    return {"type": "output_text", "text": text, "annotations": []}


def _refusal(text: str) -> dict:
    """A ``refusal`` content item -- the model declining, in its own words."""
    return {"type": "refusal", "refusal": text}


def _body(*contents: dict, status: str = "completed", reason: str | None = None) -> dict:
    """A Responses API body with one assistant message holding ``contents``.

    Only the fields ``parsing.py`` and the SDK's response model need are filled in.

    Args:
        *contents: Content items, built with ``_text`` / ``_refusal``.
        status: ``"completed"``, or ``"incomplete"`` together with ``reason``.
        reason: ``"max_output_tokens"`` or ``"content_filter"`` when incomplete.

    Returns:
        dict: The response JSON.
    """
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": list(contents),
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _call(body: dict, *, sent: dict | None = None, **options):
    """Run ``parse_structured`` for an ``EditorVerdict`` against a client that answers ``body``.

    Args:
        body: The JSON the fake server returns.
        sent: If given, filled with the JSON request the SDK actually sent.
        **options: Passed through to ``parse_structured`` -- e.g. ``temperature``.

    Returns:
        EditorVerdict: Whatever ``parse_structured`` returns. Raises whatever it raises.
    """

    def answer(request: httpx2.Request) -> httpx2.Response:
        if sent is not None:
            sent.update(json.loads(request.content))
        return httpx2.Response(200, json=body)

    client = OpenAI(
        api_key="offline-test-key",
        base_url="http://offline.invalid/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )
    return parse_structured(
        client,
        schema=EditorVerdict,
        model="test-model",
        input="prompt",
        case_id=CASE_ID,
        role=ROLE,
        **options,
    )


VALID = '{"concerns": []}'
TRUNCATED = '{"concerns": [{"section": "SOURCE_FIDE'
OUT_OF_SCHEMA = '{"concerns": [{"section": "REASONING_EXTRA_INFO", "explanation": "x"}]}'
REFUSAL_TEXT = "I can't help with that."


# --------------------------------------------------------------------------------------
# The happy path, and what was sent
# --------------------------------------------------------------------------------------


def test_valid_output_returns_the_model():
    """``_call(_body(_text(VALID)))`` returns an ``EditorVerdict`` with no concerns."""
    verdict = _call(_body(_text(VALID)))
    assert isinstance(verdict, EditorVerdict)
    assert verdict.concerns == []


def test_request_is_sent_strict():
    """The request the SDK sends carries the schema with ``strict`` true, and passes options on.

    Call with ``sent={}`` and ``temperature=0.1``, then look at ``sent["text"]["format"]
    ["strict"]`` and ``sent["temperature"]``. Strict mode is what every guarantee in D11 rests on;
    if it silently stopped being sent, every other test here would still pass.
    """
    sent = {}
    _call(_body(_text(VALID)), sent=sent, temperature=0.1)
    assert sent["text"]["format"]["strict"] is True
    assert sent["temperature"] == 0.1


# --------------------------------------------------------------------------------------
# Cut short  (D14, D18)
# --------------------------------------------------------------------------------------


def test_truncation_is_not_a_schema_error():
    """A body cut off at ``max_output_tokens`` raises ``TruncationError`` -- not ``SchemaError``.

    **The D18 regression test.** Use ``_body(_text(TRUNCATED), status="incomplete",
    reason="max_output_tokens")``: broken JSON *and* an incomplete status. Plain
    ``responses.parse()`` raises a Pydantic ``ValidationError`` on exactly this body. If anyone
    "simplifies" the wrapper back onto it, this is the test that fails.
    """
    with pytest.raises(TruncationError) as excinfo:
        _call(_body(_text(TRUNCATED), status="incomplete", reason="max_output_tokens"))
    assert excinfo.value.case_id == CASE_ID
    assert excinfo.value.role == ROLE


def test_content_filter_is_a_provider_refusal_without_text():
    """``reason="content_filter"`` raises ``RefusalError`` with ``stopped_by`` PROVIDER, no text.

    Check the class, ``stopped_by``, and that ``refusal_text is None`` -- ``is None``, not
    falsy, since ``""`` would be the coercion D17 rules out.
    """
    with pytest.raises(RefusalError) as excinfo:
        _call(_body(status="incomplete", reason="content_filter"))
    assert excinfo.value.case_id == CASE_ID
    assert excinfo.value.role == ROLE
    assert excinfo.value.stopped_by == StoppedBy.PROVIDER
    assert excinfo.value.refusal_text is None


# --------------------------------------------------------------------------------------
# Refused  (D14, D17)
# --------------------------------------------------------------------------------------


def test_model_refusal_keeps_its_words():
    """A ``refusal`` content item raises ``RefusalError``, ``stopped_by`` MODEL, text preserved.

    **A refusal is not an approval.** The notebook turned this exact body into
    ``flags="NONE"``. Check ``refusal_text == REFUSAL_TEXT``: the words are the evidence.
    """
    with pytest.raises(RefusalError) as excinfo:
        _call(_body(_refusal(REFUSAL_TEXT)))
    assert excinfo.value.case_id == CASE_ID
    assert excinfo.value.role == ROLE
    assert excinfo.value.stopped_by == StoppedBy.MODEL
    assert excinfo.value.refusal_text == REFUSAL_TEXT


# --------------------------------------------------------------------------------------
# Finished, but wrong  (D18)
# --------------------------------------------------------------------------------------


def test_out_of_schema_output_is_a_schema_error_with_evidence():
    """A completed body that fails validation raises ``SchemaError`` carrying everything.

    Check ``raw_output == OUT_OF_SCHEMA``, and that ``__cause__`` is a Pydantic
    ``ValidationError`` -- the field-level detail lives there rather than in a field.
    """
    with pytest.raises(SchemaError) as excinfo:
        _call(_body(_text(OUT_OF_SCHEMA)))
    assert excinfo.value.case_id == CASE_ID
    assert excinfo.value.role == ROLE
    assert excinfo.value.raw_output == OUT_OF_SCHEMA
    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_empty_output_is_a_schema_error():
    """A completed body with no text at all raises ``SchemaError``, with ``raw_output == ""``.

    ``_body()`` with no contents. An empty answer is not a valid ``EditorVerdict`` -- and in
    particular is not an empty list of concerns.
    """
    with pytest.raises(SchemaError) as excinfo:
        _call(_body())
    assert excinfo.value.case_id == CASE_ID
    assert excinfo.value.role == ROLE
    assert excinfo.value.raw_output == ""


# --------------------------------------------------------------------------------------
# Every failure is traceable  (D15)
# --------------------------------------------------------------------------------------


def test_errors_carry_case_and_role():
    """Whichever error is raised, it carries ``case_id == CASE_ID`` and ``role == ROLE``.

    One failure is enough -- the truncation body, say. Catch it as ``CurationError`` if you
    like; that is the point of the base class.
    """
    with pytest.raises(TruncationError) as excinfo:
        _call(_body(_text(TRUNCATED), status="incomplete", reason="max_output_tokens"))
    assert excinfo.value.case_id == CASE_ID
    assert excinfo.value.role == ROLE