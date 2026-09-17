"""Call a model for structured output, and turn every way that can fail into a typed error.

Under strict structured outputs the SDK does the parsing, so this module is not the "tolerant
replacement for ``split_text_to_tag``" PLAN section 5 imagined (D3). Its job is the three failure
modes the notebook swallowed -- a refusal recorded as an approval, a truncation never checked,
a malformed answer dropped on a ``KeyError`` -- each mapped onto ``curator.schemas``' taxonomy.

**Why this does not simply call ``client.responses.parse()`` and read ``output_parsed``** (D18).
In ``openai==3.3.1``, ``parse()`` validates the JSON *inside the call*, as a post-parser, before
anything is returned. A response cut off at the token limit is broken JSON, so ``parse()`` would
raise a Pydantic ``ValidationError`` and the ``status == "incomplete"`` that explains it would
never be seen. Every truncation would then be misreported as a schema error -- the exact
misdiagnosis ``TruncationError`` exists to prevent.

So the request goes through ``with_raw_response.parse()``: the SDK still builds the strict
schema from the Pydantic model -- no private helper is imported here -- but the post-parser only
runs if ``.parse()`` is called on the result, and it never is. The raw JSON body goes to
``interpret_response``, which checks in the one safe order:

1. **Was the output cut short?** Provider filter -> ``RefusalError``; token budget ->
   ``TruncationError``.
2. **Did the model refuse?** -> ``RefusalError``.
3. **Does what it wrote match the schema?** No -> ``SchemaError``.

``interpret_response`` takes a plain dict and makes no network call, so every branch above is
testable offline from a hand-written body. That split is the seam: ``parse_structured`` is the
only line that touches the network, and it contains no decisions.

Transport failures -- rate limits, auth, timeouts -- are **not** mapped. The SDK raises them as
``openai.APIError`` subclasses; they say nothing about the case, and whether to retry them is T5's
decision, not this module's.
"""

from typing import Any, TypeVar

from openai import OpenAI
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, ValidationError

from curator.schemas import RefusalError, Role, SchemaError, StoppedBy, TruncationError

ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_structured(
    client: OpenAI,
    *,
    schema: type[ModelT],
    model: str,
    input: str | ResponseInputParam,
    case_id: str,
    role: Role,
    **options: Any,
) -> ModelT:
    """Ask ``model`` for an answer shaped like ``schema``, and return it or raise why not.

    Args:
        client: An ``OpenAI`` client. Its API key comes from the environment, never from
            ``Configuration`` (D10).
        schema: The Pydantic model the answer must match -- ``Grading``, ``Draft`` or
            ``EditorVerdict``. Sent as a strict JSON schema.
        model: The model id.
        input: The prompt, as a string or a list of input items (text and images).
        case_id: The case directory name, recorded on any error raised.
        role: Which role is calling, recorded on any error raised.
        **options: Passed through to the request unchanged -- ``temperature``,
            ``max_output_tokens`` and so on.

    Returns:
        ModelT: The validated answer.

    Raises:
        RefusalError: The model refused, or the provider's content filter stopped the output.
        TruncationError: The output reached ``max_output_tokens`` before it finished.
        SchemaError: The model finished, and the output still did not match ``schema``.
        openai.APIError: A transport failure. Not a curation outcome; not mapped.
    """
    raw = client.responses.with_raw_response.parse(
        text_format=schema, model=model, input=input, **options
    )
    return interpret_response(raw.http_response.json(), schema=schema, case_id=case_id, role=role)


def interpret_response(
    body: dict[str, Any],
    *,
    schema: type[ModelT],
    case_id: str,
    role: Role,
) -> ModelT:
    """Turn a raw Responses API body into a validated answer, or the typed error that explains it.

    Pure: no client, no network. Every key is read by indexing rather than ``.get`` on purpose. A
    body without ``status`` or ``output`` is not a curation outcome but a broken contract with the
    API, and a ``KeyError`` that says so beats a default that quietly reads as "no refusal".

    Args:
        body: The response JSON, as ``http_response.json()`` returns it.
        schema: The Pydantic model the answer must match.
        case_id: The case directory name, recorded on any error raised.
        role: Which role was calling, recorded on any error raised.

    Returns:
        ModelT: The validated answer.

    Raises:
        RefusalError: See ``parse_structured``.
        TruncationError: See ``parse_structured``.
        SchemaError: See ``parse_structured``.
        RuntimeError: A status or incomplete reason this module was not written for. Only
            ``completed`` and ``incomplete`` occur outside background mode, which the pipeline
            does not use; anything else means the API has changed under us, and crashing is
            better than filing it under a category it does not belong to.
    """
    status = body["status"]

    # 1. Cut short? Checked first: a truncated output is also broken JSON, and would otherwise
    #    fall through to step 3 and be misreported as a schema error.
    if status == "incomplete":
        reason = body["incomplete_details"]["reason"]
        if reason == "content_filter":
            raise RefusalError(
                message=f"{role} output for {case_id} was stopped by the provider's content filter",
                case_id=case_id,
                role=role,
                refusal_text=None,
                stopped_by=StoppedBy.PROVIDER,
            )
        if reason == "max_output_tokens":
            raise TruncationError(
                message=f"{role} output for {case_id} reached max_output_tokens",
                case_id=case_id,
                role=role,
            )
        raise RuntimeError(f"unrecognised incomplete reason {reason!r} for {case_id}")

    if status != "completed":
        raise RuntimeError(f"unexpected response status {status!r} for {case_id}")

    # 2. Refused? A refusal arrives as its own content item, beside or instead of output text.
    #    Collected in one pass with the text, mirroring the SDK's own ``Response.output_text``.
    refusals: list[str] = []
    texts: list[str] = []
    for item in body["output"]:
        if item["type"] != "message":
            continue
        for content in item["content"]:
            if content["type"] == "refusal":
                refusals.append(content["refusal"])
            elif content["type"] == "output_text":
                texts.append(content["text"])

    if refusals:
        raise RefusalError(
            message=f"{role} refused to answer for {case_id}",
            case_id=case_id,
            role=role,
            refusal_text="\n".join(refusals),
            stopped_by=StoppedBy.MODEL,
        )

    # 3. Matches the schema? An empty ``output_text`` is not valid JSON, so a completed response
    #    with nothing in it lands here too -- as a schema error, with the empty text kept.
    raw_output = "".join(texts)
    try:
        return schema.model_validate_json(raw_output)
    except ValidationError as err:
        raise SchemaError(
            message=f"{role} output for {case_id} did not match {schema.__name__}",
            case_id=case_id,
            role=role,
            raw_output=raw_output,
        ) from err
