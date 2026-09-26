"""The only module that owns a live client, and the only one that waits.

Three concerns the notebook spread across its nodes, gathered here:

- **the client**, built once per (key, retry policy) and reused;
- **retries**, delegated to the SDK's own backoff rather than hand-rolled;
- **pacing**, which replaces seven ``time.sleep(5)`` calls -- two of them inside ``grade_route``,
  a *router*, which is both a rate-limit concern in a routing function and a sleep on a code path
  that makes no request at all.

Rate limiting is a property of the API key, not of any one node, so a node should not be able to
get it wrong. Nodes here describe *what to ask*; this module decides *when it is allowed to go*.
That is also what makes T5 fast: stub ``LLM.parse`` and the whole refine loop runs with no client,
no network and no waiting.

**Pacing is an interval, not a sleep.** The notebook slept a fixed five seconds after every call,
including the last one in a run, and not at all between two calls that happened to be separated by
a slow one. This module instead enforces a *minimum interval between requests*: it waits only for
the time still outstanding, and never before the first call. Identical protection, no wasted
wall-clock.

The interval is measured from when a call **finished**, including a failed one -- a 429 or a
timeout consumed quota just as a success did, so it starts the clock too.

API keys are read from the environment here and nowhere else (D10). ``Configuration`` holds model
ids and numbers, and T7 snapshots it into every run record; a key in it would write a secret to
``runs/`` once per case.
"""

import os
import time
from dataclasses import dataclass
from functools import cache
from typing import Any, TypeVar

from openai import OpenAI
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel

from curator.parsing import parse_structured
from curator.schemas import Role

ModelT = TypeVar("ModelT", bound=BaseModel)

#: Environment variable holding the key for the extractor/grader roles.
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

#: Environment variable holding the editor/judge key. A separate variable because D4 requires a
#: different model *family*, which in practice means a different vendor and therefore a
#: different key -- and because a judge that silently fell back to the curator key would
#: reproduce the self-enhancement setup it exists to avoid, with nothing in the record saying so.
JUDGE_API_KEY_ENV = "JUDGE_API_KEY"

# When the most recent request finished, on the monotonic clock. Module-level on purpose: the
# quota belongs to the key, so two LLM instances sharing one key must share one clock. `None`
# means nothing has been sent yet, which is why the first call never waits.
_last_call_finished: float | None = None


# --------------------------------------------------------------------------------------
# Pacing
# --------------------------------------------------------------------------------------


def _wait_turn(pause_seconds: float) -> float:
    """Block until at least ``pause_seconds`` has passed since the last request finished.

    Args:
        pause_seconds (float): The minimum interval between requests. Zero or less disables
            pacing entirely, which is what tests and cassette replay (T8) want.

    Returns:
        float: Seconds actually slept -- ``0.0`` if the interval had already elapsed, or if this
        is the first request. Returned rather than discarded so a test can assert on pacing
        without measuring wall-clock time.
    """
    if pause_seconds <= 0 or _last_call_finished is None:
        return 0.0

    outstanding = (_last_call_finished + pause_seconds) - time.monotonic()
    if outstanding <= 0:
        return 0.0

    time.sleep(outstanding)
    return outstanding


def _mark_finished() -> None:
    """Start the interval clock. Called after every request, successful or not."""
    global _last_call_finished
    _last_call_finished = time.monotonic()


def reset_pacing() -> None:
    """Forget when the last request finished, so the next one goes immediately.

    For tests, which must not inherit a clock from whatever ran before them.
    """
    global _last_call_finished
    _last_call_finished = None


# --------------------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------------------


@cache
def _client(api_key_env: str, request_retries: int, base_url: str | None = None) -> OpenAI:
    """Return the shared client for one key, retry policy and endpoint, building it on first use.

    Cached because a client holds a connection pool: one per key is the point of having one.
    Keyed on all three arguments so the editor's judge, which reads a different variable and may
    point somewhere else entirely, gets its own.

    Built lazily rather than at import so that importing ``llm`` -- which ``nodes.py`` does
    unconditionally -- does not require a key to be set.

    Args:
        api_key_env (str): Name of the environment variable holding the key.
        request_retries (int): How many times the SDK retries a failed request itself, with its
            own exponential backoff. This covers transport faults and 429s, which
            ``parsing.interpret_response`` deliberately does not map to curation errors.
        base_url (str | None): Endpoint to talk to, or ``None`` for OpenAI's own. A non-OpenAI
            vendor is reachable this way **only if it serves the Responses API**: everything
            below ``LLM.parse`` is written against that shape, not merely against this SDK.
            See ``review_case`` for what that means for the judge.

    Returns:
        OpenAI: The client.

    Raises:
        KeyError: The variable is unset. Loud on purpose: a missing key is a setup fault, and
            every alternative here is a default that turns it into an auth error later.
    """
    return OpenAI(
        api_key=os.environ[api_key_env], max_retries=request_retries, base_url=base_url
    )


# --------------------------------------------------------------------------------------
# What the nodes call
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LLM:
    """One model, bound to its pacing and retry policy. Built per node call; cheap.

    Frozen and clientless: the client itself lives in ``_client``'s cache, so this object stays
    a plain description of *which* model to ask and *how hard* to try. That is what lets a node
    build one from ``runtime.context`` on every call without cost, and what keeps the seam T5
    stubs -- ``LLM.parse`` -- free of construction details.

    Attributes:
        model (str): The model id, from ``Configuration``.
        pause_seconds (float): Minimum interval between requests sharing this key.
        request_retries (int): Passed to the SDK as ``max_retries``.
        api_key_env (str): Which environment variable holds the key.
        base_url (str | None): Endpoint override, for a role pointed at another vendor. ``None``
            means OpenAI's own. Part of the identity of the client, so two ``LLM``s differing
            only here do not share a connection pool or a key.
        temperature (float | None): Sent with every request when set; **omitted** when
            ``None``. Reasoning models reject the parameter outright, so "not sent" has to be
            expressible -- and it is a different request from sending any number, which is why
            this is ``None`` rather than a default value.
    """

    model: str
    pause_seconds: float
    request_retries: int
    api_key_env: str = DEFAULT_API_KEY_ENV
    temperature: float | None = None
    base_url: str | None = None

    def parse(
        self,
        schema: type[ModelT],
        *,
        input: str | ResponseInputParam,
        case_id: str,
        role: Role,
        **options: Any,
    ) -> ModelT:
        """Wait for this key's turn, then ask for an answer shaped like ``schema``.

        The single seam between the pipeline and the network. Stub this in tests and nothing
        below it runs -- no client is built, no key is read, nothing sleeps.

        Args:
            schema (type[ModelT]): The Pydantic model the answer must match.
            input (str | ResponseInputParam): The prompt, as text or as content parts.
            case_id (str): The case directory name, recorded on any error raised.
            role (Role): Which role is calling, recorded on any error raised.
            **options (Any): Passed through to the request unchanged -- ``max_output_tokens``
                and so on. Not ``temperature``: that belongs to the ``LLM``, and passing it
                here as well raises ``TypeError`` rather than silently picking one.

        Returns:
            ModelT: The validated answer.

        Raises:
            RefusalError: The model refused, or a content filter stopped the output.
            TruncationError: The output hit ``max_output_tokens`` before finishing.
            SchemaError: The model finished and the output still did not match ``schema``.
            openai.APIError: A transport failure that survived ``request_retries`` attempts.
                Not a curation outcome, so it is not mapped to one.
        """
        sampling = {} if self.temperature is None else {"temperature": self.temperature}
        _wait_turn(self.pause_seconds)
        try:
            return parse_structured(
                _client(self.api_key_env, self.request_retries, self.base_url),
                schema=schema,
                model=self.model,
                input=input,
                case_id=case_id,
                role=role,
                **sampling,
                **options,
            )
        finally:
            _mark_finished()
