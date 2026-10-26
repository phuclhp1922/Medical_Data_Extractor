r"""A real run, recorded once, replayed through the whole pipeline with no network (T8).

Everything above the network is real: the corpus, the prompts, the graph, ``LLM.parse``,
``parsing.py``'s interpretation of each response. Only the HTTP transport is swapped for a
``CassetteTransport`` (``tests/cassette.py``). That makes this the one test that exercises the
code between the graph and the wire -- the part every other test stubs out.

**pytest never spends money here.** Replay is the only mode a test uses. Recording is a separate
command, run by hand, once:

    .venv\Scripts\python.exe -m tests.test_replay

It reads the API keys from ``.env``, makes at most ``1 + 2 x (1 + 1) = 5`` calls on case 5, and
writes ``tests/cassettes/case-5-max-refines-1.json``. It refuses to overwrite an existing one.

**The settings are pinned below, not read from ``.env``.** Replay matches each request by a hash of
its exact body, and the body contains the model ids, temperatures and image detail. Reading them
from ``.env`` would make the test pass or fail depending on what someone last set there.

**The cassette is gitignored** (``tests/cassettes/``): its responses are the model's teaching case,
which quotes the copyrighted source. Whether to record a committable one against an open-access
case instead is D1 in the T8 brief -- the user's decision, still open.

**Note on authorship.** The agent wrote the transport, this harness and the recording command;
every ``assert`` and ``pytest.raises`` is yours.
"""

from pathlib import Path
from types import SimpleNamespace

import httpx2
import openai  # noqa: F401
import pytest
from dotenv import load_dotenv

import curator.llm as llm_module
import curator.prompts.grader as grader_module  # noqa: F401
from curator.config import (
    CORPUS_DIR_ENV,
    CURATOR_MODEL_ENV,
    IMAGE_DETAIL_ENV,
    JUDGE_MODEL_ENV,
    JUDGE_TEMPERATURE_ENV,
    MAX_REFINES_ENV,
    PAUSE_SECONDS_ENV,
    REQUEST_RETRIES_ENV,
    TEMPERATURE_ENV,
    Configuration,
    corpus_dir_from_env,
)
from curator.graph import build_graph
from curator.llm import DEFAULT_API_KEY_ENV, JUDGE_API_KEY_ENV
from curator.records import summarise_usage
from curator.run import list_sources, select_sources
from curator.state import TerminalState  # noqa: F401
from tests.cassette import CassetteMiss, CassetteTransport  # noqa: F401

#: The case recorded: the shortest source text among the first ten, so the cheapest to record.
CASE = 5

#: Where the recording lives. Gitignored -- see the module docstring.
CASSETTE = Path(__file__).parent / "cassettes" / f"case-{CASE}-max-refines-1.json"

_REPO_ROOT = Path(__file__).resolve().parents[1]


def pinned_config(corpus_dir: Path, *, recording: bool) -> Configuration:
    """The settings the cassette was recorded under. Change one and replay misses -- on purpose.

    ``max_refines=1`` lets the run take one retry, so the retry prompt is exercised if the editor
    complains, while capping the recording at five calls. Pacing is on only while recording.
    """
    return Configuration.from_env(
        {
            CURATOR_MODEL_ENV: "gpt-4.1",
            JUDGE_MODEL_ENV: "gpt-5.5",
            TEMPERATURE_ENV: "0.1",
            JUDGE_TEMPERATURE_ENV: "none",
            MAX_REFINES_ENV: "1",
            IMAGE_DETAIL_ENV: "auto",
            PAUSE_SECONDS_ENV: "2" if recording else "0",
            REQUEST_RETRIES_ENV: "0",
            CORPUS_DIR_ENV: str(corpus_dir),
        }
    )


def run_through(
    transport: CassetteTransport,
    patcher: pytest.MonkeyPatch,
    corpus_dir: Path,
    *,
    recording: bool,
) -> dict:
    """Run case ``CASE`` through the real graph, with every HTTP request going to ``transport``.

    The client is real; only its transport is replaced, by handing ``OpenAI`` an ``httpx2``
    client built on the cassette. ``_client`` is cached, so the cache is cleared on both sides
    -- a client built without the cassette must never be reused, and this one must not leak out.
    """
    real_openai = llm_module.OpenAI

    def openai_on_cassette(**kwargs):
        return real_openai(**kwargs, http_client=httpx2.Client(transport=transport, timeout=600.0))

    patcher.setattr(llm_module, "OpenAI", openai_on_cassette)
    llm_module._client.cache_clear()
    llm_module.reset_pacing()
    source = select_sources(list_sources(corpus_dir), [str(CASE)], None)[0]
    try:
        return build_graph().invoke(
            {"source": source}, context=pinned_config(corpus_dir, recording=recording)
        )
    finally:
        llm_module._client.cache_clear()


@pytest.fixture
def replay(monkeypatch):
    """Replay case ``CASE`` from its cassette. Skips if the corpus or the cassette is missing.

    Returns an object with ``run()`` -- runs the case and returns the final state, as
    ``build_graph().invoke`` does -- and ``transport``, whose ``unused`` says how many recorded
    responses were never asked for. Call ``run()`` once per test.

    The API keys are set to placeholders: the client insists on one, and none is ever sent
    anywhere, because the transport answers every request itself.
    """
    corpus_dir = corpus_dir_from_env()
    if not corpus_dir.is_dir():
        pytest.skip("needs the local corpus (extracted_data/, not redistributable)")
    if not CASSETTE.exists():
        pytest.skip("no cassette yet -- record one with: python -m tests.test_replay")

    monkeypatch.setenv(DEFAULT_API_KEY_ENV, "replay-sends-nothing")
    monkeypatch.setenv(JUDGE_API_KEY_ENV, "replay-sends-nothing")
    transport = CassetteTransport(CASSETTE, record=False)
    return SimpleNamespace(
        run=lambda: run_through(transport, monkeypatch, corpus_dir, recording=False),
        transport=transport,
    )


# --------------------------------------------------------------------------------------
# Replay  [yours]
# --------------------------------------------------------------------------------------


def test_a_recorded_run_replays_offline(replay):
    """The whole pipeline runs from the cassette and reaches one of the four exits.

    Two claims worth making: how the run ended, and that it asked for every response the
    recording holds -- ``replay.transport.unused`` -- no more (that would be a ``CassetteMiss``)
    and no fewer (the replay took a shorter path than the real run did).
    """
    pytest.skip("yours")


def test_the_replay_keeps_every_draft_review_and_call(replay):
    """The history survives the round trip through real response parsing.

    Available in the final state: ``drafts``, ``verdicts`` and ``usage`` (one entry per call).
    Think about how the three lengths relate for a run that got past grading.
    """
    pytest.skip("yours")


def test_a_changed_prompt_misses_the_cassette(replay, monkeypatch):
    """Replay is strict: a request that differs by one byte is not answered from the recording.

    Setup idea: wrap ``grader_module.render`` with ``monkeypatch.setattr`` so it appends a single
    space, then ``replay.run()``. The grader's request hash no longer matches.

    What surfaces is not ``CassetteMiss`` itself: the OpenAI client catches transport failures
    and raises its own ``openai.APIConnectionError``, with the original on ``__cause__``. Claim
    both.
    """
    pytest.skip("yours")


# --------------------------------------------------------------------------------------
# Recording -- run by hand, never by pytest
# --------------------------------------------------------------------------------------


def record() -> None:
    """Record the cassette: one real run of case ``CASE``. Costs money -- at most 5 calls."""
    load_dotenv(_REPO_ROOT / ".env")
    corpus_dir = corpus_dir_from_env()
    transport = CassetteTransport(CASSETTE, record=True)  # refuses if the cassette exists
    with pytest.MonkeyPatch.context() as patcher:
        out = run_through(transport, patcher, corpus_dir, recording=True)
    usage = [call.model_dump(mode="json") for call in out["usage"]]
    print(f"recorded {len(transport.entries)} call(s), ended {out['terminal_state'].value}")
    print(f"tokens   {summarise_usage(usage)}")
    for call in usage:
        print(
            f"  {call['role']:<9} {call['model']:<8} in {call['input_tokens']}"
            f" (cached {call['cached_input_tokens']})  out {call['output_tokens']}"
            f" (reasoning {call['reasoning_tokens']})"
        )
    print(f"cassette {CASSETTE}")


if __name__ == "__main__":
    record()
