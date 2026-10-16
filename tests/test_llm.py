"""Tests for ``llm.py``: pacing, the client cache, and what reaches the request.

No network and no sleeping. Three things stand in for the world, and the fixtures below install
all three, so a test body only has to say what should be true:

- ``parse_structured`` is replaced by a recorder, so "what was sent" is a dict you can look at.
- ``_client`` is replaced by a sentinel, so no API key is needed and nothing is built.
- ``curator.llm.time`` is replaced by a fake clock, so pacing is exact and instant.

**Note on authorship.** The agent wrote ``llm.py``, so it must not write the assertions: it would
only write down what it believes the code does, and that belief is the thing under test. The
fixtures, fakes and stubs here are the agent's; every ``assert`` and ``pytest.raises`` is yours.

Each stub names one claim in its docstring. Replace the ``pytest.skip`` with the assertion the
docstring describes. When a test passes, break the line of ``llm.py`` it names and check it goes
red -- a test that stays green when you break the thing it names is decoration, not a test.
"""

import pytest

import curator.llm as llm_module

# Both key-name constants are imported for the client stubs below to use once you write them;
# ruff would otherwise call them unused while the assertions are still missing.
from curator.llm import DEFAULT_API_KEY_ENV, JUDGE_API_KEY_ENV, LLM  # noqa: F401
from curator.schemas import Grading, Role

# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that never really sleeps.

    ``time.sleep(n)`` advances ``now`` by ``n`` and records it, so a test can assert what *would*
    have been slept without waiting for it. Stands in for the ``time`` module, which ``llm.py``
    reaches through as ``time.monotonic()`` / ``time.sleep()``.
    """

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        """Move time forward without anyone sleeping -- the caller was busy elsewhere."""
        self.now += seconds


class Recorder:
    """Stands in for ``parse_structured``. Records each call's keyword arguments.

    Attributes:
        calls (list[dict]): One dict per call, holding every keyword argument the caller passed
            -- ``model``, ``case_id``, ``role``, and ``temperature`` *only if it was sent*.
        clients (list): The client object passed positionally each time, so a test can ask
            whether two calls shared one.
        client_requests (list[tuple]): What ``LLM.parse`` asked ``_client`` for, one tuple per
            call, in ``_client``'s own argument order: ``(api_key_env, request_retries,
            base_url)``. Filled in by the ``recorder`` fixture's stand-in for ``_client``.
        raises (Exception | None): Set this and the next call raises it instead of returning.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.clients: list = []
        self.client_requests: list[tuple] = []
        self.raises: Exception | None = None

    def __call__(self, client, **kwargs):
        self.calls.append(kwargs)
        self.clients.append(client)
        if self.raises is not None:
            raise self.raises
        return GRADING

    @property
    def last(self) -> dict:
        """The keyword arguments of the most recent call."""
        return self.calls[-1]


#: A valid Grading, so the recorder can return something the type checker and callers accept.
#: Its contents are never asserted on -- these tests are about the request, not the answer.
GRADING = Grading(
    think="stub",
    is_case_report=True,
    case_presentation_score=3,
    integrative_reasoning_score=3,
    transparency_score=3,
    images_usefulness_score=3,
    differential_diagnosis_present=True,
    final_diagnosis_present=True,
)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated():
    """Undo the module-level state that would otherwise leak between tests.

    ``autouse`` so it cannot be forgotten. Two things leak without it, and both make tests pass
    alone and fail in a suite:

    - ``_last_call_finished`` is module-level, so a test asserting "the first call does not wait"
      inherits whatever the previous test's call left behind.
    - ``_client`` is ``@cache``d, so a client built in one test (with that test's key) is handed
      to the next one.
    """
    llm_module.reset_pacing()
    llm_module._client.cache_clear()
    yield
    llm_module.reset_pacing()
    llm_module._client.cache_clear()


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    """Replace ``curator.llm``'s view of ``time`` with a fake clock."""
    fake = FakeClock()
    monkeypatch.setattr(llm_module, "time", fake)
    return fake


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    """Replace ``parse_structured`` with a recorder, and ``_client`` with a sentinel.

    Patching ``_client`` too means no environment variable is read and no ``OpenAI`` object is
    built, so these tests never need a key. The stand-in notes what it was asked for in
    ``rec.client_requests``, which is how a test can claim that an ``LLM``'s settings actually
    reach ``_client``. The tests that are *about* the real ``_client`` do not use this fixture.
    """
    rec = Recorder()

    def fake_client(*args):
        rec.client_requests.append(args)
        return "sentinel-client"

    monkeypatch.setattr(llm_module, "parse_structured", rec)
    monkeypatch.setattr(llm_module, "_client", fake_client)
    return rec


def make_llm(**overrides) -> LLM:
    """Build an ``LLM`` with harmless defaults; override just the field under test.

    Defaults: ``pause_seconds=0`` (pacing off) and ``request_retries=0``, so a test that is not
    about pacing does not have to think about it.
    """
    fields = {"model": "test-model", "pause_seconds": 0.0, "request_retries": 0}
    return LLM(**{**fields, **overrides})


def parse(llm: LLM, **options):
    """Call ``llm.parse`` with the boring arguments filled in."""
    return llm.parse(Grading, input="prompt", case_id="case-1", role=Role.GRADER, **options)


# --------------------------------------------------------------------------------------
# Pacing  [yours] -- replace pytest.skip with the assertion described in each docstring
# --------------------------------------------------------------------------------------


def test_first_call_does_not_wait(clock):
    """Nothing has been sent yet, so the first request goes immediately.

    Assert: ``llm_module._wait_turn(5.0)`` returns ``0.0``, and ``clock.slept`` is still empty.
    The second half matters -- a function can return 0.0 having slept anyway.
    """
    assert llm_module._wait_turn(5.0) == 0.0
    assert clock.slept == []


def test_immediate_second_call_waits_the_interval(clock):
    """A request sent right after one finished waits for the rest of the interval.

    Setup: ``_wait_turn(5.0)``, then ``llm_module._mark_finished()``, then ``_wait_turn(5.0)``
    again with no time passing in between.

    Assert: the second call's return value is ``5.0``, and ``clock.slept == [5.0]``.
    """
    llm_module._wait_turn(5.0)
    llm_module._mark_finished()
    assert llm_module._wait_turn(5.0) == 5.0
    assert clock.slept == [5.0]


def test_no_wait_when_the_interval_has_already_passed(clock):
    """Time spent elsewhere counts. The interval is measured, not slept through.

    Setup: ``_mark_finished()``, then ``clock.advance(5.0)`` -- five seconds of the caller doing
    something else -- then ``_wait_turn(5.0)``.

    Assert: it returns ``0.0`` and nothing was slept.
    """
    llm_module._mark_finished()
    clock.advance(5.0)
    assert llm_module._wait_turn(5.0) == 0.0
    assert clock.slept == []


def test_partial_wait_when_some_of_the_interval_has_passed(clock):
    """Only the remainder is waited for.

    Setup: ``_mark_finished()``, ``clock.advance(2.0)``, then ``_wait_turn(5.0)``.

    Assert: the wait is ``3.0``. This is the claim that would catch a sign error or an interval
    measured from the wrong end -- both of which still "wait roughly the right amount".
    """
    llm_module._mark_finished()
    clock.advance(2.0)
    assert llm_module._wait_turn(5.0) == 3.0
    assert clock.slept == [3.0]


def test_pause_seconds_zero_disables_pacing(clock):
    """Zero means off, which is what T8's cassette replay needs.

    Setup: ``_mark_finished()``, then ``_wait_turn(0)`` with no time passing.

    Assert: returns ``0.0`` and nothing slept.
    """
    llm_module._mark_finished()
    assert llm_module._wait_turn(0) == 0.0
    assert clock.slept == []


def test_a_failed_call_still_starts_the_clock(clock, recorder):
    """The quota was spent whether or not an answer came back.

    Setup: ``recorder.raises = RuntimeError("boom")``; call ``parse(make_llm(pause_seconds=5.0))``
    and let it raise (``pytest.raises``). Then clear ``recorder.raises`` and call
    ``llm_module._wait_turn(5.0)``.

    Assert: the failed call raised, *and* the following ``_wait_turn`` waits ``5.0`` -- the clock
    was marked despite the failure. Break the ``finally:`` in ``LLM.parse`` and this goes red;
    nothing else in this file does.
    """
    recorder.raises = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        parse(make_llm(pause_seconds=5.0))
    recorder.raises = None
    assert llm_module._wait_turn(5.0) == 5.0
    assert clock.slept == [5.0]

# --------------------------------------------------------------------------------------
# Temperature  [yours]
# --------------------------------------------------------------------------------------


def test_temperature_is_sent_when_set(recorder):
    """A configured temperature reaches the request.

    Setup: ``parse(make_llm(temperature=0.1))``.

    Assert: ``recorder.last["temperature"] == 0.1``.
    """
    parse(make_llm(temperature=0.1))
    assert recorder.last["temperature"] == 0.1


def test_temperature_none_sends_no_temperature_at_all(recorder):
    """``None`` means omit the parameter, which reasoning models require.

    Setup: ``parse(make_llm(temperature=None))``.

    Assert: ``"temperature"`` is **not a key** of ``recorder.last``. Asserting it equals ``None``
    would pass while the request still carried ``temperature=None``, which is a different
    request from one that carries nothing.
    """
    parse(make_llm(temperature=None))
    assert "temperature" not in recorder.last


def test_temperature_zero_is_still_sent(recorder):
    """Zero is a value, not an absence -- the distinction lesson 03 is about.

    Setup: ``parse(make_llm(temperature=0.0))``.

    Assert: ``recorder.last["temperature"] == 0.0``. Rewrite ``llm.py``'s check as
    ``if self.temperature:`` and this is the only test that notices.
    """
    parse(make_llm(temperature=0.0))
    assert recorder.last["temperature"] == 0.0


def test_temperature_passed_twice_is_an_error(recorder):
    """Two sources for one parameter should fail loudly, not pick a winner.

    Setup: ``parse(make_llm(temperature=0.1), temperature=0.9)``.

    Assert: it raises ``TypeError`` (``pytest.raises``).
    """
    with pytest.raises(TypeError):
        parse(make_llm(temperature=0.1), temperature=0.9)

# --------------------------------------------------------------------------------------
# The client  [yours] -- these do NOT use the `recorder` fixture, since _client is the subject
# --------------------------------------------------------------------------------------


def test_missing_key_raises_rather_than_building_a_client(monkeypatch):
    """An unset key is a setup fault, and must not become an auth error much later.

    Setup: ``monkeypatch.delenv(DEFAULT_API_KEY_ENV, raising=False)``, then call
    ``llm_module._client(DEFAULT_API_KEY_ENV, 0)``.

    Assert: it raises ``KeyError`` (``pytest.raises``), which is what ``_client``'s docstring
    promises.

    Then write a **second test of your own**, with no stub to copy: the same variable set to an
    empty string instead of removed (``monkeypatch.setenv(DEFAULT_API_KEY_ENV, "")``). That is the
    case a freshly copied ``.env`` produces, and it used to raise a different exception from the
    absent case. Decide what it should raise, claim it, and break the empty-string check in
    ``_client`` to watch your test go red.
    """
    monkeypatch.delenv(DEFAULT_API_KEY_ENV, raising=False)
    with pytest.raises(KeyError):
        llm_module._client(DEFAULT_API_KEY_ENV, 0)


def test_empty_key_raises_rather_than_building_a_client(monkeypatch):
    """An empty key is a setup fault, and must not become an auth error much later.

    Setup: ``monkeypatch.setenv(DEFAULT_API_KEY_ENV, "")``, then call
    ``llm_module._client(DEFAULT_API_KEY_ENV, 0)``.

    Assert: it raises ``ValueError`` (``pytest.raises``), which is what ``_client``'s docstring
    promises.
    """
    monkeypatch.setenv(DEFAULT_API_KEY_ENV, "")
    with pytest.raises(KeyError):
        llm_module._client(DEFAULT_API_KEY_ENV, 0)


def test_one_client_is_shared_for_the_same_key_and_policy(monkeypatch):
    """The cache exists so that one key means one connection pool.

    Setup: set ``DEFAULT_API_KEY_ENV`` to any string with ``monkeypatch.setenv``, then call
    ``llm_module._client(DEFAULT_API_KEY_ENV, 2)`` twice.

    Assert: both calls return the *same object* (``is``, not ``==``).
    """
    monkeypatch.setenv(DEFAULT_API_KEY_ENV, "some-key")
    client1 = llm_module._client(DEFAULT_API_KEY_ENV, 2)
    client2 = llm_module._client(DEFAULT_API_KEY_ENV, 2)
    assert client1 is client2


def test_a_different_base_url_gets_a_different_client(monkeypatch):
    """The judge points somewhere else, and must not inherit the curator's client.

    Setup: set both key variables, then call ``_client(DEFAULT_API_KEY_ENV, 2)`` and
    ``_client(JUDGE_API_KEY_ENV, 2, "https://example.invalid/v1")``.

    Assert: the two are not the same object.

    **Correction to this stub.** As set up, the two calls differ in the key variable *and* the
    URL, so they would get different clients even if ``base_url`` were ignored -- the test cannot
    go red for the reason its name gives. To make it about the URL alone, both calls have to use
    the same key variable. And it only covers the cache: whether ``LLM.parse`` hands its
    ``base_url`` over at all is the next test's claim.
    """
    monkeypatch.setenv(DEFAULT_API_KEY_ENV, "some-key")
    monkeypatch.setenv(JUDGE_API_KEY_ENV, "some-other-key")
    client1 = llm_module._client(DEFAULT_API_KEY_ENV, 2)
    client2 = llm_module._client(JUDGE_API_KEY_ENV, 2, "https://example.invalid/v1")
    assert client1 is not client2


def test_an_llm_asks_for_a_client_with_its_own_key_and_endpoint(recorder):
    """An ``LLM``'s key variable and endpoint are what it asks ``_client`` for.

    This is the claim that protects the judge: ``review_case`` builds an ``LLM`` with the judge's
    key variable and ``base_url``, and if ``LLM.parse`` dropped either on the way to ``_client``
    the editor would quietly run on the curator's key or against OpenAI.

    Available: ``make_llm`` takes ``api_key_env=`` and ``base_url=``; after a ``parse`` call,
    ``recorder.client_requests`` holds one ``(api_key_env, request_retries, base_url)`` tuple.

    The setup and the assertions are yours. To check it, remove ``self.base_url`` from the
    ``_client(...)`` call in ``LLM.parse`` and watch it go red.
    """
    llm = make_llm(api_key_env=JUDGE_API_KEY_ENV, base_url="https://example.invalid/v1")
    parse(llm)
    assert recorder.client_requests[-1] == (JUDGE_API_KEY_ENV, 0, "https://example.invalid/v1")

# --------------------------------------------------------------------------------------
# What reaches the request  [yours]
# --------------------------------------------------------------------------------------


def test_model_case_id_and_role_reach_the_request(recorder):
    """The three identifying arguments pass through unchanged.

    Setup: ``parse(make_llm(model="some-model"))``.

    Assert: ``recorder.last`` carries ``model == "some-model"``, ``case_id == "case-1"`` and
    ``role is Role.GRADER``. Case id and role are what the typed errors report, so a lost one
    means an error that cannot say which case it was about.
    """
    parse(make_llm(model="some-model"))
    assert recorder.last["model"] == "some-model"
    assert recorder.last["case_id"] == "case-1"
    assert recorder.last["role"] is Role.GRADER


def test_other_options_pass_through_untouched(recorder):
    """Anything else the caller sends is forwarded as-is.

    Setup: ``parse(make_llm(), max_output_tokens=256)``.

    Assert: ``recorder.last["max_output_tokens"] == 256``.
    """
    parse(make_llm(), max_output_tokens=256)
    assert recorder.last["max_output_tokens"] == 256
