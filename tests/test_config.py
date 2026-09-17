"""Tests for the runtime configuration object.

No corpus and no network: ``Configuration.from_env`` accepts an ``env`` mapping, so these
are pure and instant.

**Note on authorship.** The agent wrote ``from_env`` and ``corpus_dir_from_env``, so it has
not written the assertions that pin them -- a test written by whoever wrote the code proves
only that the code does what the code does. The fixtures and the skipped stubs below are the
harness; filling in the ``assert`` lines is yours. See ``teaching/learning-records/0001``.
"""

from pathlib import Path

import pytest

from curator import figures
from curator.config import (
    CORPUS_DIR_ENV,
    CURATOR_MODEL_ENV,
    DEFAULT_CORPUS_DIR,
    DEFAULT_CURATOR_MODEL,
    DEFAULT_MAX_REFINES,
    MAX_REFINES_ENV,
    Configuration,
    GradeThresholds,
    corpus_dir_from_env,
)
from curator.schemas import Grading

TODO = "assertion is yours -- see the authorship note in this module's docstring"


# --------------------------------------------------------------------------------------
# Fixtures  [agent]
# --------------------------------------------------------------------------------------


@pytest.fixture
def empty_env() -> dict[str, str]:
    """An environment with nothing set."""
    return {}


@pytest.fixture
def blank_env() -> dict[str, str]:
    """An environment where every variable is present but empty.

    This is what a freshly copied ``.env.example`` actually produces, and it is the case a
    naive ``os.environ.get(name, default)`` gets wrong -- it returns ``""``, not the default.
    """
    return {CORPUS_DIR_ENV: "", CURATOR_MODEL_ENV: "", MAX_REFINES_ENV: ""}


@pytest.fixture
def full_env() -> dict[str, str]:
    """An environment with every variable overridden to a non-default value."""
    return {
        CORPUS_DIR_ENV: "/tmp/some-other-corpus",
        CURATOR_MODEL_ENV: "some-other-model",
        MAX_REFINES_ENV: "7",
    }


@pytest.fixture
def clean_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every curator variable from the real ``os.environ``.

    Needed only by tests that exercise the no-argument path (``corpus_root()`` takes no
    ``env``), because a developer machine will usually have ``CURATOR_CORPUS_DIR`` set.
    """
    for name in (CORPUS_DIR_ENV, CURATOR_MODEL_ENV, MAX_REFINES_ENV):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------------------
# Stubs  [yours] -- replace pytest.skip with the assertion described in each docstring
# --------------------------------------------------------------------------------------


def test_defaults_when_env_is_empty(empty_env):
    """``from_env`` fills in every default when nothing is set.

    Assert: ``curator_model`` is ``DEFAULT_CURATOR_MODEL``, ``max_refines`` is
    ``DEFAULT_MAX_REFINES``, ``corpus_dir`` is ``DEFAULT_CORPUS_DIR``.
    """
    cfg = Configuration.from_env(empty_env)

    assert cfg.curator_model == DEFAULT_CURATOR_MODEL
    assert cfg.max_refines == DEFAULT_MAX_REFINES
    assert cfg.corpus_dir == DEFAULT_CORPUS_DIR

def test_blank_values_are_treated_as_absent(blank_env):
    """A present-but-empty variable must fall back to the default, not to ``""``.

    Assert the same three defaults as above. This is the test that would have caught a
    ``.env`` copied from ``.env.example`` silently configuring an empty model id.
    """
    cfg = Configuration.from_env(blank_env)

    assert cfg.curator_model == DEFAULT_CURATOR_MODEL
    assert cfg.max_refines == DEFAULT_MAX_REFINES
    assert cfg.corpus_dir == DEFAULT_CORPUS_DIR


def test_env_overrides_defaults(full_env):
    """Values present in the environment win.

    Assert ``max_refines == 7`` -- note it must be an ``int``, not the string ``"7"``.
    """
    cfg = Configuration.from_env(full_env)

    assert cfg.curator_model == "some-other-model"
    assert cfg.max_refines == 7
    assert cfg.corpus_dir == Path("/tmp/some-other-corpus")


def test_corpus_root_agrees_with_config(clean_os_environ):
    """``figures.corpus_root()`` and the config must not drift.

    Assert ``corpus_root() == corpus_dir_from_env()``, and that both equal
    ``DEFAULT_CORPUS_DIR`` under a cleaned environment. Tautological by construction -- which
    is the point: it fails loudly if someone reintroduces a second default.
    """
    assert figures.corpus_root() == DEFAULT_CORPUS_DIR
    assert corpus_dir_from_env(clean_os_environ) == DEFAULT_CORPUS_DIR


def test_corpus_dir_is_a_path(empty_env):
    """``corpus_dir`` must be a ``Path``, not a ``str``.

    Assert ``isinstance(Configuration.from_env(empty_env).corpus_dir, Path)``. Without this
    the agreement test above passes or fails depending on string formatting.
    """
    cfg = Configuration.from_env(empty_env)
    assert isinstance(cfg.corpus_dir, Path)


# --------------------------------------------------------------------------------------
# The gate  [yours] -- the truth table for GradeThresholds.passes()
# --------------------------------------------------------------------------------------


def _grading(**overrides) -> Grading:
    """Build a ``Grading`` that passes every gate, then apply ``overrides``.

    Args:
        **overrides: Fields to replace in the all-passing baseline.

    Returns:
        Grading: A validated grading.
    """
    grading = {
        "think": "baseline reasoning",
        "is_case_report": True,
        "case_presentation_score": 4,
        "integrative_reasoning_score": 4,
        "transparency_score": 4,
        "images_usefulness_score": 4,
        "differential_diagnosis_present": True,
        "final_diagnosis_present": True,
    }
    grading.update(overrides)
    return Grading.model_validate(grading)


def test_gate_accepts_a_passing_grading():
    """All six gates satisfied -> True."""
    grading = _grading()
    assert GradeThresholds().passes(grading) is True


def test_gate_rejects_a_score_below_threshold():
    """One int gate below its threshold -> False.
    """
    grading = _grading(transparency_score=2)
    assert GradeThresholds().passes(grading) is False

    grading = _grading(case_presentation_score=2)
    assert GradeThresholds().passes(grading) is False

    grading = _grading(images_usefulness_score=2)
    assert GradeThresholds().passes(grading) is False

    grading = _grading(integrative_reasoning_score=2)
    assert GradeThresholds().passes(grading) is False


def test_gate_rejects_a_no():
    """One True/False gate answered False -> False.
    """
    grading = _grading(differential_diagnosis_present=False)
    assert GradeThresholds().passes(grading) is False

    grading = _grading(final_diagnosis_present=False)
    assert GradeThresholds().passes(grading) is False


def test_thresholds_are_actually_read():
    """Raising a threshold must reject a grading that passed at the old one.

    Assert that a grading scoring 3 passes ``GradeThresholds()`` but fails
    ``GradeThresholds(case_presentation_score=4, ...)``. **This is the important one** --
    it is the test that fails if ``passes()`` hardcodes ``>= 3`` instead of comparing
    against ``self``, which is the whole reason T1 exists.
    """
    grading = _grading(case_presentation_score=3)
    assert GradeThresholds().passes(grading) is True
    assert GradeThresholds(case_presentation_score=4).passes(grading) is False


def test_gate_rejects_a_non_case_report_even_with_perfect_scores():
    """``is_case_report=False`` -> False, whatever the scores say.

    D13's guard. Strict mode forces the grader to emit four scores even for an article with no
    patient in it, and those numbers mean nothing. Build ``_grading(is_case_report=False)`` with
    every score at 5 -- the case most likely to slip through -- and assert ``passes`` is False.
    Without the guard, this grading clears every threshold and goes on to the extractor.
    """
    grading = _grading(
        is_case_report=False,
        case_presentation_score=5,
        integrative_reasoning_score=5,
    )
    assert GradeThresholds().passes(grading) is False
