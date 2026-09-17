"""Tests for ``curator.schemas``: what each role can say, and what it cannot.

No network and no corpus. Every test builds its input by hand.

**Note on authorship.** The helpers and parameter lists below are the agent's. Every ``assert``
and every ``pytest.raises`` block is the test author's -- see ``teaching/learning-records/0001``.
Each stub's docstring says what to check; replace its ``pytest.skip`` with that check.

Imports you will need that the harness does not use yet: ``from pydantic import
ValidationError`` for the Pydantic models, and ``Concern`` / ``EditorVerdict`` /
``RefusalError`` from ``curator.schemas``.
"""

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from curator.schemas import Concern, Draft, EditorVerdict, Grading, RefusalError, Role, StoppedBy

# --------------------------------------------------------------------------------------
# Harness  [agent]
# --------------------------------------------------------------------------------------


def _grading(**overrides) -> dict:
    """A complete, valid ``Grading`` payload, with any field replaced.

    To test a *missing* field, build this and ``del`` the key -- an override cannot remove one.

    Args:
        **overrides: Fields to replace.

    Returns:
        dict: The payload, ready for ``Grading.model_validate``.
    """
    grading = {
        "think": "reasoning",
        "is_case_report": True,
        "case_presentation_score": 4,
        "integrative_reasoning_score": 4,
        "transparency_score": 4,
        "images_usefulness_score": 4,
        "differential_diagnosis_present": True,
        "final_diagnosis_present": True,
    }
    grading.update(overrides)
    return grading


def _refusal_kwargs(**overrides) -> dict:
    """Every argument ``RefusalError`` needs except the two whose agreement is under test.

    Args:
        **overrides: Arguments to add or replace -- normally ``refusal_text`` and
            ``stopped_by``.

    Returns:
        dict: Keyword arguments for ``RefusalError(**...)``.
    """
    kwargs = {"role": Role.EDITOR, "message": "refused", "case_id": "case_28"}
    kwargs.update(overrides)
    return kwargs


def _strict_schema(model) -> dict:
    """The JSON schema the SDK actually sends for ``model`` under strict mode.

    ``to_strict_json_schema`` is private SDK API. Acceptable in a test and never in ``src/``: if
    an SDK upgrade moves or changes it, this test *should* break, because D11 was verified
    against exactly this function.
    """
    return to_strict_json_schema(model)


# The three models the LLM fills in. The error classes are not here: D11 does not govern them.
LLM_MODELS = [
    pytest.param(Grading, id="Grading"),
    pytest.param(Draft, id="Draft"),
    pytest.param(EditorVerdict, id="EditorVerdict"),
]

# D17: refusal_text is None exactly when stopped_by is PROVIDER.
CONTRADICTORY_REFUSALS = [
    pytest.param(StoppedBy.MODEL, None, id="model-without-text"),
    pytest.param(StoppedBy.PROVIDER, "I can't help with that.", id="provider-with-text"),
]
CONSISTENT_REFUSALS = [
    pytest.param(StoppedBy.MODEL, "I can't help with that.", id="model-with-text"),
    pytest.param(StoppedBy.PROVIDER, None, id="provider-without-text"),
]


# --------------------------------------------------------------------------------------
# Absence cannot enter  (D11, D12, learning record 0002)
# --------------------------------------------------------------------------------------


def test_grading_missing_a_field_is_rejected():
    """A ``Grading`` without one of its fields raises ``ValidationError``.

    T3's "done when", first half. Delete one key from ``_grading()`` -- a score is the case that
    matters, since the notebook read a missing score as ``0`` -- and check ``model_validate``
    raises instead of producing a ``Grading`` that looks fine.
    """
    grading = _grading()
    del grading["case_presentation_score"]
    with pytest.raises(ValidationError):
        Grading.model_validate(grading)


def test_out_of_vocabulary_section_is_rejected():
    """A ``Concern`` whose section is not a ``ChecklistSection`` raises ``ValidationError``.

    T3's "done when", second half. Use a name from the notebook's old worked example, such as
    ``"REASONING_EXTRA_INFO"`` -- exactly the drift PLAN section 2.5 describes.
    """
    concern = {"section": "SOURCE_FIDELITY", "explanation": "This is a concern."}
    concern["section"] = "REASONING_EXTRA_INFO"  # not in ChecklistSection
    with pytest.raises(ValidationError):
        Concern.model_validate(concern)


def test_null_concern_list_is_rejected():
    """``EditorVerdict`` with ``concerns=None`` raises ``ValidationError``.

    D12: "clean" has one spelling. If this ever passes validation, ``null`` has become a way for
    a non-answer to read as an approval.
    """
    verdict = {"concerns": None}
    with pytest.raises(ValidationError):
        EditorVerdict.model_validate(verdict)


def test_empty_concern_list_is_clean():
    """``EditorVerdict`` with ``concerns=[]`` validates, and ``not verdict.concerns`` is true.

    The other half of the previous test. Rejecting ``null`` is only safe if the clean case is
    still representable.
    """
    verdict = EditorVerdict.model_validate({"concerns": []})
    assert verdict.concerns == []


# --------------------------------------------------------------------------------------
# RefusalError's invariant  (D17)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("stopped_by", "refusal_text"), CONTRADICTORY_REFUSALS)
def test_contradictory_refusal_is_unconstructible(stopped_by, refusal_text):
    """``RefusalError`` raises ``ValueError`` when ``refusal_text`` and ``stopped_by`` disagree.
    """
    with pytest.raises(ValueError):
        RefusalError(**_refusal_kwargs(stopped_by=stopped_by, refusal_text=refusal_text))


@pytest.mark.parametrize(("stopped_by", "refusal_text"), CONSISTENT_REFUSALS)
def test_consistent_refusal_builds(stopped_by, refusal_text):
    """``RefusalError`` builds when the two agree, and keeps both values.

    Without this, a check that raised on *every* combination would pass the previous test.
    """
    error = RefusalError(**_refusal_kwargs(stopped_by=stopped_by, refusal_text=refusal_text))
    assert error.stopped_by == stopped_by
    assert error.refusal_text == refusal_text

# --------------------------------------------------------------------------------------
# What the SDK sends  (D11)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", LLM_MODELS)
def test_strict_schema_requires_every_field(model):
    """Under strict mode, every field of ``model`` is in the schema's ``required`` list.

    D11, re-checked on every run instead of once by hand. Compare ``_strict_schema(model)
    ["required"]`` against ``model.model_fields`` -- as sets, since order is not the point here.
    If an SDK upgrade stops forcing every field required, this is what notices.
    """
    schema = _strict_schema(model)
    required = set(schema["required"])
    fields = set(model.model_fields)
    assert required == fields