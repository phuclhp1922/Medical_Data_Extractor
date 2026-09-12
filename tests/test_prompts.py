"""Tests for the prompt modules: the RULES/render split and its fingerprints.

No corpus and no network. Every payload here is rendered from a sentinel string, so these are
pure and instant -- which is the point of the split: the part of the pipeline that decides what
the model is told can be tested without ever calling a model.

**Note on authorship.** The agent wrote all of ``curator/prompts/``, so it has not written the
assertions that pin it -- a test written by whoever wrote the code proves only that the code
does what the code does. The helpers and the skipped stubs below are the harness; filling in
the ``assert`` lines is yours. See ``teaching/learning-records/0001``.
"""

import pytest

from curator.prompts import editor, extractor, fingerprint, grader

TODO = "assertion is yours -- see the authorship note in this module's docstring"

# A token that cannot occur in real prompt text, so counting it is unambiguous.
SENTINEL = "ZZQQ_CASE_SENTINEL"

# Stand-ins for a draft teaching case. Distinct values, so a renderer that puts the wrong one in
# the wrong slot is visible rather than hidden behind six identical strings.
_DRAFT = (
    "DRAFT_THINK",
    "DRAFT_IMAGE_FINDING",
    "DRAFT_CASE_PROMPT",
    "DRAFT_REASONING_POINTS",
    "DRAFT_REASONING_NARRATIVE",
    "DRAFT_FINAL_DIAGNOSIS",
)


# --------------------------------------------------------------------------------------
# Harness  [agent]
# --------------------------------------------------------------------------------------


def _payload(module) -> str:
    """Render ``module``'s main prompt with ``SENTINEL`` standing in for the case text.

    The three modules take different arguments -- that is honest, they have different jobs -- so
    this hides the difference for the tests that care only about the shared structure.

    Args:
        module: One of ``grader``, ``extractor`` or ``editor``.

    Returns:
        str: The rendered payload.
    """
    if module is editor:
        return editor.render(SENTINEL, *_DRAFT)
    return module.render(SENTINEL)


# Used by @pytest.mark.parametrize below. The ``id=`` is what shows up in ``pytest -v`` output,
# so a failure names the module rather than "case 2".
PROMPT_MODULES = [
    pytest.param(grader, id="grader"),
    pytest.param(extractor, id="extractor"),
    pytest.param(editor, id="editor"),
]


# --------------------------------------------------------------------------------------
# The split holds  [yours] -- replace pytest.skip with the assertion in each docstring
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("module", PROMPT_MODULES)
def test_rules_are_non_empty(module):
    """Every module's ``RULES`` has content.

    Catches a split that cut at the wrong point and left the constant hollow -- 
    which would ship a prompt with no instructions in it.
    """
    assert module.RULES.strip(), f"{module.__name__}.RULES is empty" 

@pytest.mark.parametrize("module", PROMPT_MODULES)
def test_rules_hold_no_placeholder(module):
    """``RULES`` is a plain string, so it must contain no ``{`` placeholder.

    This is the half-finished-split test: a placeholder left in a non-f-string never
    interpolates, so the model receives the literal text ``{full_text}`` where the case should
    be. Nothing raises. Nothing logs. The run just silently means nothing.
    """
    assert "{" not in module.RULES


@pytest.mark.parametrize("module", PROMPT_MODULES)
def test_render_includes_its_rules(module):
    """The rendered payload still contains the spec it was built from.

    Catches a renderer that stopped concatenating ``RULES`` -- 
    the model would then be asked to do the job with no rules at all.
    """
    assert module.RULES in _payload(module), (
        f"{module.__name__}.RULES is missing from the rendered payload"
    )


@pytest.mark.parametrize("module", PROMPT_MODULES)
def test_invariant_text_precedes_the_case(module):
    """Constants come first in the payload, the per-case text last.

    This one is about money, not correctness. Providers cache on *prefix*: a payload that opens
    with ~4 KB of invariant rubric gets a cache hit on that rubric across all 93 cases. 
    """
    payload = _payload(module)
    assert payload.index(module.RULES) < payload.index(SENTINEL), (
        f"{module.__name__}.RULES does not precede the case text"
    )


# --------------------------------------------------------------------------------------
# The section 2.2 fix  [yours] -- the regression test for the empty-rulebook bug
# --------------------------------------------------------------------------------------


def test_editor_spec_contains_the_extractor_spec():
    """The editor audits against the extractor's actual rules.

    The notebook's editor prompt announced "here is the original guideline the draft was
    produced in accordance with" and interpolated whatever the caller passed -- and
    ``review_case`` passed ``""``. Every audit it ever ran judged compliance against an empty
    rulebook, and it never raised, never logged, never looked wrong. If anyone unplugs the
    splice again, this is the only thing in the project that would notice.
    """
    assert extractor.RULES in editor.RULES, "editor.RULES does not contain extractor.RULES"


def test_case_text_appears_once_in_the_editor_payload():
    """The editor sees the case report exactly once.

    This is why the editor composes ``extractor.RULES`` and not ``extractor.render(full_text)``:
    the latter would carry its own copy of the case, and the editor prompt already includes one
    further down. Two copies is not merely wasteful -- it is ambiguous about which one the audit
    refers to.
    """
    assert _payload(editor).count(SENTINEL) == 1, (
        "editor payload carries the case text more than once"
    )


def test_retry_shares_the_extractor_spec():
    """The retry pass obeys the same rules as the first pass.

    Pins the deduplication. The notebook had two copies of these rules -- 113 of 121 lines
    shared -- and they had already drifted apart: different bullet characters, and one copy was
    missing a line about image paths. One role now has one spec, and this is what keeps it that
    way.
    """
    assert extractor.RULES in extractor.render_retry(SENTINEL, *_DRAFT, "FLAGS", "COMMENTS"), (
        "extractor.RULES is missing from the retry payload"
    )


# --------------------------------------------------------------------------------------
# Fingerprints  [yours]
# --------------------------------------------------------------------------------------


def test_fingerprint_algorithm_is_pinned():
    """``fingerprint`` is SHA-256, UTF-8, truncated to 12 hex characters -- and stays that way.

    """
    assert fingerprint("hello") == "2cf24dba5fb0"


def test_fingerprint_moves_when_the_text_moves():
    """Any change to the text changes the fingerprint.

    Assert ``fingerprint(grader.RULES) != fingerprint(grader.RULES + " ")``. A single trailing
    space is the smallest edit possible, and it must be enough -- whitespace changes the bytes
    the model receives, so it has to change the recorded version too.
    """
    assert fingerprint(grader.RULES) != fingerprint(grader.RULES + " ")


def test_editor_version_tracks_the_extractor_spec():
    """Editing the extractor's rules changes the editor's recorded version.

    Not a tautology worth skipping: it documents a consequence that will surprise you later.
    Because the editor splices the extractor's spec, a commit that appears to touch only the
    extractor will move ``editor.RULES_VERSION`` too. That is correct -- a different extractor
    spec means a genuinely different audit -- but it is worth having written down.
    """
    assert fingerprint(editor.RULES) != fingerprint(editor.RULES.replace(extractor.RULES, ""))