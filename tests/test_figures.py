"""Tests for the deterministic figure resolver.

The corpus is not redistributable, so these are marked ``corpus`` and skip when it is
absent. The numbers below were measured against the 93-case MinerU corpus; treat any
change as a regression to investigate, not a threshold to relax.
"""

import pytest

from curator.figures import (
    CorpusError,
    _caption_label,
    agreement_report,
    chapter_of,
    corpus_root,
    resolve_case,
    resolve_corpus,
)

CORPUS = corpus_root()
needs_corpus = pytest.mark.skipif(not CORPUS.is_dir(), reason=f"corpus not found at {CORPUS}")


# --------------------------------------------------------------------------------------
# Pure logic -- no corpus required
# --------------------------------------------------------------------------------------


def test_chapter_parsed_from_directory_name():
    assert chapter_of("24---A-14-Year-Old-Boy-from-Tanzania") == 24


def test_chapter_requires_leading_number():
    with pytest.raises(CorpusError):
        chapter_of("some-untagged-directory")


def test_caption_label_accepted_when_it_matches():
    label = _caption_label("Fig. 24.1 Spastic paraparesis.", chapter=24, image_count=2)
    assert label == "Fig. 24.1"


def test_chapter_guard_rejects_cross_reference():
    """Case 22 credits another book's figure: '(Reproduced from ... Fig. 43.1.)'."""
    caption = "Global distribution of malaria. (Reproduced from Farrar, J. et al., Fig. 43.1.)"
    assert _caption_label(caption, chapter=22, image_count=1) is None


def test_count_guard_rejects_ocr_splice():
    """Case 24: the label was spliced into prose, so a greedy regex reads '24.12'.

    The true text is 'started about [Fig. 24.1] 2 years earlier'. The case holds one image,
    so figure 12 cannot exist.
    """
    caption = "spastic paraparesis. His illness started about Fig. 24.12 years earlier"
    assert _caption_label(caption, chapter=24, image_count=1) is None
    # With enough images the same number would be legitimate, so the guard is a bound, not a ban.
    assert _caption_label(caption, chapter=24, image_count=20) == "Fig. 24.12"


def test_missing_label_is_not_an_error():
    assert _caption_label("An unlabelled photograph.", chapter=5, image_count=3) is None


# --------------------------------------------------------------------------------------
# Against the real corpus
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resolved():
    return resolve_corpus()


@needs_corpus
def test_corpus_shape(resolved):
    report = agreement_report(resolved)
    assert report["cases"] == 93
    assert report["figures"] == 142, "219 .jpg files exist but only 142 are real figures"


@needs_corpus
def test_every_figure_gets_a_label(resolved):
    """No unlabelled image may reach the model -- the prompt asks it to cite 'Fig N.M'."""
    for case, figures in resolved.items():
        for figure in figures:
            assert figure.label.startswith("Fig. "), f"{case}: unlabelled figure"


@needs_corpus
def test_two_methods_never_disagree(resolved):
    """The headline QA metric. Any disagreement is an anomaly to inspect by hand."""
    clashes = [
        (case, f.label, f.caption_label)
        for case, figures in resolved.items()
        for f in figures
        if f.disagrees
    ]
    assert clashes == []


@needs_corpus
def test_caption_independently_confirms_most_figures(resolved):
    """116/142 confirmed by the caption path, which is independent of position."""
    report = agreement_report(resolved)
    assert report["caption_confirmed"] >= 116


@needs_corpus
def test_every_resolved_image_exists_on_disk(resolved):
    missing = [
        str(f.image_path)
        for figures in resolved.values()
        for f in figures
        if not f.image_path.is_file()
    ]
    assert missing == []


@needs_corpus
def test_labels_are_unique_within_a_case(resolved):
    for case, figures in resolved.items():
        labels = [f.label for f in figures]
        assert len(labels) == len(set(labels)), f"{case}: duplicate labels {labels}"


@needs_corpus
def test_case_22_falls_back_to_positional():
    """Its only caption cites Fig. 43.1; the chapter guard rejects it and position wins."""
    case_dir = next(CORPUS.glob("22---*"))
    figures = resolve_case(case_dir)
    assert len(figures) == 1
    assert figures[0].label == "Fig. 22.1"
    assert figures[0].caption_label is None
    assert "Fig. 43.1" in figures[0].caption


@needs_corpus
def test_case_24_survives_the_ocr_splice():
    case_dir = next(CORPUS.glob("24---*"))
    figures = resolve_case(case_dir)
    assert figures[0].label == "Fig. 24.1"
    assert figures[0].caption_label is None


@needs_corpus
def test_pruned_corpus_raises_a_useful_error(tmp_path):
    """A corpus pruned to .md + images/ cannot be resolved -- say so clearly."""
    pruned = tmp_path / "24---pruned-case"
    (pruned / "auto" / "images").mkdir(parents=True)
    with pytest.raises(CorpusError, match="_content_list.json"):
        resolve_case(pruned)
