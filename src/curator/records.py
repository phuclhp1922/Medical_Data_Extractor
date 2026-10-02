"""What a finished run leaves behind: one JSON record on disk, and a readable rendering of it.

The graph's final state is a working object -- full of Pydantic models, ``Path``s and enums, and
holding the case's source text. A record is the opposite: plain JSON, written once, never edited,
readable without importing this package. Two decisions shape it.

**The source text is left out.** It is copyrighted (the corpus derives from an Elsevier
textbook), it is large, and it is recoverable: ``source`` names the case directory, and the
corpus is the authority on what that directory contains. The drafts and verdicts *are* kept, and
they quote the source -- which is why ``runs/`` is gitignored and must stay that way.

**Everything needed to explain the outcome is kept.** The configuration (model ids,
temperatures, thresholds, the refine cap), and the fingerprint of every prompt that ran. A record
that says ``hit_max_refines`` without saying the cap was 3 and which rubric was in force cannot
be compared with any other record; Phase 2 compares nothing else.

This is the first slice of T7, not all of it. Batch running, per-case error isolation and resume
are still owed -- a record is only written for a run that finished.
"""

import dataclasses
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any

from curator.config import Configuration
from curator.prompts import editor, extractor, grader

#: Bumped whenever a field is renamed, removed or changes meaning, so a reader can refuse a record
#: it was not written for instead of misreading it.
RECORD_SCHEMA_VERSION = 1

#: Where records go unless told otherwise. Gitignored -- the records quote the source.
DEFAULT_RUNS_DIR = Path("runs")


def _prompt_versions() -> dict[str, str]:
    """The fingerprint of every prompt a run can send, as of import time."""
    return {
        "grader": grader.RULES_VERSION,
        "editor": editor.RULES_VERSION,
        "extractor_rules": extractor.RULES_VERSION,
        "extractor_first_try": extractor.FIRST_TRY_VERSION,
        "extractor_retry": extractor.RETRY_VERSION,
    }


def _jsonable(value: Any) -> Any:
    """``json.dumps`` fallback for the few non-JSON types in a configuration."""
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def build_record(
    out: Mapping[str, Any],
    cfg: Configuration,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    """Turn a finished graph state into a plain, self-describing record.

    Args:
        out (Mapping[str, Any]): The state ``build_graph().invoke(...)`` returned.
        cfg (Configuration): The configuration the run used.
        started_at (datetime): When the run began, timezone-aware.
        finished_at (datetime): When it ended, timezone-aware.

    Returns:
        dict[str, Any]: JSON-ready. Keys are stable within ``RECORD_SCHEMA_VERSION``.
    """
    grading = out["grading"]
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "source": out["source"],
        "terminal_state": out["terminal_state"].value,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 1),
        # A snapshot, so a reader knows which models, sampling and thresholds produced this.
        "config": json.loads(json.dumps(dataclasses.asdict(cfg), default=_jsonable)),
        "prompt_versions": _prompt_versions(),
        # The figure files are named, not embedded: the corpus holds the pixels.
        "figures": [
            {
                "label": figure.label,
                "file": figure.image_path.name,
                "caption": figure.caption,
                "caption_label": figure.caption_label,
            }
            for figure in out["figures"]
        ],
        "grading": grading.model_dump(mode="json"),
        # Derivable from the two fields above it. Stored anyway because a record is a log, not
        # state: nothing reads it back into the graph, so it cannot drift from anything, and a
        # funnel count should not have to re-import the thresholds to know the gate's answer.
        "passes_thresholds": cfg.thresholds.passes(grading),
        "drafts": [draft.model_dump(mode="json") for draft in out["drafts"]],
        "verdicts": [verdict.model_dump(mode="json") for verdict in out["verdicts"]],
    }


def write_record(record: Mapping[str, Any], runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Write one record to ``<runs_dir>/<source>/<UTC timestamp>.json``.

    One directory per case, one file per run, so reruns of a case sit side by side and nothing is
    overwritten. The timestamp is the run's start, in a form that sorts correctly and contains no
    character Windows forbids in a filename.

    Args:
        record (Mapping[str, Any]): What ``build_record`` returned.
        runs_dir (Path): The root of all records.

    Returns:
        Path: The file written.

    Raises:
        FileExistsError: A record with the same case and start time already exists. Opened in
            exclusive mode on purpose: a record is written once, and silently replacing one is
            how a rerun erases the evidence of the run before it.
    """
    started = datetime.fromisoformat(record["started_at"]).astimezone(UTC)
    path = runs_dir / record["source"] / f"{started:%Y%m%dT%H%M%SZ}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


#: The draft fields a reader sees, in reading order. ``think`` is left out on purpose: it is the
#: extractor's working notes, not part of the teaching case. It stays in the record.
_DRAFT_SECTIONS = [
    ("CASE PROMPT (what the student sees)", "case_prompt"),
    ("IMAGE FINDINGS", "image_finding"),
    ("REASONING POINTS", "reasoning_points"),
    ("REASONING NARRATIVE", "reasoning_narrative"),
    ("FINAL DIAGNOSIS", "final_diagnosis"),
]


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line.strip() else line for line in text.strip().splitlines())


def format_record(record: Mapping[str, Any]) -> str:
    """Render a record as the teaching case a person would read.

    Reads the *record*, not the graph state, so it works on any file in ``runs/`` as well as on
    a run that just finished -- and so what is shown is exactly what was saved.

    The extractor's ``think`` field is left out: it is the model's working notes, not part of
    the teaching case. It stays in the record.

    Args:
        record (Mapping[str, Any]): What ``build_record`` returned, or a record read back from disk.

    Returns:
        str: Plain text, sectioned, ready to print.
    """
    grading = record["grading"]
    drafts = record["drafts"]
    verdicts = record["verdicts"]
    rule = "-" * 78

    lines = [
        rule,
        f"CASE     {record['source']}",
        f"OUTCOME  {record['terminal_state']}  "
        f"({len(drafts)} extract pass(es), {len(verdicts)} review(s), "
        f"{record['duration_seconds']}s)",
        f"MODELS   curator {record['config']['curator_model']}  |  "
        f"editor {record['config']['judge_model']}",
        rule,
        "",
        "GRADING OF THE SOURCE REPORT",
        f"  case report: {_yes_no(grading['is_case_report'])}   "
        f"presentation {grading['case_presentation_score']}   "
        f"reasoning {grading['integrative_reasoning_score']}   "
        f"transparency {grading['transparency_score']}   "
        f"images {grading['images_usefulness_score']}",
        f"  differential stated: {_yes_no(grading['differential_diagnosis_present'])}   "
        f"final diagnosis stated: {_yes_no(grading['final_diagnosis_present'])}   "
        f"-> {'passes' if record['passes_thresholds'] else 'fails'} the thresholds",
    ]

    if not drafts:
        lines += ["", f"No teaching case: the run stopped at grading ({record['terminal_state']})."]
        return "\n".join(lines) + "\n"

    lines += [
        "",
        "EDITOR'S CONCERNS, PASS BY PASS",
        "  " + " -> ".join(str(len(verdict["concerns"])) for verdict in verdicts),
    ]

    # Each extract pass is followed by exactly one review, so drafts[i] is the draft verdicts[i]
    # judged. zip_longest rather than zip: a run that stopped between the two (a crash in the
    # editor, once T7 records failed runs) shows its unreviewed last draft instead of hiding it.
    for number, (draft, verdict) in enumerate(zip_longest(drafts, verdicts), start=1):
        heading = f"PASS {number} OF {len(drafts)}"
        if number == len(drafts):
            heading += "  (final draft)"
        lines += ["", rule, heading, rule]

        for section, field in _DRAFT_SECTIONS:
            lines += ["", section, _indent(draft[field])]

        lines += ["", f"EDITOR'S VERDICT ON PASS {number}"]
        if verdict is None:
            lines.append("  Not reviewed.")
        elif not verdict["concerns"]:
            lines.append("  No concerns.")
        else:
            for concern in verdict["concerns"]:
                lines.append(f"  [{concern['section']}] {concern['explanation']}")

    return "\n".join(lines) + "\n"
