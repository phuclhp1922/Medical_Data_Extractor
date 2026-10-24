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

**A crash leaves a record too** (T7, D29). A case whose run raised gets an *error record*: the
same snapshot, and the exception in place of an outcome. Without one, a crashed case would vanish
from every count -- the notebook's ``dropna()`` again -- and the funnel's partition would not
sum to the number of cases attempted.

**Two layouts.** A batch (``curator.run``) writes ``runs/batch-<time>/<source>.json``, one folder
per batch, so the funnel reads one folder and resume asks which sources it already holds. A
single case run from ``scripts/try_case.py`` writes ``runs/<source>/<time>.json``, as before.
"""

import dataclasses
import json
import subprocess
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from itertools import zip_longest
from pathlib import Path
from typing import Any

from curator.config import Configuration
from curator.prompts import editor, extractor, grader

#: Bumped whenever a field is renamed, removed or changes meaning, so a reader can refuse a record
#: it was not written for instead of misreading it. 2: adds ``batch``, ``git`` and ``usage``,
#: and error records.
RECORD_SCHEMA_VERSION = 2

#: Where records go unless told otherwise. Gitignored -- the records quote the source.
DEFAULT_RUNS_DIR = Path("runs")

#: The batch's own description, written once when the batch folder is created.
MANIFEST_NAME = "batch.json"

#: Configuration fields that change how fast a batch runs but not what it produces. A resumed
#: batch may differ in these; in anything else it may not (see ``check_manifest``).
_OPERATIONAL_FIELDS = ("pause_seconds", "request_retries")

#: The repository this module lives in, so the git commit is the code's, whatever the cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _config_snapshot(cfg: Configuration) -> dict[str, Any]:
    """The configuration as plain JSON values."""
    return json.loads(json.dumps(dataclasses.asdict(cfg), default=_jsonable))


def _git_revision() -> dict[str, Any]:
    """The commit the code was at, and whether tracked files had uncommitted changes.

    ``dirty`` matters because a commit alone does not say what ran: an edited prompt that was
    not yet committed ran under the previous commit's id. The prompt fingerprints settle the
    prompts either way; ``dirty`` says whether anything *else* may differ from the commit.

    Returns:
        dict[str, Any]: ``commit`` and ``dirty``, both ``None`` if git is unavailable -- a
        record is still worth writing without them.
    """
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _timing(started_at: datetime, finished_at: datetime) -> dict[str, Any]:
    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 1),
    }


def build_record(
    out: Mapping[str, Any],
    cfg: Configuration,
    *,
    started_at: datetime,
    finished_at: datetime,
    batch: str | None = None,
) -> dict[str, Any]:
    """Turn a finished graph state into a plain, self-describing record.

    Args:
        out (Mapping[str, Any]): The state ``build_graph().invoke(...)`` returned.
        cfg (Configuration): The configuration the run used.
        started_at (datetime): When the run began, timezone-aware.
        finished_at (datetime): When it ended, timezone-aware.
        batch (str | None): The batch folder's name, or ``None`` for a single-case run.

    Returns:
        dict[str, Any]: JSON-ready. Keys are stable within ``RECORD_SCHEMA_VERSION``.
    """
    grading = out["grading"]
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "source": out["source"],
        "batch": batch,
        "terminal_state": out["terminal_state"].value,
        **_timing(started_at, finished_at),
        # A snapshot, so a reader knows which models, sampling and thresholds produced this.
        "config": _config_snapshot(cfg),
        "prompt_versions": _prompt_versions(),
        "git": _git_revision(),
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
        # One entry per successful model call, in call order (D29). Totals are not stored:
        # they are a sum over this list, and a stored sum could disagree with it.
        "usage": [call.model_dump(mode="json") for call in out["usage"]],
    }


def _error_details(error: BaseException) -> dict[str, Any]:
    """Everything an exception can tell a reader about why a case failed.

    The typed errors (``curator.schemas.CurationError``) carry the role that was calling and,
    depending on the kind, the refusal text, who stopped the output, or the raw output that did
    not parse. Those are kept when present. The traceback is kept for every error: an untyped
    one has nothing else to say where it came from.
    """
    details: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    for name in ("role", "stopped_by", "refusal_text", "raw_output"):
        if hasattr(error, name):
            value = getattr(error, name)
            details[name] = value.value if isinstance(value, Enum) else value
    details["traceback"] = "".join(traceback.format_exception(error))
    return details


def build_error_record(
    source: str,
    error: BaseException,
    cfg: Configuration,
    *,
    started_at: datetime,
    finished_at: datetime,
    batch: str | None = None,
) -> dict[str, Any]:
    """Describe a case whose run raised instead of finishing.

    It has an ``error`` and **no** ``terminal_state``: a crashed case never reached ``finalize``,
    so it did not leave the graph by any of its exits, and the funnel counts it in a bucket of
    its own (D29). Everything that explains the circumstances -- configuration, prompts, commit
    -- is the same as on a finished record.

    What was produced before the crash (a grading, early drafts) is not kept: the graph raises
    without returning its state. Nor is the token usage of the calls that did succeed.

    Args:
        source (str): The case directory name.
        error (BaseException): What the run raised.
        cfg (Configuration): The configuration the run used.
        started_at (datetime): When the run began, timezone-aware.
        finished_at (datetime): When it raised, timezone-aware.
        batch (str | None): The batch folder's name, or ``None`` outside a batch.

    Returns:
        dict[str, Any]: JSON-ready.
    """
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "source": source,
        "batch": batch,
        "error": _error_details(error),
        **_timing(started_at, finished_at),
        "config": _config_snapshot(cfg),
        "prompt_versions": _prompt_versions(),
        "git": _git_revision(),
    }


def _write_once(record: Mapping[str, Any], path: Path) -> Path:
    """Write ``record`` to ``path``, refusing to replace a file that is already there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _stamp(iso: str) -> str:
    """An ISO time as ``20261023T020209Z``: sorts correctly, and no character Windows forbids."""
    return f"{datetime.fromisoformat(iso).astimezone(UTC):%Y%m%dT%H%M%SZ}"


def write_batch_record(record: Mapping[str, Any], batch_dir: Path) -> Path:
    """Write a finished or error record into a batch folder.

    A finished case is ``<source>.json``: one per source, and its presence is what resume reads
    as "done". An error is ``<source>.error.<start time>.json``, so retrying a failed case on
    resume adds a file beside the old one instead of replacing it -- every attempt stays on
    disk.

    Args:
        record (Mapping[str, Any]): From ``build_record`` or ``build_error_record``.
        batch_dir (Path): The batch folder.

    Returns:
        Path: The file written.

    Raises:
        FileExistsError: That file already exists. Written once, never replaced.
    """
    if "error" in record:
        name = f"{record['source']}.error.{_stamp(record['started_at'])}.json"
    else:
        name = f"{record['source']}.json"
    return _write_once(record, batch_dir / name)


def write_manifest(
    batch_dir: Path,
    cfg: Configuration,
    *,
    sources: list[str],
    created_at: datetime,
) -> Path:
    """Create a batch folder and describe the batch in ``batch.json``.

    The list of sources is fixed here, at creation. It is the batch's population: what a resume
    runs, and the denominator a funnel is read against. Letting a resume choose its own cases
    would let a ten-case batch grow into a ninety-three-case one without anyone deciding to.

    Args:
        batch_dir (Path): The folder to create. Must not exist yet.
        cfg (Configuration): The configuration every case in the batch will run under.
        sources (list[str]): The case directory names the batch is for, in run order.
        created_at (datetime): When the batch was started, timezone-aware.

    Returns:
        Path: The manifest written.

    Raises:
        FileExistsError: The folder already exists -- that is a resume, not a new batch.
    """
    batch_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "batch": batch_dir.name,
        "created_at": created_at.isoformat(),
        "sources": list(sources),
        "config": _config_snapshot(cfg),
        "prompt_versions": _prompt_versions(),
        "git": _git_revision(),
    }
    return _write_once(manifest, batch_dir / MANIFEST_NAME)


def read_manifest(batch_dir: Path) -> dict[str, Any]:
    """The batch's ``batch.json``, as written by ``write_manifest``."""
    return json.loads((batch_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def check_manifest(batch_dir: Path, cfg: Configuration) -> list[str]:
    """List how the current settings differ from the ones a batch was started with.

    A funnel is only evidence if every case in it ran under one rubric (D29), so resuming a batch
    with different prompts or a different configuration would quietly mix two experiments in
    one count. Pacing and retry settings are exempt: they change how fast a batch runs, not what
    it produces. The git commit is exempt too -- code may move without changing a prompt, and the
    fingerprints are what say whether a prompt did.

    Args:
        batch_dir (Path): An existing batch folder.
        cfg (Configuration): The configuration a resume would use.

    Returns:
        list[str]: One line per difference, empty if the resume is safe.
    """
    manifest = read_manifest(batch_dir)
    differences = []
    now_config = _config_snapshot(cfg)
    for key in sorted(set(manifest["config"]) | set(now_config)):
        if key in _OPERATIONAL_FIELDS:
            continue
        then, now = manifest["config"].get(key), now_config.get(key)
        if then != now:
            differences.append(f"config.{key}: batch has {then!r}, now {now!r}")
    now_prompts = _prompt_versions()
    for key in sorted(set(manifest["prompt_versions"]) | set(now_prompts)):
        then, now = manifest["prompt_versions"].get(key), now_prompts.get(key)
        if then != now:
            differences.append(f"prompt {key}: batch has {then}, now {now}")
    return differences


def finished_sources(batch_dir: Path) -> set[str]:
    """The sources that already have a finished record in this batch."""
    return {
        path.stem
        for path in batch_dir.glob("*.json")
        if path.name != MANIFEST_NAME and ".error." not in path.name
    }


def load_batch(batch_dir: Path) -> list[dict[str, Any]]:
    """Read a batch back as one record per source attempted.

    A source with a finished record is represented by it. A source with only error records is
    represented by its latest one -- earlier failures of a case that was retried are history,
    not separate cases. This is the list the funnel counts, so each attempted source appears
    exactly once.

    Args:
        batch_dir (Path): A batch folder.

    Returns:
        list[dict[str, Any]]: One record per source, ordered by source name.
    """
    finished: dict[str, dict[str, Any]] = {}
    latest_error: dict[str, dict[str, Any]] = {}
    for path in sorted(batch_dir.glob("*.json")):
        if path.name == MANIFEST_NAME:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "error" in record:
            # Sorted by name, and the name ends in the start time, so the last one read wins.
            latest_error[record["source"]] = record
        else:
            finished[record["source"]] = record
    chosen = {**latest_error, **finished}
    return [chosen[source] for source in sorted(chosen)]


def write_record(record: Mapping[str, Any], runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Write one single-case record to ``<runs_dir>/<source>/<UTC timestamp>.json``.

    The layout ``scripts/try_case.py`` uses; a batch uses ``write_batch_record``. One directory
    per case, one file per run, so reruns of a case sit side by side and nothing is overwritten.
    The timestamp is the run's start.

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
    return _write_once(record, runs_dir / record["source"] / f"{_stamp(record['started_at'])}.json")


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


def summarise_usage(usage: list[Mapping[str, Any]]) -> str:
    """One line of token totals over a record's ``usage`` list.

    A count the provider did not report (``None``) is left out of the sums, and the line says so,
    rather than being treated as zero.
    """
    def total(field: str) -> tuple[int, bool]:
        values = [call[field] for call in usage]
        return sum(v for v in values if v is not None), any(v is None for v in values)

    tokens_in, missing_in = total("input_tokens")
    cached, missing_cached = total("cached_input_tokens")
    tokens_out, missing_out = total("output_tokens")
    reasoning, missing_reasoning = total("reasoning_tokens")
    line = (
        f"{len(usage)} call(s): {tokens_in:,} in ({cached:,} cached), "
        f"{tokens_out:,} out ({reasoning:,} reasoning)"
    )
    if missing_in or missing_cached or missing_out or missing_reasoning:
        line += "; some counts not reported"
    return line


def _format_error(record: Mapping[str, Any], rule: str) -> str:
    error = record["error"]
    lines = [
        rule,
        f"CASE     {record['source']}",
        f"OUTCOME  errored  ({error['type']}"
        + (f" in {error['role']}" if error.get("role") else "")
        + f", after {record['duration_seconds']}s)",
        rule,
        "",
        "ERROR",
        _indent(error["message"]),
    ]
    for field, label in (("refusal_text", "REFUSAL TEXT"), ("raw_output", "RAW OUTPUT")):
        if error.get(field):
            lines += ["", label, _indent(error[field])]
    lines += ["", "TRACEBACK", _indent(error["traceback"])]
    return "\n".join(lines) + "\n"


def format_record(record: Mapping[str, Any]) -> str:
    """Render a record as the teaching case a person would read.

    Reads the *record*, not the graph state, so it works on any file in ``runs/`` as well as on
    a run that just finished -- and so what is shown is exactly what was saved. Reads records of
    every schema version: one written before version 2 has no ``usage`` line.

    The extractor's ``think`` field is left out: it is the model's working notes, not part of
    the teaching case. It stays in the record.

    Args:
        record (Mapping[str, Any]): From ``build_record`` or ``build_error_record``, or read
            back from disk.

    Returns:
        str: Plain text, sectioned, ready to print.
    """
    rule = "-" * 78
    if "error" in record:
        return _format_error(record, rule)

    grading = record["grading"]
    drafts = record["drafts"]
    verdicts = record["verdicts"]

    lines = [
        rule,
        f"CASE     {record['source']}",
        f"OUTCOME  {record['terminal_state']}  "
        f"({len(drafts)} extract pass(es), {len(verdicts)} review(s), "
        f"{record['duration_seconds']}s)",
        f"MODELS   curator {record['config']['curator_model']}  |  "
        f"editor {record['config']['judge_model']}",
    ]
    if "usage" in record:
        lines.append(f"TOKENS   {summarise_usage(record['usage'])}")
    lines += [
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
