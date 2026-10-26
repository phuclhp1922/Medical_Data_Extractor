r"""Run a batch of cases through the pipeline and leave one record per case (T7).

From the repo root, in cmd::

    .venv\Scripts\python.exe -m curator.run --limit 10        new batch: the first ten cases
    .venv\Scripts\python.exe -m curator.run 1 3 17            new batch: cases 1, 3 and 17
    .venv\Scripts\python.exe -m curator.run                   new batch: every case
    .venv\Scripts\python.exe -m curator.run --batch batch-20261023T020209Z    resume a batch
    .venv\Scripts\python.exe -m curator.run --batch batch-20261023T020209Z 5 6   add cases 5, 6
    .venv\Scripts\python.exe -m curator.run --show batch-20261023T020209Z 4   print case 4, free
    .venv\Scripts\python.exe -m curator.run --show runs\<case>\<time>.json    print a record file
    .venv\Scripts\python.exe -m curator.run --funnel batch-20261023T020209Z   count a batch, free

A batch prints its funnel when it finishes, so ``--funnel`` is for looking again later, or at a
batch that was stopped part-way.

**Costs money.** Each case spends at most ``1 + 2 x (max_refines + 1)`` requests: one grading,
then an extract and a review per pass. The header prints the ceiling for the whole batch before
anything is sent.

A batch is a folder, ``runs/batch-<UTC time>/``, holding ``batch.json`` (the configuration,
prompt fingerprints, commit and starting list of cases, fixed when the batch is created), an
``additions/`` folder if cases were added later, and one record per case. See ``records.py``
for the record itself. Four behaviours worth knowing, all from D29 and its amendments:

- **Every error is recorded, and the batch continues.** A case whose run raises gets an error
  record and the next case starts. That includes setup faults: a bad API key fails every case,
  and shows up as an ``errored`` count equal to the batch size. Ctrl+C stops the batch.
- **Resume skips finished cases and retries errored ones.** An error is not an outcome, so a
  resumed batch tries it again; the old error record stays beside the new attempt.
- **A batch grows only when you name cases.** ``--batch NAME 5 6`` adds cases 5 and 6, then
  runs everything in the batch not yet finished. ``--batch NAME`` alone never adds anything, so
  a resume cannot quietly turn a four-case batch into a ninety-three-case one.
- **Resume refuses a changed experiment.** If the configuration or any prompt fingerprint
  differs from the batch's ``batch.json``, nothing runs: one batch, one rubric. Pacing and retry
  settings may differ, since they change speed, not results.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from curator.config import Configuration
from curator.funnel import funnel
from curator.graph import build_graph
from curator.records import (
    DEFAULT_RUNS_DIR,
    MANIFEST_NAME,
    add_sources,
    batch_sources,
    build_error_record,
    build_record,
    check_manifest,
    finished_sources,
    format_record,
    load_batch,
    summarise_usage,
    write_batch_record,
    write_manifest,
)

# --------------------------------------------------------------------------------------
# Choosing the cases
# --------------------------------------------------------------------------------------


def case_number(source: str) -> int | None:
    """The leading number of a case directory name -- ``"3---A-26-Year..."`` gives ``3``."""
    head, separator, _ = source.partition("---")
    return int(head) if separator and head.isdigit() else None


def list_sources(corpus_dir: Path) -> list[str]:
    """Every case directory in the corpus, in case-number order.

    Numeric order, not text order: as text, ``"10---"`` sorts before ``"2---"``, which is how
    ``cases[1]`` once turned out to be case 10. Names without a number sort last, by name.
    """
    names = [path.name for path in corpus_dir.iterdir() if path.is_dir()]
    return sorted(
        names, key=lambda name: (case_number(name) is None, case_number(name) or 0, name)
    )


def select_sources(available: list[str], wanted: Sequence[str], limit: int | None) -> list[str]:
    """Resolve the cases asked for on the command line against the corpus.

    Args:
        available (list[str]): Every case directory, in run order.
        wanted (Sequence[str]): Case numbers (``"3"``) or full directory names. Empty means all.
        limit (int | None): Keep only the first ``limit`` of the selection.

    Returns:
        list[str]: Directory names, in the order asked for (or corpus order, for all).

    Raises:
        ValueError: A case number or name that matches nothing in the corpus.
    """
    if not wanted:
        chosen = list(available)
    else:
        chosen = []
        for item in wanted:
            if item.isdigit():
                matches = [name for name in available if case_number(name) == int(item)]
            else:
                matches = [name for name in available if name == item]
            if not matches:
                raise ValueError(f"no case {item!r} in the corpus")
            chosen.append(matches[0])
    return chosen if limit is None else chosen[:limit]


# --------------------------------------------------------------------------------------
# Running them
# --------------------------------------------------------------------------------------


def run_batch(
    sources: Sequence[str],
    cfg: Configuration,
    batch_dir: Path,
    *,
    graph: Any,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Run each source through ``graph`` and write its record into ``batch_dir``.

    One case at a time, in order. A case that already has a finished record is skipped; one that
    raises gets an error record and the loop moves on (D29).

    Args:
        sources (Sequence[str]): The case directory names to run.
        cfg (Configuration): Passed to every run as its context.
        batch_dir (Path): An existing batch folder (``write_manifest`` creates one).
        graph: A compiled graph -- ``build_graph()``, or anything with the same ``invoke``.
        now (Callable[[], datetime]): The clock. Replaceable so a test can fix the times.
        log (Callable[[str], None]): Where the one-line progress reports go.

    Returns:
        list[Path]: The records written by this call, in order. Skipped cases add none.
    """
    done = finished_sources(batch_dir)
    written: list[Path] = []
    for index, source in enumerate(sources, start=1):
        prefix = f"[{index}/{len(sources)}] {source}"
        if source in done:
            log(f"{prefix}  already finished, skipped")
            continue

        started_at = now()
        try:
            out = graph.invoke({"source": source}, context=cfg)
        except Exception as error:  # D29: record every error, and carry on with the next case
            record = build_error_record(
                source, error, cfg, started_at=started_at, finished_at=now(), batch=batch_dir.name
            )
            log(f"{prefix}  errored  {type(error).__name__}: {error}")
        else:
            record = build_record(
                out, cfg, started_at=started_at, finished_at=now(), batch=batch_dir.name
            )
            log(
                f"{prefix}  {record['terminal_state']}  "
                f"{len(record['drafts'])} draft(s)  {record['duration_seconds']}s"
            )
        written.append(write_batch_record(record, batch_dir))
    return written


# --------------------------------------------------------------------------------------
# Reporting a batch
# --------------------------------------------------------------------------------------


def format_funnel(batch_dir: Path) -> str:
    """The batch's funnel as a small table, read from its records on disk alone.

    Counts come from ``funnel`` over ``load_batch`` -- one record per case attempted. Cases the
    batch was created for but has not reached yet (a batch stopped with Ctrl+C) are listed on a
    separate line, *outside* the funnel: they have no record, so they are not an outcome.

    Args:
        batch_dir (Path): A batch folder.

    Returns:
        str: Plain text, ready to print.
    """
    records = load_batch(batch_dir)
    counts = funnel(records)
    planned = len(batch_sources(batch_dir))
    width = max(len(bucket) for bucket in counts)
    lines = [f"FUNNEL  {batch_dir.name}"]
    lines += [f"  {bucket:<{width}}  {count:>4}" for bucket, count in counts.items()]
    lines.append(f"  {'total':<{width}}  {sum(counts.values()):>4}  of {planned} case(s) planned")
    if len(records) < planned:
        lines.append(f"  {'not yet run':<{width}}  {planned - len(records):>4}  (resume to finish)")
    usage = [call for record in records for call in record.get("usage", [])]
    if usage:
        lines.append(f"TOKENS  {summarise_usage(usage)}  (successful calls only)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------


def _batch_dir(name: str, runs_dir: Path) -> Path:
    """A batch folder given as a path, or as a name under ``runs_dir``."""
    path = Path(name)
    return path if path.is_dir() else runs_dir / name


def _record_to_show(show: list[str], runs_dir: Path) -> dict[str, Any]:
    """Find the record ``--show`` asks for: a file, or a batch plus a case number.

    For a case in a batch it is the record the funnel counts -- the finished one if the case has
    one, otherwise its latest error -- so what is shown and what is counted always agree.

    Raises:
        ValueError: Nothing matches, with a message saying what was looked for.
    """
    if len(show) == 1:
        path = Path(show[0])
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        if (_batch_dir(show[0], runs_dir) / MANIFEST_NAME).is_file():
            raise ValueError("that is a batch: add a case number, e.g. --show <batch> 4")
        raise ValueError(f"no record file at {path}")
    if len(show) != 2 or not show[1].isdigit():
        raise ValueError("--show takes a record file, or a batch and a case number")
    batch_dir = _batch_dir(show[0], runs_dir)
    if not (batch_dir / MANIFEST_NAME).is_file():
        raise ValueError(f"no batch at {batch_dir} (no {MANIFEST_NAME})")
    number = int(show[1])
    for record in load_batch(batch_dir):
        if case_number(record["source"]) == number:
            return record
    in_batch = [case_number(source) for source in batch_sources(batch_dir)]
    raise ValueError(f"case {number} has no record in {batch_dir.name}; its cases are {in_batch}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m curator.run",
        description="Run a batch of cases. Costs money: every case calls the models.",
    )
    parser.add_argument(
        "cases", nargs="*", help="case numbers or directory names; none means every case"
    )
    parser.add_argument("--limit", type=int, help="run only the first N of the selection")
    parser.add_argument(
        "--batch",
        help="resume this batch (a folder name under runs/, or a path); cases named with it are "
        "added to it",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="where batches live (runs/)"
    )
    parser.add_argument(
        "--show",
        nargs="+",
        metavar="TARGET",
        help="print one case and exit, costs nothing: a record file, or a batch and a case number",
    )
    parser.add_argument(
        "--funnel", metavar="BATCH", help="print a batch's funnel and exit; costs nothing"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m curator.run``. Returns the process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)

    # Records hold characters such as "μ" that Windows' default console encoding cannot print,
    # and a crash at the print would come after the money was spent.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.show:
        try:
            record = _record_to_show(args.show, args.runs_dir)
        except ValueError as error:
            parser.error(str(error))
        print(format_record(record))
        return 0

    if args.funnel:
        batch_dir = Path(args.funnel)
        if not batch_dir.is_dir():
            batch_dir = args.runs_dir / args.funnel
        if not (batch_dir / MANIFEST_NAME).is_file():
            parser.error(f"no batch at {batch_dir} (no {MANIFEST_NAME})")
        print(format_funnel(batch_dir))
        return 0

    load_dotenv()
    cfg = Configuration.from_env()

    if args.batch:
        if args.limit is not None:
            parser.error("--limit means 'the first N cases'; to add cases to a batch, name them")
        batch_dir = Path(args.batch)
        if not batch_dir.is_dir():
            batch_dir = args.runs_dir / args.batch
        if not (batch_dir / MANIFEST_NAME).is_file():
            parser.error(f"no batch at {batch_dir} (no {MANIFEST_NAME})")
        # Checked before anything is added: an addition is the same experiment, or nothing.
        differences = check_manifest(batch_dir, cfg)
        if differences:
            print(f"Refusing to resume {batch_dir}: it was started under different settings.")
            for line in differences:
                print(f"  {line}")
            print("Start a new batch instead, or restore the settings it was started with.")
            return 2
        if args.cases:
            try:
                wanted = select_sources(list_sources(cfg.corpus_dir), args.cases, None)
            except ValueError as error:
                parser.error(str(error))
            added = add_sources(batch_dir, wanted, added_at=datetime.now(UTC))
            print(f"added   {len(added)} new case(s) to the batch"
                  + (f" ({len(wanted) - len(added)} already in it)" if len(added) < len(wanted)
                     else ""))
        sources = batch_sources(batch_dir)
    else:
        try:
            sources = select_sources(list_sources(cfg.corpus_dir), args.cases, args.limit)
        except ValueError as error:
            parser.error(str(error))
        created_at = datetime.now(UTC)
        batch_dir = args.runs_dir / f"batch-{created_at:%Y%m%dT%H%M%SZ}"
        write_manifest(batch_dir, cfg, sources=sources, created_at=created_at)

    calls_per_case = 1 + 2 * (cfg.max_refines + 1)
    print(f"batch   {batch_dir}")
    print(f"cases   {len(sources)}  (already finished: {len(finished_sources(batch_dir))})")
    print(f"curator {cfg.curator_model} (t={cfg.temperature})")
    print(f"editor  {cfg.judge_model} (t={cfg.judge_temperature}) base_url={cfg.judge_base_url}")
    print(
        f"cap     max_refines={cfg.max_refines}: at most {calls_per_case} calls per case, "
        f"{calls_per_case * len(sources)} for the batch"
    )
    print()

    run_batch(sources, cfg, batch_dir, graph=build_graph())
    print(f"\nrecords in {batch_dir}\n")
    print(format_funnel(batch_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
