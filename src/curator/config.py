"""Runtime configuration.

A typed object rather than module-level constants, so Phase 2 can sweep judge model and
grader thresholds without editing code. Imports nothing but the standard library: everything
in the package depends on this module, which is what obliges it to stay cheap. In particular
it must never import ``openai`` -- it holds model *ids*, which are strings.

API keys are deliberately absent. ``T7`` snapshots this object into every run record, so a
key held here would write a secret to ``runs/`` on every case. ``llm.py`` reads keys from the
environment directly.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# Environment variable names. Mirrors .env.example - keep the two in step.
# --------------------------------------------------------------------------------------

CORPUS_DIR_ENV = "CURATOR_CORPUS_DIR"
CURATOR_MODEL_ENV = "CURATOR_MODEL"
JUDGE_MODEL_ENV = "JUDGE_MODEL"
TEMPERATURE_ENV = "CURATOR_TEMPERATURE"
MAX_REFINES_ENV = "CURATOR_MAX_REFINES"

DEFAULT_CORPUS_DIR = Path("extracted_data")

# NOTE: the model ids are YOUR decision, not mine, and they are Phase 2 material.
# `gpt-4o-mini` is a placeholder that certainly exists; check for a current id before the
# first real run. JUDGE_MODEL is deliberately left empty because .env.example requires it to
# be a *different model family* from CURATOR_MODEL - self-enhancement bias, PLAN.md section
# 2.6 - and there is no sane cross-family default to pick on your behalf.
DEFAULT_CURATOR_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODEL = ""

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_REFINES = 3


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    """Read ``name`` from ``env``, treating an empty value as absent.

    ``.env`` files routinely carry ``CURATOR_MODEL=`` with nothing after the equals sign -
    .env.example ships exactly that - and ``os.environ.get(name, default)`` would hand back
    the empty string rather than the default. This coalesces both cases.

    Args:
        env (Mapping[str, str]): The environment mapping to read from.
        name (str): The variable name.
        default (str): Value to use when the variable is absent or empty.

    Returns:
        str: The value, or ``default``.
    """
    return env.get(name, "").strip() or default


def corpus_dir_from_env(env: Mapping[str, str] | None = None) -> Path:
    """Return the corpus root directory.

    The single source of truth for ``CURATOR_CORPUS_DIR``. ``figures.corpus_root()``
    delegates here, so the default path is written down exactly once.

    Kept as a module-level function rather than a ``Configuration`` field lookup on purpose:
    the figure resolver needs only this one value, and should not have to construct a whole
    ``Configuration`` (with model ids and thresholds it will never read) to get at it.

    Args:
        env (Mapping[str, str] | None): Environment to read; defaults to ``os.environ``.

    Returns:
        Path: The configured corpus root, or ``extracted_data`` if unset.
    """
    env = os.environ if env is None else env
    return Path(_get(env, CORPUS_DIR_ENV, str(DEFAULT_CORPUS_DIR)))


@dataclass(frozen=True)
class GradeThresholds:
    """A data class representing the grading thresholds for different categories."""
    case_presentation_score: int = 3
    images_usefulness_score: int = 3
    integrative_reasoning_score: int = 3
    transparency_score: int = 3
    differential_diagnosis_score: bool = True
    final_diagnosis_score: bool = True

    # the four >=3 gates and the two Yes/No gates, at today's values
    def passes(self, grading) -> bool: 
        """Determine if the grading passes all thresholds.

        Args:
            grading (dict): A dictionary containing the scores for each category.

        Returns:
            bool: True if the grading passes all thresholds, False otherwise.
        """
        return (
            grading.get("case_presentation_score", 0) >= self.case_presentation_score
            and grading.get("images_usefulness_score", 0) >= self.images_usefulness_score
            and grading.get("integrative_reasoning_score", 0) >= self.integrative_reasoning_score
            and grading.get("transparency_score", 0) >= self.transparency_score
            and grading.get("differential_diagnosis_score", False)
            == self.differential_diagnosis_score
            and grading.get("final_diagnosis_score", False) == self.final_diagnosis_score
        )
    

@dataclass(frozen=True)
class Configuration:
    # curator_model, judge_model, temperature, max_refines, corpus_dir
    curator_model: str
    judge_model: str
    temperature: float
    max_refines: int
    corpus_dir: Path
    thresholds: GradeThresholds

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Configuration":
        """Build a configuration from environment variables.

        This is the only place the environment is read. Nothing happens at import time, so a
        Phase 2 sweep that sets a variable between runs gets the new value, and a test can
        pass ``env`` directly instead of monkeypatching ``os.environ``.

        Args:
            env (Mapping[str, str] | None): Environment to read; defaults to ``os.environ``.

        Returns:
            Configuration: The configuration, with defaults filled in for anything absent or
            empty.
        """
        env = os.environ if env is None else env
        return cls(
            curator_model=_get(env, CURATOR_MODEL_ENV, DEFAULT_CURATOR_MODEL),
            judge_model=_get(env, JUDGE_MODEL_ENV, DEFAULT_JUDGE_MODEL),
            temperature=float(_get(env, TEMPERATURE_ENV, str(DEFAULT_TEMPERATURE))),
            max_refines=int(_get(env, MAX_REFINES_ENV, str(DEFAULT_MAX_REFINES))),
            corpus_dir=corpus_dir_from_env(env),
            thresholds=GradeThresholds()
        )