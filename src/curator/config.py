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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from curator.schemas import Grading

# --------------------------------------------------------------------------------------
# Environment variable names. Mirrors .env.example - keep the two in step.
# --------------------------------------------------------------------------------------

CORPUS_DIR_ENV = "CURATOR_CORPUS_DIR"
CURATOR_MODEL_ENV = "CURATOR_MODEL"
JUDGE_MODEL_ENV = "JUDGE_MODEL"
TEMPERATURE_ENV = "CURATOR_TEMPERATURE"
JUDGE_TEMPERATURE_ENV = "JUDGE_TEMPERATURE"
MAX_REFINES_ENV = "CURATOR_MAX_REFINES"
JUDGE_BASE_URL_ENV = "JUDGE_BASE_URL"
PAUSE_SECONDS_ENV = "CURATOR_PAUSE_SECONDS"
REQUEST_RETRIES_ENV = "CURATOR_REQUEST_RETRIES"


DEFAULT_CORPUS_DIR = Path("extracted_data")
# NOTE: the model ids are YOUR decision, not mine, and they are Phase 2 material.
# `gpt-4o-mini` is a placeholder that certainly exists; check for a current id before the
# first real run. JUDGE_MODEL is deliberately left empty because .env.example requires it to
# be a *different model family* from CURATOR_MODEL - self-enhancement bias, PLAN.md section
# 2.6 - and there is no sane cross-family default to pick on your behalf.
DEFAULT_CURATOR_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODEL = ""
DEFAULT_TEMPERATURE = 0.1
# The one value of CURATOR_TEMPERATURE that is not a number: it omits the parameter from the
# request. Reasoning models reject `temperature` with a 400, so "no temperature at all" has to
# be sayable -- and it is a different request from any number, blank included (blank means the
# default above, which is why an empty string cannot serve as the sentinel).
NO_TEMPERATURE = "none"
# The judge keeps its own setting because the constraint belongs to the *model*, not to the
# pipeline: one role can be a reasoning model that rejects `temperature` while the other is not.
# Sharing one value forced a choice between sending it to a model that refuses it and dropping
# it from a model that wants it -- and dropping it means the provider's default, which for
# extraction is far higher than anything chosen here.
DEFAULT_JUDGE_TEMPERATURE = DEFAULT_TEMPERATURE
DEFAULT_MAX_REFINES = 3
DEFAULT_PAUSE_SECONDS = 5.0
DEFAULT_REQUEST_RETRIES = 2


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


def _temperature(env: Mapping[str, str], name: str, default: float) -> float | None:
    """Read a temperature, where the word ``none`` means "omit the parameter".

    Args:
        env (Mapping[str, str]): The environment mapping to read from.
        name (str): The variable name.
        default (float): Value to use when the variable is absent or empty.

    Returns:
        float | None: The temperature, or ``None`` to send no ``temperature`` at all.
    """
    raw = _get(env, name, str(default))
    return None if raw.lower() == NO_TEMPERATURE else float(raw)


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
    differential_diagnosis_present: bool = True
    final_diagnosis_present: bool = True

    # the four >=3 gates and the two Yes/No gates, at today's values
    def passes(self, grading: 'Grading') -> bool:
        """Determine if the grading passes all thresholds.

        Args:
            grading (dict): A dictionary containing the scores for each category.

        Returns:
            bool: True if the grading passes all thresholds, False otherwise.
        """
        if not grading.is_case_report:
            return False
        else:
            return (
                grading.case_presentation_score >= self.case_presentation_score and
                grading.images_usefulness_score >= self.images_usefulness_score and
                grading.integrative_reasoning_score >= self.integrative_reasoning_score and
                grading.transparency_score >= self.transparency_score and
                grading.differential_diagnosis_present == self.differential_diagnosis_present and
                grading.final_diagnosis_present == self.final_diagnosis_present
            )


@dataclass(frozen=True)
class Configuration:
    curator_model: str
    judge_model: str
    #: Endpoint for the judge's vendor, or ``None`` for OpenAI's own. The key itself is never
    #: held here (see the module docstring); only where to send it.
    judge_base_url: str | None
    #: Sampling temperature for the curator roles (extractor, grader); ``None`` sends none.
    temperature: float | None
    #: The editor's own, because it is a different model with different rules.
    judge_temperature: float | None
    max_refines: int
    corpus_dir: Path
    thresholds: GradeThresholds
    pause_seconds: float
    request_retries: int

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
            judge_base_url=env.get(JUDGE_BASE_URL_ENV, "").strip() or None,
            temperature=_temperature(env, TEMPERATURE_ENV, DEFAULT_TEMPERATURE),
            judge_temperature=_temperature(
                env, JUDGE_TEMPERATURE_ENV, DEFAULT_JUDGE_TEMPERATURE
            ),
            max_refines=int(_get(env, MAX_REFINES_ENV, str(DEFAULT_MAX_REFINES))),
            corpus_dir=corpus_dir_from_env(env),
            thresholds=GradeThresholds(),
            pause_seconds=float(_get(env, PAUSE_SECONDS_ENV, str(DEFAULT_PAUSE_SECONDS))),
            request_retries=int(_get(env, REQUEST_RETRIES_ENV, str(DEFAULT_REQUEST_RETRIES)))
        )
