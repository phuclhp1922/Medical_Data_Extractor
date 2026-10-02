"""Turning resolved figures into the content parts a Responses request carries.

The seam this module exists to create: **state holds ``Figure`` objects, never bytes.** A
``Figure`` is a label, a path and a caption -- small, serialisable, and safe to write into a T7
run record. Base64 is produced here, at call time, and discarded with the request. Putting it in
a state channel instead would write tens of megabytes of copyrighted figures into ``runs/`` once
per case, and the record would be unreadable besides.

**Each image is introduced by its label.** The parts come back interleaved -- a text part naming
the figure, then the image itself, then the next label, and so on -- because a bare sequence of
images gives the model no way to obey "cite Fig 24.1". That instruction is impossible against the
notebook's payload (PLAN 2.1): it globbed all 219 ``.jpg`` in the corpus, sent 77 tables and
decorations among them, and labelled none. ``figures.resolve_case()`` supplies the 142 real
figures with labels; this module is where the label reaches the model.

The text part carries ``Figure.label`` -- the positional one, which every figure has -- followed
by the textbook's own caption where there is one. The caption is already inside ``full_text``, so
this duplicates tokens; it buys the model a description of the image sitting immediately beside
the image, which a label alone does not give.

The two are never reconciled. A caption usually opens with its own "Fig. 24.1", so the label is
often repeated, and where ``caption_label`` *disagrees* with ``Figure.label`` both spellings go
to the model as they are. That disagreement is a property of the corpus worth reporting
(``figures.agreement_report``), not something to paper over in a prompt.

**No prose lives here.** Wording like "the figures from this case follow" belongs in
``prompts/``, where it is versioned and fingerprinted with the rest of the instructions. This
module contributes only labels and pixels, so that a prompt change is always visible as a prompt
change.
"""

import base64
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from openai.types.responses import ResponseInputImageParam, ResponseInputTextParam

from curator.figures import CorpusError, Figure

#: Content-part types this module emits, in the order a request carries them.
ContentPart = ResponseInputTextParam | ResponseInputImageParam

#: Image formats the Responses API accepts, keyed by file suffix. Anything else in a case
#: directory is a corpus fault, not something to send and hope: an unsupported type comes back as
#: a 400 whose message does not say which file caused it.
#:
#: Written out rather than asked of ``mimetypes.guess_type``, which on Windows consults the
#: registry -- so the media type of a ``.jpg`` becomes a property of whatever software is
#: installed. A machine whose registry maps ``.jpg`` to ``image/pjpeg`` would fail every figure in
#: the corpus (all 219 are ``.jpg``) and report it as a corpus fault. A run must not depend on the
#: machine it runs on, least of all invisibly.
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _data_url(image_path: Path) -> str:
    """Read an image file and encode it as a ``data:`` URL.

    Inline rather than uploaded: the notebook's ``client.files.upload`` / ``files.delete`` pair
    has no equivalent here and simply disappears. A file lifecycle buys nothing for images used
    exactly once, and leaks storage whenever a run dies between the two calls.

    Args:
        image_path (Path): Path to the image, as ``figures.resolve_case()`` recorded it.

    Returns:
        str: ``data:<media type>;base64,<payload>``.

    Raises:
        FileNotFoundError: The file named by the ``Figure`` is not there. Loud on purpose --
            a resolved figure whose file has moved means the corpus and the index disagree,
            and silently dropping it would quietly shrink the evidence the model reasons from.
        CorpusError: The suffix is not one the API accepts.
    """
    media_type = MEDIA_TYPES.get(image_path.suffix.lower())
    if media_type is None:
        raise CorpusError(
            f"{image_path} has suffix {image_path.suffix!r}; "
            f"expected one of {sorted(MEDIA_TYPES)}"
        )

    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"


def _caption_text(figure: Figure) -> str:
    """Render the text part that introduces one figure.

    Args:
        figure (Figure): The figure being introduced.

    Returns:
        str: The label, and the caption on the line below it when the figure has one. A figure
        whose caption is empty gets its label alone -- an empty line under the label would
        read to the model as a caption that says nothing, rather than as no caption.
    """
    if not figure.caption:
        return figure.label
    return f"{figure.label}\n{figure.caption}"


def as_content_parts(
    figures: Iterable[Figure],
    *,
    detail: Literal["low", "high", "auto"] = "auto",
) -> Sequence[ContentPart]:
    """Render figures as label-then-image content parts, in order.

    Args:
        figures (Iterable[Figure]): The case's figures, as ``figures.resolve_case()`` returns
            them. An empty iterable yields an empty sequence -- a case with no figures is a
            case with no figure parts, not an error.
        detail (Literal["low", "high", "auto"]): The API's image-fidelity hint, passed through
            unchanged. ``"low"`` is markedly cheaper per image and is the knob to reach for if
            a full corpus run proves too expensive; that would be a measured decision, so the
            default changes nothing today.

    Returns:
        Sequence[ContentPart]: Twice as many parts as figures -- a text part naming and
        describing each figure, immediately followed by its image.

    Raises:
        FileNotFoundError: See ``_data_url``.
        CorpusError: See ``_data_url``.
    """
    parts: list[ContentPart] = []
    for figure in figures:
        parts.append(ResponseInputTextParam(type="input_text", text=_caption_text(figure)))
        parts.append(
            ResponseInputImageParam(
                type="input_image",
                image_url=_data_url(figure.image_path),
                detail=detail,
            )
        )
    return parts
