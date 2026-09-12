"""Prompt definitions: RULES constants split from render() input helpers.

Every prompt module exposes the same two-name surface -- a ``RULES`` constant holding the text
that is identical for all 93 cases, and a ``render()`` that adds the part which is not -- plus a
fingerprint of that constant. See :func:`fingerprint`.
"""

import hashlib

#: Length of a fingerprint, in hex characters. 12 hex digits is 48 bits: collision-free for the
#: handful of prompt revisions this project will ever have, and short enough to read in a run
#: record. Same convention, and the same reasoning, as an abbreviated git hash.
FINGERPRINT_CHARS = 12


def fingerprint(text: str) -> str:
    """Return a short, stable content hash of a prompt constant.

    Prompt text is a *version-controlled input to a measurement*: six weeks from now, the only
    way to answer "which rubric produced this score" is to have recorded it. A hand-maintained
    ``RULES_VERSION = "v1"`` would answer that question wrongly the first time someone edits the
    text and forgets to bump the string -- and wrongly in the worst direction, since two runs
    would claim the same version while using different prompts. A derived hash cannot drift from
    what it describes.

    Deliberately not normalised: whitespace, line endings and punctuation all change the bytes a
    model receives, so they should all change the fingerprint. (Source line endings are safe --
    Python normalises CRLF to ``
`` when parsing, so a Windows checkout hashes the same as a
    Linux one.)

    Args:
        text (str): The prompt constant to fingerprint.

    Returns:
        str: The first ``FINGERPRINT_CHARS`` hex characters of the SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:FINGERPRINT_CHARS]
