"""Record a real run's HTTP traffic once, then replay it with no network (T8).

**Why not vcrpy**, which the T8 brief named. ``openai==3.3.1`` sends its requests through
``httpx2``, not ``httpx``, and vcrpy works by patching the HTTP libraries it knows. Rather than
add a dependency and hope it sees the traffic, this sits at the one place every request is
guaranteed to pass: the client's *transport*. ``OpenAI(http_client=httpx2.Client(transport=...))``
is supported API, and a transport is one method, ``handle_request``.

**What a cassette holds.** A JSON list, one entry per request, in the order they were made:

- the request as ``method``, ``path`` and the **SHA-256 of its body** -- never the body itself.
  The body carries the case's full source text and its figures as base64, all copyrighted;
  a hash identifies the request exactly without containing any of it.
- the response's status, content type and body. The body is the model's answer, which quotes
  the source -- so a cassette recorded on a textbook case must not be committed (D1).

**Matching on the body hash** is what the brief's "gotcha" is about: grade, extract and review
are all ``POST /v1/responses``, so method and path alone would hand the extractor the grader's
answer. The hash tells them apart, and it is also the reason replay is strict. If any byte of a
request changes -- a prompt edit, a model id, a temperature, a different figure -- its hash is
not in the cassette, and replay fails with ``CassetteMiss`` instead of returning a stale answer.
That failure means "re-record", and it is the correct behaviour.

**Headers are not stored**, except the content type. Response headers carry the organisation
and project ids, request ids and cookies; none of them are needed to replay an answer.
"""

import hashlib
import json
from pathlib import Path

import httpx2


class CassetteMiss(LookupError):
    """Replay was asked for a request the cassette does not hold."""


def _request_key(request: httpx2.Request) -> dict[str, str]:
    """What identifies a request: method, path, and a hash of the exact body bytes."""
    return {
        "method": request.method,
        "path": request.url.path,
        "body_sha256": hashlib.sha256(request.read()).hexdigest(),
    }


class CassetteTransport(httpx2.BaseTransport):
    """An ``httpx2`` transport that either records real traffic or replays a recording.

    Args:
        path (Path): The cassette file.
        record (bool): ``True`` to send every request to the real network and save it;
            ``False`` to answer every request from the cassette and never touch the network.

    Raises:
        FileExistsError: Recording was asked for and the cassette already exists. Delete it
            first: a recording is never silently replaced.
        FileNotFoundError: Replay was asked for and there is no cassette.
    """

    def __init__(self, path: Path, *, record: bool) -> None:
        self.path = path
        self.record = record
        if record:
            if path.exists():
                raise FileExistsError(f"{path} exists; delete it to record a new one")
            self.entries: list[dict] = []
            self._network = httpx2.HTTPTransport()
        else:
            self.entries = json.loads(path.read_text(encoding="utf-8"))
            self._unused = list(range(len(self.entries)))

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        key = _request_key(request)
        if self.record:
            return self._record(request, key)
        return self._replay(key)

    def _record(self, request: httpx2.Request, key: dict[str, str]) -> httpx2.Response:
        response = self._network.handle_request(request)
        content = response.read()  # decoded: gzip is undone here, so the cassette is plain text
        content_type = response.headers.get("content-type", "application/json")
        self.entries.append(
            {
                "request": key,
                "response": {
                    "status": response.status_code,
                    "content_type": content_type,
                    "body": content.decode("utf-8"),
                },
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # A fresh response, not the network's: its headers said gzip, and the body is now
        # decoded, so passing them on would have the client try to decode it a second time.
        return httpx2.Response(
            response.status_code, headers={"content-type": content_type}, content=content
        )

    def _replay(self, key: dict[str, str]) -> httpx2.Response:
        # The first unused entry with this key. Order matters only between identical requests,
        # which this pipeline never makes twice -- but if it did, they replay in recorded order.
        for position, index in enumerate(self._unused):
            entry = self.entries[index]
            if entry["request"] == key:
                del self._unused[position]
                response = entry["response"]
                return httpx2.Response(
                    response["status"],
                    headers={"content-type": response["content_type"]},
                    content=response["body"].encode("utf-8"),
                )
        raise CassetteMiss(
            f"no recorded response for {key['method']} {key['path']} with body "
            f"{key['body_sha256'][:12]}... -- the request differs from the recording "
            "(a prompt, model, setting or figure changed). Re-record the cassette."
        )

    @property
    def unused(self) -> int:
        """How many recorded responses replay has not handed out yet."""
        return 0 if self.record else len(self._unused)
