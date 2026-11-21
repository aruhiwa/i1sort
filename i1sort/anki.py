"""AnkiConnect clients: reads over the legacy API (version 6), writes over API v1.

Checked against AnkiConnect on Anki 26.09 (legacy v6 at ``/``, JSON-RPC v1 at ``/api/v1``).
Read-side helpers for phase 2. All requests are batched: notes are fetched in
chunks via ``notesInfo`` and searches go through Anki's own search syntax, so we
never make one HTTP call per card. The client refuses to work on any profile
other than the configured one.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable, Sequence

API_VERSION = 6
CHUNK = 1000


class AnkiConnectError(RuntimeError):
    pass


def quote(s: str) -> str:
    """Quote a value for Anki's search syntax ("note:Japanese Core 2k/6k+" → "note:\\"...\\"")."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class AnkiConnect:
    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 120, profile: str | None = None):
        self.url = url
        self.timeout = timeout
        self.profile = profile
        self._checked = False

    # ------------------------------------------------------------------ transport
    def invoke(self, action: str, **params: Any) -> Any:
        body = json.dumps({"action": action, "version": API_VERSION, "params": params}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.URLError as e:
            raise AnkiConnectError(f"cannot reach AnkiConnect at {self.url} (is Anki running?): {e}") from e
        if not isinstance(data, dict) or set(data) != {"result", "error"}:
            raise AnkiConnectError(f"unexpected AnkiConnect response to {action}: {str(data)[:200]}")
        if data["error"] is not None:
            raise AnkiConnectError(f"{action}: {data['error']}")
        return data["result"]

    def multi(self, actions: Sequence[tuple[str, dict]]) -> list[Any]:
        res = self.invoke("multi", actions=[{"action": a, "version": API_VERSION, "params": p} for a, p in actions])
        out = []
        for (a, _), r in zip(actions, res):
            if isinstance(r, dict) and set(r) == {"result", "error"}:
                if r["error"] is not None:
                    raise AnkiConnectError(f"{a}: {r['error']}")
                r = r["result"]
            out.append(r)
        return out

    # ------------------------------------------------------------------ checks
    def check(self, required: Iterable[str] = ()) -> dict:
        """Verify version, active profile and available actions. Returns a summary."""
        version = self.invoke("version")
        if version < API_VERSION:
            raise AnkiConnectError(f"AnkiConnect API version {version} < {API_VERSION}; update the add-on")
        profile = self.invoke("getActiveProfile")
        if self.profile and profile != self.profile:
            raise AnkiConnectError(
                f"Anki's active profile is {profile!r}, but config says {self.profile!r}; switch profiles in Anki"
            )
        actions = set(self.invoke("apiReflect", scopes=["actions"], actions=None)["actions"])
        missing = sorted(set(required) - actions)
        if missing:
            raise AnkiConnectError(f"AnkiConnect lacks actions: {', '.join(missing)}")
        self._checked = True
        return {"version": version, "profile": profile, "actions": actions}

    def _ensure_checked(self) -> None:
        if not self._checked:
            self.check(("findNotes", "notesInfo", "multi"))

    # ------------------------------------------------------------------ reads
    def find_notes(self, query: str) -> list[int]:
        self._ensure_checked()
        return self.invoke("findNotes", query=query)

    def find_notes_many(self, queries: Sequence[str]) -> list[list[int]]:
        self._ensure_checked()
        return self.multi([("findNotes", {"query": q}) for q in queries])

    def notes_info(self, note_ids: Sequence[int]) -> list[dict]:
        self._ensure_checked()
        out: list[dict] = []
        for k in range(0, len(note_ids), CHUNK):
            out.extend(self.invoke("notesInfo", notes=list(note_ids[k:k + CHUNK])))
        return out

    def deck_names(self) -> list[str]:
        return self.invoke("deckNames")

    def model_field_names(self, model: str) -> list[str]:
        return self.invoke("modelFieldNames", modelName=model)


class AnkiV1:
    """Write side, over AnkiConnect's JSON-RPC API v1 (``/api/v1``; Anki 26.08+).

    Every write is pinned to the collection session read at ``open()``: if Anki switches
    profile or reopens the collection mid-run, the server answers ``staleSession`` instead
    of writing to the wrong collection.
    """

    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 300, profile: str | None = None):
        self.url = url.rstrip("/") + "/api/v1"
        self.timeout = timeout
        self.profile = profile
        self.session: str | None = None
        self._id = 0

    def call(self, method: str, **params: Any) -> Any:
        self._id += 1
        if self.session is not None and method != "collection.get":
            params = {**params, "session": self.session}
        body = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            raise AnkiConnectError(f"{method}: HTTP {e.code} from {self.url} (is AnkiConnect up to date?)") from e
        except urllib.error.URLError as e:
            raise AnkiConnectError(f"cannot reach AnkiConnect at {self.url} (is Anki running?): {e}") from e
        if data.get("id") != self._id:
            raise AnkiConnectError(f"{method}: response id mismatch")
        if "error" in data:
            err = data["error"]
            raise AnkiConnectError(f"{method}: {err.get('message')} {json.dumps(err.get('data'), ensure_ascii=False)[:300]}")
        return data["result"]

    def call_batch(self, calls: Sequence[tuple[str, dict]]) -> list[Any]:
        """Up to 100 independent calls in one JSON-RPC batch (unordered, not a transaction)."""
        if not calls:
            return []
        reqs = []
        for method, params in calls:
            self._id += 1
            if self.session is not None:
                params = {**params, "session": self.session}
            reqs.append({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        req = urllib.request.Request(self.url, json.dumps(reqs).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.URLError as e:
            raise AnkiConnectError(f"cannot reach AnkiConnect at {self.url}: {e}") from e
        by_id = {r.get("id"): r for r in data}
        out = []
        for r_req, (method, _) in zip(reqs, calls):
            r = by_id.get(r_req["id"])
            if r is None or "error" in r:
                err = (r or {}).get("error", {"message": "no response"})
                raise AnkiConnectError(f"{method}: {err.get('message')} {json.dumps(err.get('data'), ensure_ascii=False)[:200]}")
            out.append(r["result"])
        return out

    def open(self, required: Iterable[str] = ()) -> dict:
        """Check the API, the methods we need and the profile; pin the session."""
        d = self.call("api.describe")
        names = {m["name"] if isinstance(m, dict) else m for m in d.get("methods", [])}
        missing = sorted(set(required) - names)
        if missing:
            raise AnkiConnectError(f"AnkiConnect API v1 lacks: {', '.join(missing)} (update the add-on)")
        col = self.call("collection.get")
        if self.profile and col.get("profile") != self.profile:
            raise AnkiConnectError(
                f"Anki's active profile is {col.get('profile')!r}, but config says {self.profile!r}; switch profiles in Anki")
        self.session = col["session"]
        return {"anki": d.get("ankiVersion"), "profile": col.get("profile"), "session": self.session}

    @staticmethod
    def item_errors(result: Any) -> list[str]:
        items = result.get("items", []) if isinstance(result, dict) else []
        return [json.dumps(i.get("error"), ensure_ascii=False)[:200] for i in items if isinstance(i, dict) and "error" in i]


def v1_from_config(cfg) -> AnkiV1:
    a = cfg.get("anki", {}) or {}
    return AnkiV1(a.get("url", "http://127.0.0.1:8765"), float(a.get("timeout", 300)), a.get("profile"))


def from_config(cfg) -> AnkiConnect:
    a = cfg.get("anki", {}) or {}
    return AnkiConnect(a.get("url", "http://127.0.0.1:8765"), float(a.get("timeout", 120)), a.get("profile"))


def _chunked_invoke(client: AnkiConnect, action: str, key: str, ids: Sequence[int]) -> list[dict]:
    out: list[dict] = []
    for k in range(0, len(ids), CHUNK):
        out.extend(client.invoke(action, **{key: list(ids[k:k + CHUNK])}))
    return out


def find_cards(client: AnkiConnect, query: str) -> list[int]:
    client._ensure_checked()
    return client.invoke("findCards", query=query)


def cards_info(client: AnkiConnect, card_ids: Sequence[int]) -> list[dict]:
    """cardsInfo in chunks (it renders card HTML, ~0.4 s per 1000 cards)."""
    client._ensure_checked()
    return _chunked_invoke(client, "cardsInfo", "cards", card_ids)
