"""Tamper-evident audit log (append-only JSONL, SHA-256 hash chain).

Every AI recommendation and every officer decision is one record. Each record stores
the hash of the previous one, so editing or deleting any past line breaks the chain and
`verify()` reports exactly where. This is what "every decision is logged for auditing"
needs to mean in front of an auditor: not just a log, but a log you can prove is intact.

For production: also ship the latest hash to a second machine (or print it on a daily
report) so an attacker can't rewrite the whole file and recompute every hash.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

GENESIS = "0" * 64


def _digest(rec: dict) -> str:
    body = json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last = self._tail_hash()

    def _tail_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    def append(self, event: str, **data) -> dict:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, "data": data, "prev": self._last}
        rec["hash"] = _digest(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._last = rec["hash"]
        return rec


def verify(path: str | Path) -> tuple[bool, str]:
    prev = GENESIS
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            h = rec.pop("hash")
            if rec["prev"] != prev:
                return False, f"line {n}: chain broken (a record before it was removed or altered)"
            if _digest(rec) != h:
                return False, f"line {n}: content altered"
            prev = h
    return True, f"OK, chain intact, head={prev[:16]}..."


if __name__ == "__main__":
    import sys

    ok, msg = verify(sys.argv[1])
    print(msg)
    sys.exit(0 if ok else 1)
