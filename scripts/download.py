"""Download the lab data and models, and verify checksums.

    python scripts/download.py            # data + both models (~420 MB)
    python scripts/download.py --data     # FLEURS Amharic dev split only (~112 MB)
    python scripts/download.py --models   # the two Whisper-tiny Amharic models (~300 MB)

Everything lands inside this project: data/raw/ and models/. If the venue Wi-Fi is
down, copy the same folders from the facilitators' USB stick and run with --verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "scripts" / "downloads.json").read_text(encoding="utf-8"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_file(item: dict, verify_only: bool) -> bool:
    dest = ROOT / item["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or sha256(dest) != item["sha256"]:
        if verify_only:
            print(f"MISSING/BAD {item['dest']}")
            return False
        print(f"downloading {item['url']} ...", flush=True)
        urllib.request.urlretrieve(item["url"], dest)
    ok = sha256(dest) == item["sha256"]
    print(f"{'OK ' if ok else 'BAD'} {item['dest']}  sha256 {item['sha256'][:12]}")
    return ok


def get_model(m: dict, verify_only: bool) -> bool:
    dest = ROOT / m["dest"]
    if not verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(m["repo"], revision=m["revision"], local_dir=dest,
                          allow_patterns=["*.json", "*.safetensors", "*.txt"])
    w = dest / "model.safetensors"
    ok = w.exists() and sha256(w) == m["sha256"]
    print(f"{'OK ' if ok else 'BAD'} {m['dest']}  ({m['repo']}@{m['revision'][:7]}, {m['license']})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--verify", action="store_true", help="only check what is already there")
    a = ap.parse_args()
    both = not (a.data or a.models)
    ok = True
    if a.data or both:
        ok &= all(get_file(i, a.verify) for i in SPEC["files"])
    if a.models or both:
        ok &= all(get_model(m, a.verify) for m in SPEC["models"])
    print("ALL OK" if ok else "SOME FILES MISSING OR CORRUPT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
