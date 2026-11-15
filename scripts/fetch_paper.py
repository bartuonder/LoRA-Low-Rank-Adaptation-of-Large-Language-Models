"""Download the LoRA paper PDF into paper/ (not tracked by git)."""
import sys
import urllib.request
from pathlib import Path

URLS = [
    "https://arxiv.org/pdf/2106.09685",
    "https://arxiv.org/pdf/2106.09685v2",
    "https://arxiv.org/pdf/2106.09685v1",
]
DEST = Path(__file__).resolve().parents[1] / "paper" / "lora_2106.09685.pdf"


def main() -> int:
    if DEST.exists() and DEST.stat().st_size > 100_000:
        return 0
    for url in URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except OSError as e:
            print(f"failed {url}: {e}", file=sys.stderr)
            continue
        if data[:4] == b"%PDF":
            DEST.write_bytes(data)
            print(f"saved {DEST} ({len(data)} bytes)")
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
