"""Download only the two version-pinned release assets and verify their hashes."""
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "checkpoints.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["assets"]:
        target = args.output_dir / item["name"]
        if target.exists():
            if digest(target) != item["sha256"]:
                raise RuntimeError(f"Existing file has the wrong hash; not overwriting: {target}")
            print("Verified existing " + item["name"], flush=True)
            continue
        partial = target.with_suffix(target.suffix + ".download")
        request = urllib.request.Request(item["url"], headers={"User-Agent": "ChekiEdgeFitRT/2.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
        if partial.stat().st_size != item["bytes"] or digest(partial) != item["sha256"]:
            raise RuntimeError("Downloaded asset failed verification: " + str(partial))
        os.replace(partial, target)
        print("Downloaded and verified " + item["name"], flush=True)


if __name__ == "__main__":
    main()
