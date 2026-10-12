"""Restore the exact original CSV from Git history without a moving download."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "e45aff1:ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv"
SHA256 = "fb23b8460f1a762e4220382e91f3bfcf6030bb69286b1a5577646c33d7a608d1"


def main() -> None:
    """Restore only the known snapshot and verify existing copies."""
    output = ROOT / "data" / "raw" / "resale.csv"
    if output.exists():
        content = output.read_bytes()
    else:
        content = subprocess.run(
            ["git", "show", SOURCE],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Snapshot checksum mismatch; existing data was not changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with output.open("xb") as stream:
            stream.write(content)
    print(f"Verified {output}")


if __name__ == "__main__":
    main()
