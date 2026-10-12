"""Execute notebook smoke tests with fresh kernels and temporary outputs."""

import argparse
import importlib.util
import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    """Smoke-test by default; --full refreshes outputs using the real snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["HDB_SMOKE"] = "0" if args.full else "1"
    with tempfile.TemporaryDirectory(prefix="hdb-notebooks-") as temporary:
        os.environ["HDB_RUN_ROOT"] = str(root / "artifacts") if args.full else temporary
        for path in sorted((root / "notebooks").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            if (
                notebook.metadata.get("optional_neural")
                and importlib.util.find_spec("torch") is None
            ):
                print(
                    f"SKIP {path.name}: optional PyTorch is not installed", flush=True
                )
                continue
            NotebookClient(
                notebook,
                timeout=600 if args.full else 180,
                kernel_name="python3",
                resources={"metadata": {"path": str(root)}},
            ).execute()
            if args.full:
                nbformat.write(notebook, path)
            print(f"PASS {path.name}", flush=True)


if __name__ == "__main__":
    main()
