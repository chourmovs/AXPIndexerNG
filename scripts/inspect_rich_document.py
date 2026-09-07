#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "shared"), str(ROOT / "client")]

from axp_client.rich_documents import RichDocumentService  # noqa: E402
from axp_core.rich_document import RichDocumentError  # noqa: E402
from axp_core.runtime import load_settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Inspect an indexed DOCX rich-source manifest")
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        rich = RichDocumentService(load_settings()["db_path"]).get_document(args.document_id)
    except RichDocumentError as exc:
        parser.exit(1, f"Error: {exc.code}\n")
    manifest = rich.to_manifest()
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2)); return
    diagnostics = manifest["diagnostics"]
    print(rich.filename)
    print(f"SHA: {rich.sha256}")
    print(f"Blocks: {len(rich.blocks)}")
    print(f"Paragraphs: {diagnostics.get('paragraphs', 0)}")
    print(f"Tables: {diagnostics.get('tables', 0)}")
    print(f"Images: {diagnostics.get('images', 0)}")
    print(f"Unsupported: {diagnostics.get('unsupported_objects', 0)}")
    print(f"Fidelity: {rich.fidelity}")
    print(f"Cache: {'HIT' if diagnostics.get('cache_hit') else 'MISS'}")


if __name__ == "__main__":
    main()
