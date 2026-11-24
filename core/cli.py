"""Operator commands: ``python -m core.cli ingest`` and ``python -m core.cli search``."""

import argparse
import asyncio
import sys

from core.config import PackError, Settings, load_pack
from core.rag import RagError, build_rag


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m core.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="build or update the document index")
    ingest.add_argument("--rebuild", action="store_true", help="drop the index first")

    search = commands.add_parser("search", help="show what retrieval returns for a query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=None, help="number of results")
    search.add_argument(
        "--all-labels", action="store_true", help="include restricted documents (debugging)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = Settings()
        pack = load_pack(settings.pack)
        rag = build_rag(pack, settings.data_dir)
        if rag is None:
            print(f"{pack.root}: brain.yaml has no 'rag' section, nothing to do")
            return 1

        if args.command == "ingest":
            report = rag.ingestor.run(rebuild=args.rebuild)
            print(
                f"{pack.config.name}: {report.added} added, {report.updated} updated, "
                f"{report.unchanged} unchanged, {report.removed} removed. "
                f"Index now holds {report.chunks} chunks."
            )
        else:
            allowed = None if args.all_labels else frozenset[str]()
            hits = asyncio.run(rag.retriever.search(args.query, args.k or rag.top_k, allowed))
            if not hits:
                print("no results (is the index built? run ingest)")
            for rank, hit in enumerate(hits, start=1):
                chunk = hit.chunk
                labels = f" labels={sorted(chunk.labels)}" if chunk.labels else ""
                print(f"{rank}. {hit.score:.4f}  {chunk.source} :: {chunk.heading}{labels}")
                print(f"   {chunk.text[:160].replace(chr(10), ' ')}")
    except (PackError, RagError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
