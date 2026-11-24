"""Loader, chunker and embedders."""

from pathlib import Path

import numpy as np
import pytest

from core.config import EmbedderConfig, EmbedderKind
from core.rag import (
    Document,
    FastEmbedEmbedder,
    HashingEmbedder,
    MarkdownChunker,
    MarkdownFolderLoader,
    RagError,
    build_embedder,
)


def write(directory: Path, name: str, text: str) -> None:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- loader ------------------------------------------------------------------------------


def test_loader_reads_front_matter_and_nested_files(tmp_path: Path) -> None:
    write(
        tmp_path, "b.md", "---\ntitle: Fees\nlabels: [internal, finance]\n---\n# Ignored\n\nBody."
    )
    write(tmp_path, "sub/a.md", "# From Heading\n\nText.")
    write(tmp_path, "plain-file-name.md", "No heading here.")
    write(tmp_path, "notes.txt", "not markdown")

    docs = {doc.source: doc for doc in MarkdownFolderLoader(tmp_path).load()}

    assert sorted(docs) == ["b.md", "plain-file-name.md", "sub/a.md"]
    assert docs["b.md"].title == "Fees"
    assert docs["b.md"].labels == {"internal", "finance"}
    assert docs["b.md"].text.startswith("# Ignored")  # front matter is not part of the text
    assert docs["sub/a.md"].title == "From Heading"
    assert docs["sub/a.md"].labels == frozenset()
    assert docs["plain-file-name.md"].title == "Plain File Name"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("---\ntitle: X\n# never closed", "front matter is not closed"),
        ("---\ntitle: [unbalanced\n---\nbody", "not valid YAML"),
        ("---\nlabels: internal\n---\nbody", "'labels' must be a list of strings"),
        ("---\n- a\n---\nbody", "front matter must be a mapping"),
    ],
)
def test_loader_reports_broken_files_by_path(tmp_path: Path, text: str, message: str) -> None:
    write(tmp_path, "bad.md", text)

    with pytest.raises(RagError, match=message) as caught:
        list(MarkdownFolderLoader(tmp_path).load())
    assert "bad.md" in str(caught.value)


def test_loader_needs_the_folder_to_exist(tmp_path: Path) -> None:
    with pytest.raises(RagError, match="documents folder not found"):
        list(MarkdownFolderLoader(tmp_path / "nope").load())


def test_content_hash_covers_text_title_and_labels() -> None:
    base = Document("a.md", "T", "body")

    assert base.content_hash == Document("renamed.md", "T", "body").content_hash
    assert base.content_hash != Document("a.md", "T", "body!").content_hash
    assert base.content_hash != Document("a.md", "T2", "body").content_hash
    assert base.content_hash != Document("a.md", "T", "body", frozenset({"x"})).content_hash


# --- chunker -----------------------------------------------------------------------------

DOC = Document(
    source="policy.md",
    title="Returns Policy",
    labels=frozenset({"internal"}),
    text=(
        "# Returns Policy\n\nIntro line.\n\n"
        "## Window\n\nThirty days.\n\n### Exceptions\n\nCustom anvils.\n\n"
        "## Shipping\n\nFree for damage.\n\n```\n# not a heading\n```\n"
    ),
)


def test_chunker_follows_headings_and_keeps_the_path() -> None:
    chunks = MarkdownChunker().split(DOC)

    assert [(c.heading, c.text.splitlines()[0]) for c in chunks] == [
        ("", "Intro line."),
        ("Window", "Thirty days."),
        ("Window > Exceptions", "Custom anvils."),
        ("Shipping", "Free for damage."),
    ]
    assert "# not a heading" in chunks[-1].text  # code blocks are not split on '#'
    assert [c.position for c in chunks] == [0, 1, 2, 3]
    assert all(c.labels == {"internal"} and c.source == "policy.md" for c in chunks)
    assert chunks[2].searchable_text == "Returns Policy > Window > Exceptions\nCustom anvils."


def test_chunker_packs_paragraphs_up_to_the_limit() -> None:
    paragraphs = [f"Paragraph {i} " + "x" * 80 for i in range(6)]
    doc = Document("a.md", "T", "## S\n\n" + "\n\n".join(paragraphs))

    chunks = MarkdownChunker(max_chars=200).split(doc)

    assert len(chunks) == 3  # two ~92-char paragraphs fit in 200, three do not
    assert all(len(c.text) <= 200 for c in chunks)
    assert "\n\n".join(c.text for c in chunks) == "\n\n".join(paragraphs)  # nothing lost


def test_chunker_splits_an_oversized_paragraph_at_sentences_then_hard() -> None:
    sentences = " ".join(f"Sentence number {i} is here." for i in range(20))
    doc = Document("a.md", "T", sentences + "\n\n" + "y" * 450)

    chunks = MarkdownChunker(max_chars=200).split(doc)

    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].text.endswith(".")  # broke at a sentence end, not mid-word
    assert sum(c.text.count("y") for c in chunks) == 450


def test_chunker_returns_nothing_for_an_empty_document() -> None:
    assert MarkdownChunker().split(Document("a.md", "T", "# Only A Title\n")) == []
    with pytest.raises(ValueError, match="at least 100"):
        MarkdownChunker(max_chars=10)


# --- embedders ---------------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_unit_length() -> None:
    embedder = HashingEmbedder()
    vectors = embedder.embed_documents(["refund policy", "Refund POLICY!", "anvil pricing", ""])

    assert vectors.shape == (4, 256)
    assert np.allclose(vectors[0], vectors[1])  # case and punctuation do not matter
    assert np.allclose(np.linalg.norm(vectors[:3], axis=1), 1.0)
    assert not vectors[3].any()  # empty text is the zero vector, not NaN
    assert float(vectors[0] @ embedder.embed_query("refund")) > float(vectors[2] @ vectors[0])


def test_embedder_is_chosen_by_enum() -> None:
    assert isinstance(build_embedder(EmbedderConfig(kind=EmbedderKind.HASHING)), HashingEmbedder)
    fast = build_embedder(EmbedderConfig(kind=EmbedderKind.FASTEMBED, model="some/model"))
    assert isinstance(fast, FastEmbedEmbedder)
    assert fast.name == "fastembed:some/model"  # built lazily: no model was downloaded
