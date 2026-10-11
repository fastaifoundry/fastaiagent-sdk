"""The chunker overlaps neighbours, stays within ``chunk_size``, and emits each span once (1.88.0).

Two defects, found by the knowledge-base boundary page's proofs:

* ``chunk_overlap`` was accepted and did nothing — neighbouring chunks shared no
  text, while the docs promised a 50-character overlap, so a rule that straddled a
  seam was always split in two.
* A part longer than ``chunk_size`` left the splitter's accumulator set, so the
  text before it was emitted **twice**, the second time glued to an unrelated
  later part. At the default size a short paragraph ahead of a long one came back
  stuck to the closing paragraph.

No mocks: the real ``chunk_text`` on real text, and a real ``LocalKB``.
"""

from __future__ import annotations

import pytest

from fastaiagent.kb import LocalKB
from fastaiagent.kb.chunking import chunk_text
from fastaiagent.kb.embedding import SimpleEmbedder

POLICY = (
    "Refund policy. Items can be sent back within 30 days of purchase for a full refund. "
    "They must be unused and in their original packaging. Digital products are final.\n\n"
    "Shipping. Domestic orders ship in 3-5 business days. International orders take 7-14 "
    "business days. Express shipping can be chosen at checkout for an extra fee.\n\n"
    "Support. Email support@example.com or call 1-800-EXAMPLE. Hours are Monday to Friday, "
    "9am to 5pm EST."
)
LONG = " ".join(
    f"Sentence number {i} explains one more rule about refunds and returns." for i in range(12)
)
SANDWICH = f"Intro paragraph that is short.\n\n{LONG}\n\nClosing paragraph."


def _shared(chunks) -> list[int]:
    return [a.end_char - b.start_char for a, b in zip(chunks, chunks[1:])]


@pytest.mark.parametrize("size", [60, 120, 200, 512])
@pytest.mark.parametrize("text", [POLICY, SANDWICH], ids=["policy", "sandwich"])
def test_every_chunk_fits_and_its_offsets_are_true(text: str, size: int) -> None:
    chunks = chunk_text(text, chunk_size=size, overlap=50)
    assert chunks
    for c in chunks:
        assert len(c.content) <= size
        assert text[c.start_char : c.end_char] == c.content
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_neighbours_share_text_when_overlap_is_set() -> None:
    chunks = chunk_text(POLICY, chunk_size=120, overlap=50)
    assert all(n > 0 for n in _shared(chunks)), _shared(chunks)
    assert all(n <= 50 for n in _shared(chunks)), _shared(chunks)


def test_a_rule_that_straddles_a_seam_is_whole_in_one_chunk() -> None:
    """Without overlap, '30 days … full refund' and 'unused … original packaging'
    land in different chunks; with it, one chunk carries both."""
    without = chunk_text(POLICY, chunk_size=120, overlap=0)
    with_ = chunk_text(POLICY, chunk_size=120, overlap=50)

    def together(chunks) -> bool:
        return any("30 days" in c.content and "original packaging" in c.content for c in chunks)

    assert not together(without)
    assert together(with_)


def test_overlap_zero_keeps_the_seams_disjoint() -> None:
    chunks = chunk_text(POLICY, chunk_size=120, overlap=0)
    assert all(n <= 0 for n in _shared(chunks))


def test_an_overlap_starts_at_a_word() -> None:
    for c in chunk_text(POLICY, chunk_size=120, overlap=50)[1:]:
        assert c.start_char == 0 or POLICY[c.start_char - 1].isspace(), repr(c.content[:20])


def test_no_chunk_sits_entirely_inside_the_next() -> None:
    chunks = chunk_text(POLICY, chunk_size=120, overlap=50)
    for a, b in zip(chunks, chunks[1:]):
        assert not (b.start_char <= a.start_char and a.end_char <= b.end_char)


@pytest.mark.parametrize("overlap", [0, 50])
def test_text_before_a_long_part_is_emitted_once(overlap: int) -> None:
    chunks = chunk_text(SANDWICH, chunk_size=512, overlap=overlap)
    assert sum("Intro paragraph" in c.content for c in chunks) == 1
    # And never glued to a paragraph it was not next to.
    assert not any("Intro paragraph" in c.content and "Closing" in c.content for c in chunks)


def test_no_text_is_lost() -> None:
    for size in (30, 120, 512):
        chunks = chunk_text(POLICY, chunk_size=size, overlap=50)
        covered = set()
        for c in chunks:
            covered |= set(range(c.start_char, c.end_char))
        assert all(i in covered or POLICY[i].isspace() for i in range(len(POLICY)))


def test_an_overlap_larger_than_half_the_size_is_capped() -> None:
    chunks = chunk_text(POLICY, chunk_size=60, overlap=500)
    assert all(len(c.content) <= 60 for c in chunks)
    assert all(n <= 30 for n in _shared(chunks))


def test_a_text_that_fits_is_one_chunk() -> None:
    assert [c.content for c in chunk_text("Short text", chunk_size=100, overlap=50)] == [
        "Short text"
    ]


def test_local_kb_applies_its_chunk_overlap(tmp_path) -> None:
    kb = LocalKB(
        name="overlap",
        path=str(tmp_path),
        embedder=SimpleEmbedder(dimensions=32),
        chunk_size=120,
        chunk_overlap=50,
        search_type="keyword",
        persist=False,
    )
    kb.add(POLICY)
    hits = kb.search("30 days original packaging", top_k=1)
    assert "30 days" in hits[0].chunk.content and "original packaging" in hits[0].chunk.content
