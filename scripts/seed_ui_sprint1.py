"""Seed Sprint 1 fixtures into a snapshot DB.

Runs on top of ``seed_ui_snapshot.py`` and adds:

* a multimodal trace with one image attachment + content parts in input
* a durable execution with three checkpoints (completed / interrupted /
  pending) and one ``@idempotent`` cache row
* an extra cost-bearing trace so the cost breakdown screenshots have data

Used by ``scripts/capture-sprint1-screenshots.sh`` — never run in
production. The DB path comes from ``argv[1]``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _png_thumb_bytes() -> bytes:
    """Generate a tiny solid-color PNG via Pillow if available, else fallback.

    The Local UI's attachment endpoint streams the bytes back as
    ``Content-Type: image/jpeg``. Pillow is already a transitive dep of
    ``fastaiagent[multimodal]``; if missing we fall back to a 1x1
    embedded PNG so the seed still works.
    """
    try:
        from PIL import Image  # type: ignore

        img = Image.new("RGB", (320, 200), (88, 105, 242))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except ImportError:
        # 1x1 transparent PNG, hex-decoded. Tiny but valid.
        return bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63000100000005000148b8770000000049454e44ae426082"
        )


def seed(db_path: Path) -> None:
    now = datetime.now(tz=timezone.utc)
    with SQLiteHelper(db_path) as db:
        _seed_multimodal_trace(db, now)
        _seed_checkpoints(db, now)


def _seed_multimodal_trace(db: SQLiteHelper, now: datetime) -> None:
    """Multimodal trace + attachment so the trace detail page renders inline images."""
    trace_id = "mm00000000000000000000000000mm01"
    span_root = "mmrootspan000000000000000000mm01"
    span_llm = "mmllmspan0000000000000000000mm01"
    start = now - timedelta(minutes=5)
    end = start + timedelta(seconds=3)

    # Embed the real image bytes as a data URL inside the message so the
    # MixedContentView's <img> renders without a server round-trip — the
    # screenshot shows the actual blue rectangle, not a broken-image alt.
    img_bytes = _png_thumb_bytes()
    img_data_url = (
        "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("ascii")
    )

    # Mixed content message (text + image_url) — what extract_content_parts walks.
    user_msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image? Describe in one sentence."},
                {
                    "type": "image_url",
                    "image_url": {"url": img_data_url},
                    "media_type": "image/jpeg",
                    "size_bytes": len(img_bytes),
                    "width": 320,
                    "height": 200,
                },
            ],
        }
    ]

    root_attrs = {
        "agent.name": "vision-bot",
        "agent.input": "What's in this image?",
        "agent.output": "A solid blue rectangle.",
        "agent.tokens_used": 120,
        "agent.latency_ms": 3000,
        "fastaiagent.input.media_count": 1,
    }
    llm_attrs = {
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 20,
        "gen_ai.request.messages": json.dumps(user_msg),
        "gen_ai.response.content": "A solid blue rectangle, roughly 320 by 200 pixels.",
        # Surface the gallery on the LLM span as well so devs can see the
        # ``trace_attachments`` row — it's where the SDK persists the bytes.
        "fastaiagent.input.media_count": 1,
    }

    for sid, parent, name, attrs, status in [
        (span_root, None, "agent.vision-bot", root_attrs, "OK"),
        (span_llm, span_root, "llm.openai.gpt-4o-mini", llm_attrs, "OK"),
    ]:
        db.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]')""",
            (sid, trace_id, parent, name, _iso(start), _iso(end), status, json.dumps(attrs)),
        )

    # Associate the attachment with both the agent root span and the LLM
    # child span. Real SDK runs write the row twice (once when the agent
    # receives the multimodal input and again when the LLM sees it). The
    # screenshot shows the gallery rendering on either side.
    for sid in (span_root, span_llm):
        db.execute(
            """INSERT OR REPLACE INTO trace_attachments
               (attachment_id, trace_id, span_id, media_type, size_bytes,
                thumbnail, full_data, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                trace_id,
                sid,
                "image/jpeg",
                len(img_bytes),
                img_bytes,
                img_bytes,  # store full data so the modal opens
                json.dumps({"width": 320, "height": 200}),
                _iso(now),
            ),
        )


def _seed_checkpoints(db: SQLiteHelper, now: datetime) -> None:
    """Durable Chain execution with mixed checkpoint statuses for the inspector."""
    exec_id = "exec-sprint1-mm-00001"
    chain_name = "support-triage"
    base = now - timedelta(minutes=10)

    rows = [
        # step 0 — research, completed
        (
            uuid.uuid4().hex,
            chain_name,
            exec_id,
            "research",
            0,
            "completed",
            json.dumps({"query": "refund policy", "results": ["doc-1", "doc-2"]}),
            json.dumps({"query": "refund policy"}),
            json.dumps({"results": ["doc-1", "doc-2"]}),
            "",
            "",
            f"chain:{chain_name}",
            _iso(base),
        ),
        # step 1 — approval, interrupted
        (
            uuid.uuid4().hex,
            chain_name,
            exec_id,
            "approval",
            1,
            "interrupted",
            json.dumps(
                {"query": "refund policy", "results": ["doc-1", "doc-2"], "amount": 500}
            ),
            json.dumps({"amount": 500}),
            json.dumps({}),
            "manager_approval",
            json.dumps({"context": {"customer_id": "cust_42", "amount": 500}}),
            f"chain:{chain_name}",
            _iso(base + timedelta(seconds=5)),
        ),
    ]
    for row in rows:
        db.execute(
            """INSERT OR REPLACE INTO checkpoints
               (checkpoint_id, chain_name, execution_id, node_id, node_index,
                status, state_snapshot, node_input, node_output,
                interrupt_reason, interrupt_context, agent_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            row,
        )

    # pending interrupt row so the /approvals page also lights up
    db.execute(
        """INSERT OR REPLACE INTO pending_interrupts
           (execution_id, chain_name, node_id, reason, context, agent_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            exec_id,
            chain_name,
            "approval",
            "manager_approval",
            json.dumps({"customer_id": "cust_42", "amount": 500}),
            f"chain:{chain_name}",
            _iso(base + timedelta(seconds=5)),
        ),
    )

    # idempotency cache rows — show that @idempotent results would be skipped
    for fn_key, result in [
        ("charge_customer:cust_42:500", {"charge_id": "ch_abc"}),
        ("send_notification:alice@x.io:refund", {"sent": True}),
    ]:
        db.execute(
            """INSERT OR REPLACE INTO idempotency_cache
               (execution_id, function_key, result, created_at)
               VALUES (?, ?, ?, ?)""",
            (exec_id, fn_key, json.dumps(result), _iso(base + timedelta(seconds=2))),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=Path)
    args = parser.parse_args()
    seed(args.db_path)
    print(f"sprint1 seed applied to {args.db_path}")


if __name__ == "__main__":
    main()
