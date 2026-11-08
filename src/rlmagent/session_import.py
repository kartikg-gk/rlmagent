"""Validate a session export and register it under a fresh local identity."""

from pathlib import Path
from uuid import uuid4

from rlmagent.harness.session.index import SessionCatalog
from rlmagent.harness.session.records import SessionMetaRecord
from rlmagent.harness.session.store import deserialize_records, serialize_record
from rlmagent.refinement.store import _lock


def import_session(source: Path, directory: Path) -> str:
    if source.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Session import exceeds the 64 MiB limit")
    records = deserialize_records(source.read_text(encoding="utf-8").split("\n"))
    headers = [record for record in records if isinstance(record, SessionMetaRecord)]
    if len(headers) != 1:
        raise ValueError("Session export needs exactly one metadata root")
    identifiers = {record.id for record in records}
    if len(identifiers) != len(records):
        raise ValueError("Session export contains duplicate record IDs")
    seen = set()
    for record in records:
        if record.parent_id is not None and record.parent_id not in seen:
            raise ValueError("Session export contains an absent or forward parent")
        seen.add(record.id)
        references = [
            getattr(record, "entry_id", None),
            getattr(record, "branch_root_id", None),
            *getattr(record, "replaces_entry_ids", []),
        ]
        if any(reference is not None and reference not in identifiers for reference in references):
            raise ValueError("Session export contains an absent branch or compaction reference")
    identity = uuid4().hex[:12]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{identity}.jsonl"
    with target.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(serialize_record(record))
    catalog = SessionCatalog(directory)
    with _lock(directory / "index.lock"):
        catalog.upsert(
            catalog.prepare(
                session_id=identity,
                vault_path=target,
                cwd=headers[0].cwd,
                provider="imported",
                model="",
                title=f"Imported {source.stem}",
            )
        )
    return identity
