"""The archive documentation states facts a reader will act on.

Repository names, the commitment format and the file list are all things
somebody will type into a terminal or a script. A doc that drifts from the code
sends them somewhere that does not exist and tells them the subnet is broken,
so each claim is pinned to the constant it describes.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from capability_subnet.audit import bundle as audit_bundle
from capability_subnet.common import archive

DOCS = sorted((Path(__file__).resolve().parents[2] / "docs").glob("*.md"))
ARCHIVE_DOCS = [p for p in DOCS if "sn103-run-" in p.read_text()]


def test_the_docs_describe_the_archive_at_all():
    """Guards the rest: these all pass trivially if no doc mentions it."""
    assert ARCHIVE_DOCS, "no document describes the public archive"


@pytest.mark.parametrize("doc", ARCHIVE_DOCS, ids=lambda p: p.name)
def test_the_repository_name_matches_the_auditor(doc):
    """A reader constructs this name by hand; it has to be the real one."""
    published = audit_bundle.REPO_TEMPLATE.format(run="<N>")
    assert published in doc.read_text(), f"{doc.name} does not state {published}"


@pytest.mark.parametrize("doc", ARCHIVE_DOCS, ids=lambda p: p.name)
def test_the_commitment_format_matches_the_encoder(doc):
    """The documented payload shape has to be the one encode() produces."""
    text = doc.read_text()
    if archive.ARCHIVE_MAGIC not in text:
        pytest.skip(f"{doc.name} does not quote the payload")

    real = archive.encode(420, "sha256:" + hashlib.sha256(b"x").hexdigest(), "owner/repo@abc")
    fields = real.split(archive.SEPARATOR)
    assert fields[0] == archive.ARCHIVE_MAGIC

    documented = re.search(rf"{re.escape(archive.ARCHIVE_MAGIC)}\|[^\s`]*", text)
    assert documented, f"{doc.name} mentions the magic but shows no payload"
    assert len(documented.group(0).split(archive.SEPARATOR)) == len(fields)


@pytest.mark.parametrize("doc", ARCHIVE_DOCS, ids=lambda p: p.name)
def test_no_document_promises_a_file_the_bundle_omits(doc):
    """Traces are the one a reader would most reasonably expect to find."""
    text = doc.read_text()
    for absent in ("instances.csv", "candidates.csv", "weights.json"):
        for line in text.splitlines():
            if absent in line and "sn103-run" in text:
                assert "not" in line.lower() or "absent" in line.lower(), (
                    f"{doc.name} mentions {absent} without saying it is not published: {line}"
                )


def test_the_audit_command_the_docs_name_exists():
    """Every doc points at this command; a missing one is a dead end."""
    from capability_subnet.audit.cli import build_parser

    parser = build_parser()
    action = next(
        a for a in parser._actions if getattr(a, "choices", None) and "bundle" in a.choices
    )
    assert "bundle" in action.choices

    for doc in ARCHIVE_DOCS:
        text = doc.read_text()
        if "audit.cli" in text or "capability-audit" in text:
            assert "bundle --run" in text, f"{doc.name} points at no bundle audit"
