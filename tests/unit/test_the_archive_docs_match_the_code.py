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
    """A reader who fetches an archive must find what the table said was in it.

    Traces are published as ``instances.csv.gz``; the CSV of scores and the
    separate weight vector are not, because everything a grade depends on is in
    scores.json. Naming an omitted file sends someone looking for it.
    """
    text = doc.read_text()
    if "sn103-run" not in text:
        return
    for absent in ("candidates.csv", "weights.json"):
        for line in text.splitlines():
            if absent in line:
                assert "not" in line.lower() or "absent" in line.lower(), (
                    f"{doc.name} mentions {absent} without saying it is not published: {line}"
                )


@pytest.mark.parametrize("doc", ARCHIVE_DOCS, ids=lambda p: p.name)
def test_every_doc_names_the_traces_the_bundle_carries(doc):
    """The reverse failure: a bundle carrying traces no document mentions.

    They are the part that lets a score be checked against work rather than
    against its own arithmetic, so a reader who never learns they are there
    audits less than they could.
    """
    assert "instances.csv.gz" in doc.read_text(), (
        f"{doc.name} does not tell a reader the traces are published"
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


class TestTheDocsOnlyNameThingsThatExist:
    """A documented command that does not exist is worse than no document.

    A reader who types it concludes the subnet is broken, and the one place
    that matters most is the verification path — somebody checking whether to
    trust this network.
    """

    @pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
    def test_no_document_names_an_install_that_does_not_resolve(self, doc):
        """`pip install capability-subnet` needs the package to be on an index.

        It is not published to one, so the documented install is the editable
        install from a clone, which is what every other document already says.
        """
        assert "pip install capability-subnet\n" not in doc.read_text(), (
            f"{doc.name} tells a reader to install from an index the package is not published to"
        )

    @pytest.mark.parametrize("doc", ARCHIVE_DOCS, ids=lambda p: p.name)
    def test_every_console_script_a_doc_names_is_declared(self, doc):
        """Catches a renamed entry point before a reader does."""
        import re
        from importlib.metadata import entry_points

        # The installed entry points rather than the source declaration. A doc
        # names a command a reader will type, and what they can type is what
        # was installed — a pyproject edited without a reinstall would satisfy
        # a source check and still fail the reader.
        declared = {e.name for e in entry_points(group="console_scripts")}

        named = set(re.findall(r"^\s*(capcomp|capability-[a-z]+)\b", doc.read_text(), re.M))
        assert named <= declared, f"{doc.name} names undeclared script(s): {named - declared}"
