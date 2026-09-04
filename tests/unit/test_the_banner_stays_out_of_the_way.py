"""The mark is decoration, and must never be anything else.

`capcomp pool --json | jq` and `capcomp contract > contract.json` are both
documented, and a banner on stdout breaks them without looking broken: eight
lines of escape codes where a parser expects a document. So these are less
about the banner appearing than about every case where it must not.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from capability_subnet.common import banner

REPO = Path(__file__).resolve().parents[2]


class Terminal(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class Piped(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return False


class Ascii(io.StringIO):
    encoding = "ascii"

    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _plain_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("CAPSUB_NO_BANNER", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")


class TestItStaysQuietWhenSomethingIsReading:
    def test_a_pipe_gets_nothing(self):
        """The case that breaks `| jq`."""
        stream = Piped()
        assert banner.show(stream) is False
        assert stream.getvalue() == ""

    def test_a_terminal_that_cannot_encode_it_gets_nothing(self):
        """Better nothing than a screenful of question marks."""
        assert banner.show(Ascii()) is False

    def test_it_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("CAPSUB_NO_BANNER", "1")
        assert banner.show(Terminal()) is False

    def test_a_stream_that_lies_about_accepting_bytes_is_not_a_traceback(self):
        """A decoration is never worth failing a command over."""

        class Hostile(Terminal):
            def write(self, _data):
                raise UnicodeEncodeError("utf-8", "", 0, 1, "refused")

        assert banner.show(Hostile()) is False


class TestItAppearsWhenSomebodyIsLooking:
    def test_a_terminal_gets_the_mark(self):
        stream = Terminal()
        assert banner.show(stream) is True
        assert "CAPABILITY COMPOSITION SUBNET" in stream.getvalue()

    def test_no_color_is_honoured_and_still_prints(self, monkeypatch):
        """The convention every tool agrees on. Off means uncoloured, not gone."""
        monkeypatch.setenv("NO_COLOR", "1")
        stream = Terminal()

        assert banner.show(stream) is True
        assert "\033[" not in stream.getvalue()

    def test_a_dumb_terminal_gets_no_escape_codes(self, monkeypatch):
        monkeypatch.setenv("TERM", "dumb")
        assert "\033[" not in banner.render()

    @pytest.mark.parametrize("depth", ["truecolor", "256"])
    def test_each_colour_depth_closes_every_line_it_opens(self, depth):
        """An unreset line bleeds its colour into everything printed after."""
        for line in banner.render(depth).split("\n"):
            if "\033[" in line:
                assert line.endswith("\033[0m"), line[:40]


class TestTheDocumentedPipelinesStillWork:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "capability_subnet.miner.cli", *args],
            capture_output=True,
            text=True,
            cwd=REPO,
        )

    def test_pool_json_is_parseable(self):
        assert json.loads(self._run("pool", "--json").stdout)["adapter_count"] > 0

    def test_contract_is_parseable(self):
        assert "base_model" in json.loads(self._run("contract").stdout)

    def test_nothing_reaches_stdout_that_is_not_the_document(self):
        """Captured output is a pipe, so the banner must not be there at all."""
        assert "CAPABILITY COMPOSITION" not in self._run("pool", "--json").stdout
        assert "CAPABILITY COMPOSITION" not in self._run("pool", "--json").stderr


class TestTheReadmeShowsTheSameMark:
    """Two copies of an ASCII drawing drift, and nobody notices which is stale."""

    def test_the_readme_carries_the_art_the_cli_prints(self):
        import re

        readme = (REPO / "README.md").read_text()
        block = re.search(r"```\n( ██████╗.*?)\n```", readme, re.S)
        assert block, "the README no longer shows the mark"

        drawn = [
            line
            for line in block.group(1).split("\n")
            if line.strip() and banner.TAGLINE not in line
        ]
        assert drawn == banner.MARK.strip("\n").split("\n")

    def test_the_readme_carries_the_tagline_too(self):
        assert banner.TAGLINE in (REPO / "README.md").read_text()
