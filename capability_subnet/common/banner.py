"""The mark, for a terminal that can show it.

Printed when a miner or a validator starts a command interactively. It is
decoration and it is treated as decoration: it goes to stderr, it is skipped
whenever anything might be reading the output, and nothing anywhere depends on
it having been printed.

That matters more than it sounds. ``capcomp pool --json | jq`` and
``capcomp contract > contract.json`` are documented, and a banner on stdout
breaks both - not visibly, but by putting eight lines of escape codes where a
parser expects a document. So stdout is left alone entirely, and even on stderr
the banner is suppressed unless a person is plainly there to read it.

The colour is a gradient across the columns rather than a fixed palette, so the
mark reads as one object rather than six letters that happen to be adjacent. It
degrades in the order terminals actually vary: truecolour, then 256, then none
at all, and none is a perfectly good answer that several environments give.
"""

from __future__ import annotations

import os
import sys

#: The mark. Box-drawing characters, so it needs a UTF-8 capable terminal -
#: which is checked before printing rather than assumed.
MARK = r"""
 ██████╗ █████╗ ██████╗  ██████╗ ██████╗ ███╗   ███╗██████╗
██╔════╝██╔══██╗██╔══██╗██╔════╝██╔═══██╗████╗ ████║██╔══██╗
██║     ███████║██████╔╝██║     ██║   ██║██╔████╔██║██████╔╝
██║     ██╔══██║██╔═══╝ ██║     ██║   ██║██║╚██╔╝██║██╔═══╝
╚██████╗██║  ██║██║     ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║
 ╚═════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝
"""

TAGLINE = "CAPABILITY COMPOSITION SUBNET"

#: The gradient's ends, as RGB. A cool blue into a warm violet: enough
#: separation to read as a sweep, close enough in luminance that no part of the
#: mark disappears against either a light or a dark terminal.
START = (56, 152, 236)
END = (168, 85, 247)


def _blend(t: float) -> tuple[int, int, int]:
    """The gradient at ``t`` in [0, 1], linearly."""
    return tuple(round(a + (b - a) * t) for a, b in zip(START, END, strict=True))  # type: ignore[return-value]


def _depth() -> str:
    """How much colour this terminal admits: ``truecolor``, ``256`` or ``none``.

    Read from the environment rather than probed. NO_COLOR is honoured because
    it is the one convention every tool agrees on, and a dumb terminal is taken
    at its word.
    """
    if os.environ.get("NO_COLOR") is not None:
        return "none"
    term = os.environ.get("TERM", "")
    if term in ("dumb", ""):
        return "none"
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    return "256" if "256" in term or "color" in term else "none"


def _paint(line: str, depth: str, width: int) -> str:
    """One line of the mark, swept left to right."""
    if depth == "none":
        return line

    out: list[str] = []
    last = ""
    for column, character in enumerate(line):
        if character == " ":
            out.append(character)
            continue
        red, green, blue = _blend(column / max(width - 1, 1))
        code = (
            f"\033[38;2;{red};{green};{blue}m"
            if depth == "truecolor"
            # 6x6x6 cube: the nearest of 216, which is close enough that the
            # sweep still reads as a sweep.
            else f"\033[38;5;{16 + 36 * (red // 43) + 6 * (green // 43) + (blue // 43)}m"
        )
        if code != last:
            out.append(code)
            last = code
        out.append(character)
    return "".join(out) + "\033[0m"


def render(depth: str | None = None) -> str:
    """The banner as a string, coloured for ``depth``.

    Separated from :func:`show` so a test can assert what it produces without
    a terminal, and so anything that wants the mark somewhere else can have it.
    """
    depth = depth if depth is not None else _depth()
    lines = MARK.strip("\n").split("\n")
    width = max(len(line) for line in lines)
    painted = [_paint(line, depth, width) for line in lines]

    tagline = TAGLINE.center(width).rstrip()
    if depth != "none":
        red, green, blue = _blend(0.5)
        tagline = (
            f"\033[38;2;{red};{green};{blue}m{tagline}\033[0m"
            if depth == "truecolor"
            else f"\033[38;5;{16 + 36 * (red // 43) + 6 * (green // 43) + (blue // 43)}m"
            f"{tagline}\033[0m"
        )
    return "\n" + "\n".join(painted) + "\n" + tagline + "\n"


def show(stream=None) -> bool:
    """Print the mark, if there is plainly a person there to see it.

    Returns:
        Whether anything was printed. Callers ignore this; it is here so a test
        can assert the quiet cases stayed quiet.

    Every condition below is a way of being read by something other than a
    person. A redirected stream is a file. A pipe is another program. And a
    terminal that cannot encode the box-drawing characters would print the
    mark as a screenful of question marks, which is worse than printing
    nothing.
    """
    stream = stream if stream is not None else sys.stderr

    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.environ.get("CAPSUB_NO_BANNER"):
        return False

    encoding = (getattr(stream, "encoding", "") or "").lower()
    if "utf" not in encoding:
        return False

    try:
        stream.write(render())
        stream.flush()
    except (OSError, UnicodeEncodeError):
        # A stream that says it is a utf-8 terminal and then refuses the bytes
        # is not worth a traceback over a decoration.
        return False
    return True
