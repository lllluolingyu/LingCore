"""Terminal commands, styles, and rendering limits."""

from __future__ import annotations

import re

from prompt_toolkit.styles import Style
from rich.theme import Theme

_EXIT_COMMANDS = {"/exit", "/quit", "/q"}
_COMMANDS = ("/new", "/sessions", "/resume", "/usage", "/help", "/exit", "/quit")
_HELP_COMMANDS = (
    ("/new", "start a fresh session"),
    ("/sessions", "list stored sessions for this profile"),
    ("/resume <id>", "switch to a stored session by id prefix"),
    ("/usage", "token usage for this session"),
    ("/help", "show this help"),
    ("/exit", "quit (also /quit, /q, Ctrl-D)"),
)
_HELP_KEYS = (
    ("Enter", "send the message"),
    ("Shift/Alt-Enter", "new line (also Ctrl-J, or \\ then Enter)"),
    ("Tab", "complete /commands and @paths"),
    ("↑ / ↓", "move between lines, then browse input history"),
    ("Ctrl-C", "clear the input; stop a running turn (twice quits)"),
    ("Ctrl-D", "quit"),
    ("@path", "attach a file to the message"),
)
_ARGUMENT_COMMANDS = {"/resume"}
_COMMAND_META = {key.split()[0]: text for key, text in _HELP_COMMANDS} | {
    "/quit": "quit"
}
# Diff previews for file edits stay short; the full change is in the workspace.
_DIFF_PREVIEW_LINES = 40
_ATTACH_RE = re.compile(r'(?<!\S)@("[^"]+"|\S+)')
# Argument keys that best identify what a tool call acts on, in priority order.
_HEADLINE_KEYS = ("command", "path", "url", "query", "pattern", "name", "action")
_ERROR_PREVIEW_LINES = 4
# Markdown styles that read on both dark and light terminals (Rich's defaults
# paint inline code on a black background).
_THEME = Theme(
    {
        "markdown.code": "bold cyan",
        "markdown.code_block": "cyan",
        "markdown.h1": "bold underline",
        "markdown.h2": "bold",
        "markdown.h3": "bold",
        "markdown.h4": "bold italic",
        "markdown.block_quote": "italic dim",
        "markdown.hr": "dim",
    }
)
_PROMPT = "\n[bold cyan]❯[/] "
_CONTINUATION_PROMPT = "[dim]·[/] "
_PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "rule": "#5f5f5f",
        "continuation": "#5f5f5f",
        "bottom-toolbar": "noreverse",
        "status": "#8a8a8a",
        "status.agent": "ansicyan bold",
        "status.dim": "#5f5f5f",
        "status.key": "ansicyan",
    }
)
# Shift+Enter has no single encoding. Terminals using the CSI-u or xterm
# modifyOtherKeys schemes send these, which prompt_toolkit would otherwise read
# as plain Enter (submit); route them to Alt+Enter, which inserts a newline.
_SHIFT_ENTER_SEQUENCES = ("\x1b[13;2u", "\x1b[27;2;13~")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")

# Commands whose confirmations need shell framing.
COMMAND_TOOLS = frozenset({"run_shell", "process_start", "process_monitor"})
