#!/usr/bin/env python3
"""Interactive zsh history search with Emacs-like flex matching."""

from __future__ import annotations

import os
import json
import queue
import re
import select
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import termios
import threading
import time
import tempfile
import tty
from datetime import datetime, timezone
import unicodedata
from argparse import SUPPRESS, ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from zsh_syntax_highlighting import ansi_for_token, highlight_tokens


BASE16_TO_ANSI = {
    "base00": 0,
    "base01": 8,
    "base02": 0,
    "base03": 8,
    "base04": 7,
    "base05": 7,
    "base06": 15,
    "base07": 15,
    "base08": 1,
    "base09": 9,
    "base0A": 3,
    "base0B": 2,
    "base0C": 6,
    "base0D": 4,
    "base0E": 5,
    "base0F": 9,
}

DORIC = {
    "cursor": "#2266bb",
    "bg_main": "#ffffff",
    "fg_main": "#000000",
    "border": "#b0b0b0",
    "bg_shadow_subtle": "#efeff2",
    "fg_shadow_subtle": "#5a6268",
    "bg_neutral": "#dbdce1",
    "fg_neutral": "#424d53",
    "bg_shadow_intense": "#a0bcd0",
    "fg_shadow_intense": "#213067",
    "bg_accent": "#d8f1f3",
    "fg_accent": "#084092",
    "fg_red": "#a01010",
    "fg_green": "#106710",
    "fg_yellow": "#60400f",
    "fg_blue": "#103077",
    "fg_magenta": "#700d50",
    "fg_cyan": "#005355",
    "bg_red": "#f0c4c4",
    "bg_green": "#c0e8c2",
    "bg_yellow": "#f0f0b0",
    "bg_blue": "#c4cfe8",
    "bg_magenta": "#eec2e6",
    "bg_cyan": "#c1ebe4",
}


@dataclass
class MatchResult:
    text: str
    score: int
    positions: List[int]
    exact: bool = False
    recency: int = 0
    cwd: Optional[str] = None
    text_lower: Optional[str] = None


@dataclass
class HistoryEntry:
    text: str
    cwd: Optional[str] = None
    text_lower: str = ""
    timestamp: Optional[str] = None


def base16_ansi(name: str) -> int:
    return BASE16_TO_ANSI[name]


def fg_code(slot: int) -> str:
    if 0 <= slot <= 7:
        return str(30 + slot)
    if 8 <= slot <= 15:
        return str(90 + (slot - 8))
    return "39"


def bg_code(slot: int) -> str:
    if 0 <= slot <= 7:
        return str(40 + slot)
    if 8 <= slot <= 15:
        return str(100 + (slot - 8))
    return "49"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def style(
    *,
    fg: Optional[int] = None,
    bg: Optional[int] = None,
    fg_rgb: Optional[str] = None,
    bg_rgb: Optional[str] = None,
    bold: bool = False,
    underline: bool = False,
) -> str:
    codes: list[str] = []
    if bold:
        codes.append("1")
    if underline:
        codes.append("4")
    if fg is not None:
        codes.append(fg_code(fg))
    if bg is not None:
        codes.append(bg_code(bg))
    if fg_rgb is not None:
        r, g, b = hex_to_rgb(fg_rgb)
        codes.extend(["38", "2", str(r), str(g), str(b)])
    if bg_rgb is not None:
        r, g, b = hex_to_rgb(bg_rgb)
        codes.extend(["48", "2", str(r), str(g), str(b)])
    if not codes:
        return ""
    return f"\x1b[{';'.join(codes)}m"


RESET = "\x1b[0m"
QUERY_SELECTION_BG = style(fg_rgb=DORIC["fg_blue"], bg_rgb=DORIC["bg_blue"])
CLEAR_LINE = "\x1b[2K"
CLEAR_TO_END = "\x1b[K"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
ENABLE_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1006h"
DISABLE_MOUSE = "\x1b[?1000l\x1b[?1002l\x1b[?1006l"
MAX_RETURNED_RESULTS = 100

TERM_OUT = sys.stdout


def move_to(row: int, col: int = 1) -> str:
    return f"\x1b[{max(1, row)};{max(1, col)}H"


def term_write(text: str) -> None:
    TERM_OUT.write(text)


def term_flush() -> None:
    TERM_OUT.flush()


def clear_rows(top: int, bottom: int) -> None:
    if bottom < top:
        return
    for row in range(max(1, top), max(1, bottom) + 1):
        term_write(move_to(row, 1) + CLEAR_LINE)


def tty_terminal_size(fd: int, fallback: tuple[int, int] = (120, 24)) -> os.terminal_size:
    try:
        return os.get_terminal_size(fd)
    except OSError:
        size = shutil.get_terminal_size(fallback)
        return os.terminal_size((max(1, size.columns), max(1, size.lines)))


def write_clipboard(text: str) -> bool:
    if shutil.which("pbcopy") is None:
        return False
    try:
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def read_clipboard() -> str:
    if shutil.which("pbpaste") is None:
        return ""
    try:
        proc = subprocess.run(["pbpaste"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.replace("\r\n", "\n").replace("\r", "\n")


def normalize_pasted_text(text: str) -> str:
    # Keep multiline content, but strip terminal control artifacts.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    # Drop CSI sequences (including stray bracketed-paste/mouse reports).
    normalized = re.sub(r"\x1b\[[0-9;?<>]*[ -/]*[@-~]", "", normalized)
    # Drop leaked bracketed-paste markers even if ESC got stripped.
    normalized = normalized.replace("200~", "").replace("201~", "")
    # Drop leaked SGR mouse payloads when ESC is missing.
    normalized = re.sub(r"<\d+;\d+;\d+[mM]", "", normalized)
    return normalized


def normalize_shell_command(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    cleaned = cleaned.replace("\\\n", "")
    return cleaned.strip("\n")


class RawTerminal:
    def __init__(self, fd: int) -> None:
        self.fd = fd
        self._old: Optional[list] = None

    def __enter__(self) -> "RawTerminal":
        self._old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        try:
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except termios.error:
            pass
        # Start with mouse reporting disabled; it will be enabled lazily
        # once the user types the first character in the query.
        term_write(DISABLE_MOUSE)
        # Keep the terminal's default cursor visible while editing input.
        term_write(SHOW_CURSOR)
        term_flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        term_write(DISABLE_MOUSE + SHOW_CURSOR + RESET)
        term_flush()
        if self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)


def query_cursor_position(fd: int) -> Optional[tuple[int, int]]:
    # Drain any stale input bytes so we do not parse an old cursor response.
    while True:
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            break
        try:
            os.read(fd, 4096)
        except OSError:
            break

    term_write("\x1b[6n")
    term_flush()
    buf = b""
    deadline = time.monotonic() + 0.2
    last_match: Optional[tuple[int, int]] = None
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.02)
        if not ready:
            continue
        buf += os.read(fd, 64)
        for m in re.finditer(rb"\x1b\[(\d+);(\d+)R", buf):
            last_match = (int(m.group(1)), int(m.group(2)))
        if last_match is not None:
            # Return as soon as we have a valid cursor report instead of
            # waiting out the full timeout on every startup.
            break
    return last_match


def normalize_cwd_value(cwd: str) -> str:
    stripped = cwd.strip()
    if not stripped:
        return ""
    return os.path.normpath(stripped)


def make_history_entry(text: str, *, cwd: Optional[str] = None, timestamp: Optional[str] = None) -> HistoryEntry:
    return HistoryEntry(text=text, cwd=cwd, text_lower=text.lower(), timestamp=timestamp)


def load_history(path: Path) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    if not path.exists():
        return entries

    raw = path.read_text(encoding="utf-8", errors="replace")
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Support plain history and extended history in the same file.
    # Extended entry format:
    #   : 1700012345:0;command
    header_line_re = re.compile(r"^: \d+:\d+;(.*)$")
    current_extended: Optional[str] = None

    def push_entry(text: str) -> None:
        cmd = text.rstrip("\n").replace("\\\n", "").strip()
        if cmd:
            entries.append(make_history_entry(cmd))

    for line in normalized.split("\n"):
        match = header_line_re.match(line)
        if match:
            if current_extended is not None:
                push_entry(current_extended)
            current_extended = match.group(1)
            continue

        if current_extended is not None:
            current_extended += "\n" + line
            continue

        plain = line.strip()
        if plain:
            entries.append(make_history_entry(plain))

    if current_extended is not None:
        push_entry(current_extended)

    # Preserve recency ordering (newest first), then remove duplicate command
    # text while keeping the newest occurrence of each command.
    newest_first = list(reversed(entries))
    return dedupe_history_entries_preserving_order(newest_first)


def dedupe_history_entries_preserving_order(entries: list[HistoryEntry]) -> list[HistoryEntry]:
    deduped: list[HistoryEntry] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.text in seen:
            continue
        seen.add(entry.text)
        deduped.append(entry)
    return deduped


def default_custom_history_path() -> Path:
    return Path(__file__).resolve().parent / "history.db"


def parse_history_length_arg(raw: str) -> int:
    value = raw.strip().lower().replace("_", "")
    match = re.fullmatch(r"(\d+)([km]?)", value)
    if match is None:
        raise ValueError(f"invalid history length: {raw!r}")
    count = int(match.group(1))
    suffix = match.group(2)
    if suffix == "k":
        count *= 1_000
    elif suffix == "m":
        count *= 1_000_000
    if count <= 0:
        raise ValueError("history length must be positive")
    return count


def ensure_custom_history_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS custom_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                cwd TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_custom_history_command_cwd ON custom_history(command, cwd)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_custom_history_id_desc ON custom_history(id DESC)"
        )
        conn.commit()


def load_custom_history_rows(path: Path, *, limit: Optional[int] = None) -> list[HistoryEntry]:
    if not path.exists():
        return []
    query = "SELECT command, cwd, timestamp FROM custom_history ORDER BY id DESC"
    params: tuple[object, ...] = ()
    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params = (limit,)
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(query, params).fetchall()
    except (OSError, sqlite3.Error):
        return []
    entries: list[HistoryEntry] = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) < 3:
            continue
        cmd = row[0]
        cwd = row[1]
        timestamp = row[2]
        if not isinstance(cmd, str):
            continue
        normalized_cwd = normalize_cwd_value(cwd) if isinstance(cwd, str) else ""
        normalized_timestamp = timestamp if isinstance(timestamp, str) else None
        cleaned = cmd.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip("\n")
        if cleaned.strip():
            entries.append(make_history_entry(cleaned, cwd=normalized_cwd or None, timestamp=normalized_timestamp))
    return entries


def load_custom_history(path: Path, *, existing_history: Optional[list[HistoryEntry]] = None) -> list[HistoryEntry]:
    if not existing_history:
        return load_custom_history_rows(path)

    recent_entries = load_custom_history_rows(path, limit=10)
    if not recent_entries:
        return []

    existing_keys = {
        (entry.timestamp, entry.text, entry.cwd)
        for entry in existing_history
        if entry.timestamp is not None
    }
    overlap_at: Optional[int] = None
    for idx, entry in enumerate(recent_entries):
        if (entry.timestamp, entry.text, entry.cwd) in existing_keys:
            overlap_at = idx
            break

    if overlap_at is None:
        return load_custom_history_rows(path)

    newer_entries = [
        entry
        for entry in recent_entries[:overlap_at]
        if (entry.timestamp, entry.text, entry.cwd) not in existing_keys
    ]
    if not newer_entries:
        return existing_history

    replaced_pairs = {(entry.text, entry.cwd) for entry in newer_entries}
    merged_history = [entry for entry in existing_history if (entry.text, entry.cwd) not in replaced_pairs]
    return newer_entries + merged_history


def load_history_source(
    path: Path,
    *,
    use_custom_history: bool,
    existing_history: Optional[list[HistoryEntry]] = None,
) -> list[HistoryEntry]:
    if use_custom_history:
        return load_custom_history(path, existing_history=existing_history)
    return load_history(path)


def append_custom_history_entry(path: Path, command: str, cwd: str, timestamp: str) -> bool:
    normalized_command = command.strip()
    normalized_cwd = normalize_cwd_value(cwd)
    if not normalized_command:
        return False
    try:
        ensure_custom_history_file(path)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "DELETE FROM custom_history WHERE command = ? AND cwd = ?",
                (normalized_command, normalized_cwd),
            )
            conn.execute(
                "INSERT INTO custom_history(command, cwd, timestamp) VALUES(?, ?, ?)",
                (normalized_command, normalized_cwd, timestamp),
            )
            conn.commit()
    except (OSError, sqlite3.Error):
        return False
    return True


def trim_custom_history(path: Path, max_entries: int) -> None:
    if max_entries <= 0:
        return
    ensure_custom_history_file(path)
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM custom_history").fetchone()
        total_entries = row[0] if isinstance(row, tuple) and row and isinstance(row[0], int) else 0
        if total_entries <= max_entries:
            return
        conn.execute(
            """
            DELETE FROM custom_history
            WHERE id NOT IN (
                SELECT id
                FROM custom_history
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (max_entries,),
        )
        conn.commit()


def spawn_history_loader(path: Path) -> queue.Queue[tuple[str, object]]:
    updates: queue.Queue[tuple[str, object]] = queue.Queue()

    def _load() -> None:
        try:
            loaded = load_history(path)
            updates.put(("loaded", loaded))
        except Exception as exc:
            updates.put(("error", exc))

    thread = threading.Thread(target=_load, daemon=True, name="history-loader")
    thread.start()
    return updates


def default_daemon_socket_path(*, use_custom_history: bool = False) -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        base_dir = Path(runtime_dir)
    else:
        base_dir = Path(tempfile.gettempdir())
    suffix = "-custom" if use_custom_history else ""
    return base_dir / f"zsh-flex-history-{os.getuid()}{suffix}.sock"


def history_file_signature(path: Path) -> tuple[int, int]:
    try:
        st = path.stat()
    except OSError:
        return (0, 0)
    return (st.st_mtime_ns, st.st_size)


def daemon_debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[zsh_flex_history daemon] {message}", file=sys.stderr)


def flex_match(query: str, candidate: str, *, candidate_lower: Optional[str] = None) -> Optional[MatchResult]:
    if not query:
        return MatchResult(candidate, 0, [], text_lower=candidate_lower or candidate.lower())

    q = "".join(ch for ch in query.lower() if not ch.isspace())
    c = candidate_lower if candidate_lower is not None else candidate.lower()
    if not q:
        return MatchResult(candidate, 0, [], text_lower=c)

    positions: list[int] = []
    at = 0
    for ch in q:
        idx = c.find(ch, at)
        if idx == -1:
            return None
        positions.append(idx)
        at = idx + 1

    # Approximate Emacs flex behavior: in-order match with strong preference
    # for contiguous runs, token boundaries, and earlier starts.
    score = 0
    contiguous = 0
    gap_penalty = 0
    boundary_bonus = 0

    for i, pos in enumerate(positions):
        if i == 0:
            if pos == 0:
                boundary_bonus += 12
            elif pos > 0 and candidate[pos - 1] in " _-/.:":
                boundary_bonus += 8
            continue

        prev = positions[i - 1]
        gap = pos - prev - 1
        gap_penalty += gap * 2
        if gap == 0:
            contiguous += 10
        if candidate[pos - 1] in " _-/.:":
            boundary_bonus += 6

    span = positions[-1] - positions[0] + 1
    start_bonus = max(0, 30 - positions[0])
    compact_bonus = max(0, 20 - (span - len(q)))

    score += contiguous + boundary_bonus + start_bonus + compact_bonus
    score -= gap_penalty
    score -= len(candidate) // 8

    return MatchResult(candidate, score, positions, text_lower=c)


def token_bounds(query: str, cursor_pos: int) -> tuple[int, int]:
    cursor = max(0, min(cursor_pos, len(query)))
    tokens: list[tuple[int, int]] = []
    i = 0
    while i < len(query):
        while i < len(query) and query[i].isspace():
            i += 1
        if i >= len(query):
            break
        start = i
        quote: Optional[str] = None
        escaped = False
        while i < len(query):
            ch = query[i]
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\":
                escaped = True
                i += 1
                continue
            if quote is not None:
                if ch == quote:
                    quote = None
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
                i += 1
                continue
            if ch.isspace():
                break
            i += 1
        end = i
        tokens.append((start, end))
    for start, end in tokens:
        if start <= cursor <= end:
            return start, end
    return cursor, cursor


def strip_enclosing_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    if token.startswith(("'", '"')):
        return token[1:]
    if token.endswith(("'", '"')):
        return token[:-1]
    return token


def enclosing_quote(token: str) -> tuple[Optional[str], bool]:
    if not token:
        return None, False
    if token[0] not in ("'", '"'):
        return None, False
    quote = token[0]
    return quote, len(token) > 1 and token[-1] == quote


def shell_unescape_fragment(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            i += 1
            out.append(text[i])
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def shell_escape_fragment(text: str) -> str:
    escaped: list[str] = []
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/~")
    for ch in text:
        if ch in safe:
            escaped.append(ch)
        else:
            escaped.append("\\" + ch)
    return "".join(escaped)


def command_token_for_query(query: str) -> Optional[str]:
    stripped = query.strip()
    if not stripped:
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    if not tokens:
        return None
    return tokens[0]


def shell_words_for_matching(text: str) -> list[str]:
    stripped = text.strip().lower()
    if not stripped:
        return []
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    return [token for token in tokens if token]


def path_completion_replacements(
    query: str,
    cursor_pos: int,
    *,
    cwd: Optional[Path] = None,
    dirs_only: bool = False,
    limit: int = 64,
) -> list[str]:
    start, end = token_bounds(query, cursor_pos)
    raw_token = query[start:end]
    stripped = strip_enclosing_quotes(raw_token)
    if not stripped:
        return []

    unescaped = shell_unescape_fragment(stripped)
    working_dir = cwd or Path.cwd()

    if unescaped.endswith("/"):
        parent_part = unescaped[:-1]
        name_prefix = ""
    else:
        parent_part, sep, name_prefix = unescaped.rpartition("/")
        if not sep:
            parent_part = ""

    if unescaped.startswith("/"):
        base_dir = Path(parent_part) if parent_part else Path("/")
        display_prefix = parent_part if parent_part else "/"
    elif unescaped.startswith("~"):
        full = Path(unescaped).expanduser()
        if unescaped.endswith("/"):
            base_dir = full
            name_prefix = ""
        else:
            base_dir = full.parent
            name_prefix = full.name
        display_prefix = str(Path(unescaped).parent)
        if display_prefix == ".":
            display_prefix = "~"
    else:
        rel_parent = Path(parent_part) if parent_part else Path(".")
        base_dir = (working_dir / rel_parent).resolve()
        display_prefix = parent_part

    if not base_dir.exists() or not base_dir.is_dir():
        return []

    replacements: list[str] = []
    quote, has_closing_quote = enclosing_quote(raw_token)
    direct_matches: list[tuple[str, Path, bool]] = []

    def format_token(token: str) -> str:
        if quote is not None:
            inner = token.replace("\\", "\\\\")
            if quote == '"':
                inner = inner.replace('"', '\\"')
            else:
                inner = inner.replace("'", "\\'")
            return quote + inner + (quote if has_closing_quote else "")
        return shell_escape_fragment(token)

    def push_replacement(token: str) -> None:
        replacement = query[:start] + format_token(token) + query[end:]
        replacements.append(replacement)

    try:
        with os.scandir(base_dir) as entries:
            for entry in entries:
                if entry.name.startswith(".") and not name_prefix.startswith("."):
                    continue
                if not entry.name.startswith(name_prefix):
                    continue
                if dirs_only and not entry.is_dir(follow_symlinks=False):
                    continue
                suggested_piece = entry.name
                is_dir = entry.is_dir(follow_symlinks=False)
                if is_dir:
                    suggested_piece += "/"
                if display_prefix == "":
                    completed_token = suggested_piece
                elif display_prefix == ".":
                    completed_token = "./" + suggested_piece
                elif display_prefix == "/":
                    completed_token = "/" + suggested_piece
                else:
                    completed_token = display_prefix.rstrip("/") + "/" + suggested_piece
                direct_matches.append((completed_token, Path(entry.path), is_dir))
    except OSError:
        return []

    direct_matches.sort(key=lambda item: item[0])
    for completed_token, _, _ in direct_matches:
        push_replacement(completed_token)
        if len(replacements) >= limit:
            return replacements

    dir_matches = [item for item in direct_matches if item[2]]
    if len(dir_matches) == 1:
        matched_token, matched_path, _ = dir_matches[0]
        try:
            child_dirs: list[str] = []
            with os.scandir(matched_path) as child_entries:
                for child in child_entries:
                    if not child.is_dir(follow_symlinks=False):
                        continue
                    child_dirs.append(child.name)
            child_dirs.sort()
            for child_name in child_dirs:
                descendant_token = matched_token.rstrip("/") + "/" + child_name + "/"
                push_replacement(descendant_token)
                if len(replacements) >= limit:
                    break
        except OSError:
            pass

    return replacements


_PATH_COMMANDS_CACHE: Optional[list[str]] = None


def path_commands() -> list[str]:
    global _PATH_COMMANDS_CACHE
    if _PATH_COMMANDS_CACHE is not None:
        return _PATH_COMMANDS_CACHE

    commands: set[str] = set()
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not path_dir:
            continue
        try:
            with os.scandir(path_dir) as entries:
                for entry in entries:
                    try:
                        if entry.is_file() and os.access(entry.path, os.X_OK):
                            commands.add(entry.name)
                    except OSError:
                        continue
        except OSError:
            continue
    _PATH_COMMANDS_CACHE = sorted(commands)
    return _PATH_COMMANDS_CACHE


def command_completion_replacements(query: str, cursor_pos: int, *, limit: int = 64) -> list[str]:
    stripped = query.lstrip()
    if not stripped:
        return []
    leading_ws = len(query) - len(stripped)
    first_end = leading_ws
    while first_end < len(query) and not query[first_end].isspace():
        first_end += 1
    if not (leading_ws <= cursor_pos <= first_end):
        return []

    prefix = query[leading_ws:first_end]
    bare_prefix = strip_enclosing_quotes(prefix)
    if not bare_prefix:
        return []

    replacements: list[str] = []
    for cmd in path_commands():
        if not cmd.startswith(bare_prefix):
            continue
        replacement = query[:leading_ws] + cmd + query[first_end:]
        replacements.append(replacement)
        if len(replacements) >= limit:
            break
    return replacements


def cursor_in_first_token(query: str, cursor_pos: int) -> bool:
    stripped = query.lstrip()
    if not stripped:
        return False
    leading_ws = len(query) - len(stripped)
    first_end = leading_ws
    while first_end < len(query) and not query[first_end].isspace():
        first_end += 1
    return leading_ws <= cursor_pos <= first_end


def resolve_runtime_matches(
    query: str,
    cursor_pos: int,
    *,
    cwd: Optional[Path] = None,
    limit: int = 64,
) -> list[MatchResult]:
    if not query.strip():
        return []
    if not cursor_in_first_token(query, cursor_pos):
        return []

    command = command_token_for_query(query) or ""
    dirs_only = command == "cd"
    runtime_candidates: list[str] = []
    runtime_candidates.extend(
        path_completion_replacements(
            query,
            cursor_pos,
            cwd=cwd,
            dirs_only=dirs_only,
            limit=limit,
        )
    )
    runtime_candidates.extend(command_completion_replacements(query, cursor_pos, limit=limit))

    seen: set[str] = set()
    resolved: list[MatchResult] = []
    for candidate in runtime_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        candidate_lower = candidate.lower()
        matched = flex_match(query, candidate, candidate_lower=candidate_lower)
        positions = matched.positions if matched is not None else []
        score = matched.score if matched is not None else -10_000
        resolved.append(MatchResult(candidate, score, positions, text_lower=candidate_lower))
        if len(resolved) >= limit:
            break
    return resolved


def dedupe_match_results_preserving_order(results: list[MatchResult]) -> list[MatchResult]:
    deduped: list[MatchResult] = []
    seen: set[str] = set()
    for item in results:
        if item.text in seen:
            continue
        seen.add(item.text)
        deduped.append(item)
    return deduped


def search_history_only(
    query: str,
    history: list[HistoryEntry],
    *,
    candidate_indices: Optional[list[int]] = None,
    limit: Optional[int] = None,
) -> tuple[list[MatchResult], list[int]]:
    candidates: range | list[int]
    if candidate_indices is None:
        candidates = range(len(history))
    else:
        candidates = candidate_indices
    result_limit = limit if (limit is None or limit > 0) else None
    if not query:
        results: list[MatchResult] = []
        if result_limit is None:
            source = candidates
        elif candidate_indices is None:
            source = range(min(result_limit, len(history)))
        else:
            source = candidate_indices[:result_limit]
        for idx in source:
            entry = history[idx]
            results.append(
                MatchResult(
                    entry.text,
                    0,
                    [],
                    exact=False,
                    recency=-idx,
                    cwd=entry.cwd,
                    text_lower=entry.text_lower,
                )
            )
        if candidate_indices is None:
            return results, list(range(len(history)))
        return results, candidate_indices

    matched_indices: list[int] = []
    history_results: list[MatchResult] = []
    normalized_query = query.strip().lower()

    for idx in candidates:
        entry = history[idx]
        cmd = entry.text
        m = flex_match(query, cmd, candidate_lower=entry.text_lower)
        if m is None:
            continue

        matched_indices.append(idx)

        m.exact = bool(normalized_query) and cmd.strip().lower() == normalized_query
        m.recency = -idx
        m.cwd = entry.cwd
        m.text_lower = entry.text_lower
        history_results.append(m)

    if result_limit is not None:
        history_results = history_results[:result_limit]
    return history_results, matched_indices


def prefer_current_cwd(
    results: list[MatchResult],
    *,
    current_cwd: Optional[str],
) -> list[MatchResult]:
    if not current_cwd:
        return list(results)
    same_cwd: list[MatchResult] = []
    other: list[MatchResult] = []
    for item in results:
        if item.cwd == current_cwd:
            same_cwd.append(item)
        else:
            other.append(item)
    return same_cwd + other


def ordered_query_word_positions(query: str, text_lower: str) -> Optional[list[int]]:
    words = query.lower().split()
    if not words:
        return None

    at = 0
    positions: list[int] = []
    for word in words:
        idx = text_lower.find(word, at)
        if idx == -1:
            return None
        positions.extend(range(idx, idx + len(word)))
        at = idx + len(word)
    return positions


def apply_inner_bucket_priority(
    query: str,
    results: list[MatchResult],
    *,
    current_cwd: Optional[str],
) -> list[MatchResult]:
    words_in_order: list[MatchResult] = []
    rest: list[MatchResult] = []
    for item in results:
        text_lower = item.text_lower if item.text_lower is not None else item.text.lower()
        if ordered_query_word_positions(query, text_lower) is not None:
            words_in_order.append(item)
        else:
            rest.append(item)
    return prefer_current_cwd(words_in_order, current_cwd=current_cwd) + prefer_current_cwd(
        rest,
        current_cwd=current_cwd,
    )


def apply_prefix_priority(
    query: str,
    results: list[MatchResult],
    *,
    limit: Optional[int] = None,
    current_cwd: Optional[str] = None,
) -> list[MatchResult]:
    result_limit = limit if (limit is None or limit > 0) else None
    if not results:
        return results

    query_words = shell_words_for_matching(query)
    prefix_word_counts: list[int] = []
    max_prefix_words = 0
    if query_words:
        for item in results:
            text_lower = item.text_lower if item.text_lower is not None else item.text.lower()
            candidate_words = shell_words_for_matching(text_lower)
            matched_words = 0
            for query_word, candidate_word in zip(query_words, candidate_words):
                if not candidate_word.startswith(query_word):
                    break
                matched_words += 1
            prefix_word_counts.append(matched_words)
            if matched_words > max_prefix_words:
                max_prefix_words = matched_words
    else:
        prefix_word_counts = [0] * len(results)

    if max_prefix_words > 0:
        tier_prefix: list[MatchResult] = []
        tier_rest: list[MatchResult] = []
        for item, prefix_word_count in zip(results, prefix_word_counts):
            if prefix_word_count == max_prefix_words:
                tier_prefix.append(item)
            else:
                tier_rest.append(item)
        ordered_results = apply_inner_bucket_priority(
            query,
            tier_prefix,
            current_cwd=current_cwd,
        ) + apply_inner_bucket_priority(
            query,
            tier_rest,
            current_cwd=current_cwd,
        )
    else:
        ordered_results = apply_inner_bucket_priority(query, results, current_cwd=current_cwd)
    ordered_results = dedupe_match_results_preserving_order(ordered_results)
    if result_limit is not None:
        return ordered_results[:result_limit]
    return ordered_results


def merge_runtime_and_rank(
    query: str,
    cursor_pos: int,
    history_results: list[MatchResult],
    *,
    limit: Optional[int] = None,
    cwd: Optional[Path] = None,
) -> list[MatchResult]:
    regular_results = list(history_results)
    current_cwd = normalize_cwd_value(str(cwd)) if cwd is not None else ""

    seen = {item.text for item in regular_results}
    for runtime_match in resolve_runtime_matches(query, cursor_pos, cwd=cwd):
        if runtime_match.text in seen:
            continue
        regular_results.append(runtime_match)

    return apply_prefix_priority(query, regular_results, limit=limit, current_cwd=current_cwd or None)


def search(
    query: str,
    history: list[HistoryEntry],
    *,
    cursor_pos: int = 0,
    candidate_indices: Optional[list[int]] = None,
    limit: Optional[int] = None,
    cwd: Optional[Path] = None,
) -> tuple[list[MatchResult], list[int]]:
    history_results, matched_indices = search_history_only(
        query,
        history,
        candidate_indices=candidate_indices,
    )
    results = merge_runtime_and_rank(
        query,
        cursor_pos,
        history_results,
        limit=limit,
        cwd=cwd,
    )
    return results, matched_indices


def match_result_to_payload(item: MatchResult) -> dict[str, Any]:
    return {
        "text": item.text,
        "score": item.score,
        "positions": item.positions,
        "exact": item.exact,
        "recency": item.recency,
        "cwd": item.cwd,
    }


def match_result_from_payload(payload: object) -> Optional[MatchResult]:
    if not isinstance(payload, dict):
        return None
    text = payload.get("text")
    score = payload.get("score")
    positions = payload.get("positions")
    exact = payload.get("exact", False)
    recency = payload.get("recency", 0)
    cwd = payload.get("cwd")
    if not isinstance(text, str) or not isinstance(score, int) or not isinstance(positions, list):
        return None
    if cwd is not None and not isinstance(cwd, str):
        return None
    parsed_positions: list[int] = []
    for pos in positions:
        if not isinstance(pos, int):
            return None
        parsed_positions.append(pos)
    return MatchResult(
        text=text,
        score=score,
        positions=parsed_positions,
        exact=bool(exact),
        recency=int(recency) if isinstance(recency, int) else 0,
        cwd=cwd,
    )


def daemon_send_request(socket_path: Path, payload: dict[str, Any], *, timeout: float = 0.5) -> Optional[dict[str, Any]]:
    data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(socket_path))
            sock.sendall(data)
            chunks: list[bytes] = []
            total = 0
            limit = 64 * 1024 * 1024
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > limit:
                    return None
                if b"\n" in chunk:
                    break
    except OSError:
        return None

    raw = b"".join(chunks)
    if not raw:
        return None
    line = raw.split(b"\n", 1)[0].strip()
    if not line:
        return None
    try:
        decoded = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def launch_history_daemon(
    script_path: Path,
    history_path: Path,
    socket_path: Path,
    *,
    history_length: int,
    use_custom_history: bool = False,
) -> bool:
    try:
        socket_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--daemon",
        "--history-file",
        str(history_path),
        "--socket-path",
        str(socket_path),
        "--history-length",
        str(history_length),
    ]
    if use_custom_history:
        cmd.append("--use-custom-history")
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        return False
    return True


class HistoryDaemonClient:
    def __init__(
        self,
        socket_path: Path,
        history_path: Path,
        script_path: Path,
        *,
        debug: bool = False,
        history_length: int = 10_000,
        use_custom_history: bool = False,
    ) -> None:
        self.socket_path = socket_path
        self.history_path = history_path
        self.script_path = script_path
        self.debug = debug
        self.history_length = history_length
        self.use_custom_history = use_custom_history

    def ensure_running(self) -> bool:
        ping = daemon_send_request(self.socket_path, {"action": "ping"}, timeout=0.15)
        if isinstance(ping, dict) and ping.get("ok") is True:
            daemon_debug_log(self.debug, f"using existing daemon at {self.socket_path}")
            self._debug_log_baseline_count()
            return True

        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
                daemon_debug_log(self.debug, f"removed stale socket at {self.socket_path}")
            except OSError:
                pass

        daemon_debug_log(self.debug, f"starting new daemon at {self.socket_path}")
        if not launch_history_daemon(
            self.script_path,
            self.history_path,
            self.socket_path,
            history_length=self.history_length,
            use_custom_history=self.use_custom_history,
        ):
            daemon_debug_log(self.debug, "failed to launch daemon process")
            return False

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            ping = daemon_send_request(self.socket_path, {"action": "ping"}, timeout=0.15)
            if isinstance(ping, dict) and ping.get("ok") is True:
                daemon_debug_log(self.debug, "new daemon is ready")
                self._debug_log_baseline_count()
                return True
            time.sleep(0.03)
        daemon_debug_log(self.debug, "daemon did not become ready before timeout")
        return False

    def _debug_log_baseline_count(self) -> None:
        if not self.debug:
            return
        response = daemon_send_request(
            self.socket_path,
            {"action": "search_history", "query": "", "limit": 1},
            timeout=0.2,
        )
        if not isinstance(response, dict) or response.get("ok") is not True:
            daemon_debug_log(self.debug, "matched_count=<unavailable>")
            return
        raw_count = response.get("matched_count")
        count = raw_count if isinstance(raw_count, int) else 0
        daemon_debug_log(self.debug, f"matched_count={count} for empty query")

    def search_history(
        self,
        query: str,
        *,
        candidate_indices: Optional[list[int]] = None,
        limit: Optional[int] = None,
        cwd: Optional[str] = None,
    ) -> Optional[tuple[list[MatchResult], Optional[list[int]], int]]:
        payload: dict[str, Any] = {"action": "search_history", "query": query}
        if candidate_indices is not None and len(candidate_indices) <= 10_000:
            payload["candidate_indices"] = candidate_indices
        if limit is not None:
            payload["limit"] = limit
        if cwd:
            payload["cwd"] = normalize_cwd_value(cwd)

        response = daemon_send_request(self.socket_path, payload)
        if response is None:
            if not self.ensure_running():
                return None
            response = daemon_send_request(self.socket_path, payload)
            if response is None:
                return None

        if response.get("ok") is not True:
            return None

        raw_results = response.get("history_results")
        raw_indices = response.get("matched_indices")
        raw_count = response.get("matched_count")
        if not isinstance(raw_results, list):
            return None
        if raw_indices is not None and not isinstance(raw_indices, list):
            return None

        parsed_results: list[MatchResult] = []
        for item in raw_results:
            parsed = match_result_from_payload(item)
            if parsed is None:
                return None
            parsed_results.append(parsed)

        parsed_indices: Optional[list[int]] = None
        if isinstance(raw_indices, list):
            parsed_indices = []
            for idx in raw_indices:
                if not isinstance(idx, int):
                    return None
                parsed_indices.append(idx)

        matched_count = raw_count if isinstance(raw_count, int) else (
            len(parsed_indices) if parsed_indices is not None else 0
        )
        if self.debug:
            indices_state = "included" if parsed_indices is not None else "omitted"
            daemon_debug_log(
                True,
                f"query={query!r} matched_count={matched_count} matched_indices={indices_state}",
            )
        return parsed_results, parsed_indices, matched_count


def daemon_read_request(conn: socket.socket) -> Optional[dict[str, Any]]:
    chunks: list[bytes] = []
    total = 0
    limit = 64 * 1024 * 1024
    while True:
        try:
            chunk = conn.recv(65536)
        except OSError:
            return None
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            return None
        if b"\n" in chunk:
            break
    raw = b"".join(chunks)
    if not raw:
        return None
    line = raw.split(b"\n", 1)[0].strip()
    if not line:
        return None
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def daemon_write_response(conn: socket.socket, payload: dict[str, Any]) -> None:
    try:
        conn.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
    except OSError:
        return


def run_history_daemon(
    history_path: Path,
    socket_path: Path,
    *,
    debug: bool = False,
    history_length: int = 10_000,
    use_custom_history: bool = False,
) -> int:
    if use_custom_history:
        try:
            ensure_custom_history_file(history_path)
            trim_custom_history(history_path, history_length)
        except OSError as exc:
            print(f"zsh_flex_history daemon: failed to initialize custom history: {exc}", file=sys.stderr)
            return 1
        except sqlite3.Error as exc:
            print(f"zsh_flex_history daemon: failed to trim custom history: {exc}", file=sys.stderr)
            return 1
    history = load_history_source(history_path, use_custom_history=use_custom_history)
    signature = history_file_signature(history_path)

    try:
        socket_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"zsh_flex_history daemon: failed to create socket directory: {exc}", file=sys.stderr)
        return 1

    if socket_path.exists():
        ping = daemon_send_request(socket_path, {"action": "ping"}, timeout=0.15)
        if isinstance(ping, dict) and ping.get("ok") is True:
            daemon_debug_log(debug, f"daemon already running at {socket_path}, exiting")
            return 0
        try:
            socket_path.unlink()
            daemon_debug_log(debug, f"removed stale socket at {socket_path}")
        except OSError:
            pass

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        try:
            try:
                server.bind(str(socket_path))
                os.chmod(socket_path, 0o600)
                server.listen(16)
                daemon_debug_log(debug, f"daemon listening on {socket_path} (history={history_path})")
            except OSError as exc:
                print(f"zsh_flex_history daemon: failed to bind socket: {exc}", file=sys.stderr)
                return 1

            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    continue
                with conn:
                    request = daemon_read_request(conn)
                    if request is None:
                        daemon_write_response(conn, {"ok": False, "error": "invalid request"})
                        continue

                    new_signature = history_file_signature(history_path)
                    if new_signature != signature:
                        history = load_history_source(
                            history_path,
                            use_custom_history=use_custom_history,
                            existing_history=history if use_custom_history else None,
                        )
                        signature = new_signature

                    action = request.get("action")
                    if action == "ping":
                        daemon_write_response(conn, {"ok": True})
                        continue

                    if action != "search_history":
                        daemon_write_response(conn, {"ok": False, "error": "unknown action"})
                        continue

                    raw_query = request.get("query", "")
                    query = raw_query if isinstance(raw_query, str) else str(raw_query)
                    raw_candidates = request.get("candidate_indices")
                    candidate_indices: Optional[list[int]] = None
                    if isinstance(raw_candidates, list):
                        parsed_candidates: list[int] = []
                        max_idx = len(history) - 1
                        for item in raw_candidates:
                            if isinstance(item, int) and 0 <= item <= max_idx:
                                parsed_candidates.append(item)
                        candidate_indices = parsed_candidates
                    raw_limit = request.get("limit")
                    limit = raw_limit if isinstance(raw_limit, int) else None
                    raw_cwd = request.get("cwd")
                    current_cwd = normalize_cwd_value(raw_cwd) if isinstance(raw_cwd, str) else None

                    history_results_all, matched_indices = search_history_only(
                        query,
                        history,
                        candidate_indices=candidate_indices,
                        limit=None,
                    )
                    history_results = apply_prefix_priority(
                        query,
                        history_results_all,
                        limit=limit,
                        current_cwd=current_cwd,
                    )
                    matched_count = len(matched_indices)
                    # Avoid sending a huge full-history index list for the
                    # empty query on startup; for any non-empty query, return
                    # full indices to preserve incremental narrowing behavior.
                    indices_payload: Optional[list[int]] = None
                    if query != "":
                        indices_payload = matched_indices
                    daemon_write_response(
                        conn,
                        {
                            "ok": True,
                            "history_results": [match_result_to_payload(item) for item in history_results],
                            "matched_indices": indices_payload,
                            "matched_indices_omitted": indices_payload is None,
                            "matched_count": matched_count,
                        },
                    )
        finally:
            try:
                socket_path.unlink()
            except OSError:
                pass


def truncate_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    out: list[str] = []
    used = 0
    for ch in text:
        w = char_display_width(ch)
        if used + w > width and out:
            break
        if used + w > width:
            break
        out.append(ch)
        used += w
    return "".join(out)


def char_display_width(ch: str) -> int:
    if not ch:
        return 0
    if ch == "\n":
        return 0
    if ch == "\t":
        return 4
    codepoint = ord(ch)
    if codepoint < 32 or (0x7F <= codepoint < 0xA0):
        return 0
    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("F", "W"):
        return 2
    return 1


def text_display_width(text: str) -> int:
    return sum(char_display_width(ch) for ch in text)


def query_window(query: str, cursor_pos: int, available: int) -> tuple[int, str]:
    if available <= 0:
        return 0, ""
    max_start = max(0, len(query) - available)
    start = min(max(0, cursor_pos - available + 1), max_start)
    return start, query[start : start + available]


@dataclass
class QueryVisualRow:
    start: int
    end: int
    text: str
    display_width: int


def build_query_visual_rows(query: str, render_width: int) -> list[QueryVisualRow]:
    width = max(1, render_width)
    rows: list[QueryVisualRow] = []
    start = 0
    buf: list[str] = []
    buf_width = 0
    i = 0
    while i < len(query):
        ch = query[i]
        if ch == "\n":
            rows.append(QueryVisualRow(start=start, end=i, text="".join(buf), display_width=buf_width))
            i += 1
            start = i
            buf = []
            buf_width = 0
            continue
        ch_width = char_display_width(ch)
        if ch_width > 0 and buf and (buf_width + ch_width) > width:
            rows.append(QueryVisualRow(start=start, end=i, text="".join(buf), display_width=buf_width))
            start = i
            buf = []
            buf_width = 0
            continue
        if ch_width > 0 and not buf and ch_width > width:
            rows.append(QueryVisualRow(start=start, end=i + 1, text=ch, display_width=width))
            i += 1
            start = i
            buf = []
            buf_width = 0
            continue
        buf.append(ch)
        buf_width += ch_width
        i += 1
    rows.append(QueryVisualRow(start=start, end=len(query), text="".join(buf), display_width=buf_width))
    return rows


def query_cursor_visual_position(rows: list[QueryVisualRow], cursor_pos: int) -> tuple[int, int]:
    if not rows:
        return 0, 0
    for rindex, row in enumerate(rows):
        if cursor_pos <= row.end:
            offset = max(0, min(cursor_pos - row.start, len(row.text)))
            col = text_display_width(row.text[:offset])
            col = max(0, min(col, row.display_width))
            return rindex, col
    last = rows[-1]
    return len(rows) - 1, last.display_width


def query_pos_from_visual(
    query: str,
    render_width: int,
    row_start: int,
    click_row: int,
    click_col: int,
) -> int:
    rows = build_query_visual_rows(query, render_width)
    if not rows:
        return 0
    row_index = max(0, min(row_start + click_row, len(rows) - 1))
    row = rows[row_index]
    col = max(0, click_col)
    if col >= row.display_width:
        return row.end
    used = 0
    for idx, ch in enumerate(row.text):
        w = char_display_width(ch)
        if w <= 0:
            continue
        if col < used + w:
            return row.start + idx
        used += w
    return row.end


def wrapped_query_layout(
    query: str,
    cursor_pos: int,
    render_width: int,
    panel_rows: int,
) -> tuple[int, int, int, int]:
    render_width = max(1, render_width)
    cursor_pos = max(0, min(cursor_pos, len(query)))
    query_rows_limit = max(1, panel_rows - 1)
    rows = build_query_visual_rows(query, render_width)
    cursor_row, _cursor_col = query_cursor_visual_position(rows, cursor_pos)
    query_start = max(0, cursor_row - (query_rows_limit - 1))
    query_rows_used = min(query_rows_limit, max(1, len(rows) - query_start))
    query_view_len = 0
    results_visible = max(0, panel_rows - query_rows_used)
    return query_start, query_view_len, query_rows_used, results_visible


def selection_bounds(sel_anchor: Optional[int], sel_end: Optional[int]) -> Optional[tuple[int, int]]:
    if sel_anchor is None or sel_end is None:
        return None
    if sel_anchor == sel_end:
        return None
    return (min(sel_anchor, sel_end), max(sel_anchor, sel_end))


def render_result_line(
    item: MatchResult,
    selected: bool,
    width: int,
    *,
    query: str = "",
    unselected_white: bool = False,
    suffix_text: str = "",
) -> str:
    if width <= 0:
        return ""

    gutter_width = 1
    suffix_width = text_display_width(suffix_text) + 4 if suffix_text else 0
    body_width = max(0, width - gutter_width - suffix_width)
    display_text = item.text.replace("\r", " ").replace("\n", " ")
    text = truncate_text(display_text, body_width)
    ordered_positions = ordered_query_word_positions(query, item.text_lower if item.text_lower is not None else item.text.lower())
    pos_set = set(ordered_positions if ordered_positions is not None else item.positions)

    match_fg = DORIC["fg_red"]
    row_fg = DORIC["fg_main"] if unselected_white else DORIC["fg_neutral"]
    row_bg = DORIC["bg_shadow_subtle"]
    selected_fg = DORIC["fg_red"]
    selected_bg = DORIC["bg_red"]

    if selected:
        normal_style = RESET + style(fg_rgb=selected_fg, bg_rgb=selected_bg, bold=True)
    else:
        normal_style = RESET + style(fg_rgb=row_fg, bg_rgb=row_bg)
    if selected:
        match_style = RESET + style(fg_rgb=selected_fg, bg_rgb=selected_bg, bold=True, underline=True)
    else:
        match_style = style(fg_rgb=match_fg, bg_rgb=row_bg, underline=True)

    if selected:
        gutter = f"{style(bg_rgb=DORIC['bg_red'])} {RESET}"
    else:
        gutter = f"{style(bg_rgb=row_bg)} {RESET}"

    out: list[str] = []
    active_style = ""
    for i, ch in enumerate(text):
        is_match_char = i in pos_set and ch != " "
        target_style = match_style if is_match_char else normal_style
        if target_style != active_style:
            out.append(target_style if target_style else RESET)
            active_style = target_style
        out.append(ch)
    pad_width = max(0, body_width - text_display_width(text))
    if pad_width > 0:
        if normal_style != active_style:
            out.append(normal_style)
        out.append(" " * pad_width)
    if suffix_text:
        if normal_style != active_style:
            out.append(normal_style)
        out.append(" ")
        out.append(f"{style(fg_rgb=DORIC['fg_shadow_subtle'], bg_rgb=selected_bg if selected else row_bg)}[{suffix_text}]{RESET}")
        out.append(f"{style(bg_rgb=selected_bg if selected else row_bg)} {RESET}")
    out.append(RESET)
    return gutter + "".join(out)


def draw_panel(
    anchor_row: int,
    anchor_col: int,
    query: str,
    cursor_pos: int,
    sel_anchor: Optional[int],
    sel_end: Optional[int],
    results: list[MatchResult],
    selected: int,
    offset: int,
    panel_rows: int,
    width: int,
    status_message: str = "",
    debug_note: str = "",
    total_count: Optional[int] = None,
) -> tuple[int, int, int, int]:
    anchor_col = max(1, anchor_col)
    render_width = max(1, width - anchor_col + 1)
    muted = style(fg_rgb=DORIC["fg_shadow_subtle"])
    query_lead_cols = 1

    lines: list[str] = []
    cursor_pos = max(0, min(cursor_pos, len(query)))
    query_start, query_view_len, query_rows_used, results_visible = wrapped_query_layout(
        query,
        cursor_pos,
        render_width,
        panel_rows,
    )
    query_rows = build_query_visual_rows(query, render_width)
    visible_query_rows = query_rows[query_start : query_start + query_rows_used]
    sel = selection_bounds(sel_anchor, sel_end)
    syntax_tokens = highlight_tokens(query)
    for row, vrow in enumerate(visible_query_rows):
        seg_len = vrow.display_width
        query_parts: list[str] = [RESET]
        active_query_style = ""
        for i, ch in enumerate(vrow.text):
            qidx = vrow.start + i
            token = syntax_tokens[qidx] if qidx < len(syntax_tokens) else "default"
            token_style = ansi_for_token(token)
            if sel and sel[0] <= qidx < sel[1]:
                if active_query_style:
                    query_parts.append(RESET)
                    active_query_style = ""
                if token_style:
                    query_parts.append(f"{QUERY_SELECTION_BG}{token_style}{ch}{RESET}")
                else:
                    query_parts.append(f"{QUERY_SELECTION_BG}{ch}{RESET}")
                continue
            if token_style != active_query_style:
                query_parts.append(token_style if token_style else RESET)
                active_query_style = token_style
            query_parts.append(ch)
        if active_query_style:
            query_parts.append(RESET)
        query_line = " " + "".join(query_parts)
        if row == 0 and debug_note:
            room = max(0, render_width - seg_len)
            if room > 0:
                note_text = debug_note[: max(0, room - 1)]
                if note_text:
                    query_line += f" {muted}{note_text}{RESET}"
        lines.append(query_line)

    effective_total = max(len(results), total_count or 0)
    top_remaining = max(0, effective_total - results_visible)
    use_visible_total_for_more = top_remaining <= 97
    for i in range(results_visible):
        idx = offset + i
        if idx >= len(results):
            if i == 0 and status_message:
                lines.append(
                    f"{style(fg_rgb=DORIC['fg_shadow_intense'], bg_rgb=DORIC['bg_neutral'], bold=True)} {status_message} {RESET}"
                )
            else:
                lines.append("")
            continue
        remaining = max(0, effective_total - (offset + results_visible))
        if use_visible_total_for_more:
            remaining = max(0, len(results) - (offset + results_visible))
        is_last_visible_row = i == (results_visible - 1)
        more_text = f"{remaining} more" if (is_last_visible_row and remaining > 0) else ""
        base_line = render_result_line(
            results[idx],
            idx == selected,
            render_width,
            query=query,
            unselected_white=True,
            suffix_text=more_text,
        )
        lines.append(base_line)

    for i, line in enumerate(lines[:panel_rows]):
        term_write(move_to(anchor_row + i, anchor_col) + CLEAR_TO_END + line)

    # Put cursor on query input field.
    cursor_row_abs, cursor_col = query_cursor_visual_position(query_rows, cursor_pos)
    cursor_row = min(query_rows_used - 1, max(0, cursor_row_abs - query_start))
    cursor_col = max(0, min(cursor_col + query_lead_cols, render_width - 1))
    term_write(move_to(anchor_row + cursor_row, anchor_col + cursor_col))
    term_flush()
    return query_start, query_view_len, query_rows_used, results_visible


def read_key(fd: int, timeout: Optional[float] = 0.1) -> tuple[str, object]:
    def read_escape_tail() -> bytes:
        # Read an escape sequence byte-by-byte so we do not over-read into
        # subsequent pasted payload bytes.
        seq = b""
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            rdy, _, _ = select.select([fd], [], [], 0.01)
            if not rdy:
                if seq:
                    break
                continue
            chunk = os.read(fd, 1)
            if not chunk:
                break
            seq += chunk
            # CSI sequence: ESC [ ... <final>
            if seq.startswith(b"[") and len(seq) >= 2 and seq[-1:] and (64 <= seq[-1] <= 126):
                break
            # SS3 sequence: ESC O <final>
            if seq.startswith(b"O") and len(seq) >= 2:
                break
            # Alt-modified key (ESC + single byte).
            if not seq.startswith((b"[", b"O")) and len(seq) >= 1:
                break
        return seq

    def parse_csi_key(full: bytes) -> Optional[tuple[str, object]]:
        m_u = re.fullmatch(rb"\x1b\[(\d+)(?:;(\d+))?u", full)
        if m_u:
            codepoint = int(m_u.group(1))
            mod = int(m_u.group(2) or b"1")
            shift = (mod - 1) & 1
            ctrl = (mod - 1) & 4
            alt = (mod - 1) & 2

            if codepoint == 13:
                return "enter", None
            if codepoint == 9:
                return "tab", None
            if codepoint in (8, 127):
                if ctrl:
                    return "backspace_word", None
                return "backspace", None
            if codepoint == 27:
                return "quit", None
            if codepoint == 1 and ctrl:
                return "home", None
            if codepoint == 5 and ctrl:
                return "end", None
            if codepoint == 11 and ctrl:
                return "kill_to_end", None
            if codepoint == 21 and ctrl:
                return "kill_to_start", None
            if codepoint == 23 and ctrl:
                return "backspace_word", None
            if codepoint == 98 and alt:
                return "word_left", None
            if codepoint == 102 and alt:
                return "word_right", None
            if codepoint in (65, 97) and alt:
                return "select_all", None
            if codepoint in (67, 99) and alt:
                return "copy", None
            if codepoint in (86, 118) and alt:
                return "paste", None
            if 32 <= codepoint < 127:
                return "char", chr(codepoint)

        # Handle modified cursor keys in CSI-u style (e.g. ESC [ 1 ; 2 D).
        m = re.fullmatch(rb"\x1b\[(?:1;)?(\d+)([ABCDHF])", full)
        if m:
            mod = int(m.group(1))
            key = m.group(2)
            if mod in (1,):
                if key == b"D":
                    return "left", None
                if key == b"C":
                    return "right", None
                if key == b"H":
                    return "home", None
                if key == b"F":
                    return "end", None
            if mod == 2:
                if key == b"D":
                    return "shift_left", None
                if key == b"C":
                    return "shift_right", None
                if key == b"H":
                    return "shift_home", None
                if key == b"F":
                    return "shift_end", None
            if mod == 5:
                if key == b"D":
                    return "word_left", None
                if key == b"C":
                    return "word_right", None
        # xterm/kitty ctrl+arrow variants.
        if full in (b"\x1b[1;5D", b"\x1b[5D"):
            return "word_left", None
        if full in (b"\x1b[1;5C", b"\x1b[5C"):
            return "word_right", None
        if full in (b"\x1b[1;2D",):
            return "shift_left", None
        if full in (b"\x1b[1;2C",):
            return "shift_right", None
        if full in (b"\x1b[1;2H",):
            return "shift_home", None
        if full in (b"\x1b[1;2F",):
            return "shift_end", None
        return None

    def read_pending_burst(initial: bytes = b"") -> str:
        buf = bytearray(initial)
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.015)
            if not ready:
                break
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) >= 1_000_000:
                break
        return bytes(buf).decode("utf-8", errors="replace")

    def read_utf8_char(first_byte: int) -> str:
        if first_byte < 0x80:
            return chr(first_byte)
        need = 0
        if (first_byte & 0xE0) == 0xC0:
            need = 2
        elif (first_byte & 0xF0) == 0xE0:
            need = 3
        elif (first_byte & 0xF8) == 0xF0:
            need = 4
        if need == 0:
            return bytes((first_byte,)).decode("utf-8", errors="replace")
        buf = bytearray((first_byte,))
        deadline = time.monotonic() + 0.03
        while len(buf) < need and time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.005)
            if not ready:
                break
            chunk = os.read(fd, 1)
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf).decode("utf-8", errors="replace")

    while True:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return "timeout", None
        data = os.read(fd, 1)
        if not data:
            continue

        ch = data[0]
        if ch == 3:
            return "quit", None
        if ch == 1:
            return "home", None
        if ch == 5:
            return "end", None
        if ch in (10, 13):
            # If newline arrives with queued bytes, treat as pasted content
            # instead of immediate submit.
            queued, _, _ = select.select([fd], [], [], 0)
            if queued:
                return "paste_text", read_pending_burst(b"\n")
            return "enter", None
        if ch == 9:
            return "tab", None
        if ch in (8, 127):
            return "backspace", None
        if ch == 23:
            return "backspace_word", None
        if ch == 21:
            return "kill_to_start", None
        if ch == 11:
            return "kill_to_end", None
        if ch == 27:
            seq = read_escape_tail()
            full = b"\x1b" + seq
            if full == b"\x1b":
                return "quit", None
            if full in (b"\x1b[A",):
                return "up", None
            if full in (b"\x1b[B",):
                return "down", None
            if full in (b"\x1b[C",):
                return "right", None
            if full in (b"\x1b[D",):
                return "left", None
            if full in (b"\x1b[H", b"\x1b[1~", b"\x1bOH"):
                return "home", None
            if full in (b"\x1b[F", b"\x1b[4~", b"\x1bOF"):
                return "end", None
            if full in (b"\x1b[3~",):
                return "delete", None
            if full in (b"\x1b[5~",):
                return "pgup", None
            if full in (b"\x1b[6~",):
                return "pgdn", None
            parsed = parse_csi_key(full)
            if parsed is not None:
                return parsed

            m = re.match(rb"\x1b\[<(\d+);(\d+);(\d+)([mM])", full)
            if m:
                bstate = int(m.group(1))
                x = int(m.group(2))
                y = int(m.group(3))
                action = m.group(4).decode("ascii")
                return "mouse", (bstate, x, y, action)
            if full in (b"\x1bb", b"\x1b[1;3D"):
                return "word_left", None
            if full in (b"\x1bf", b"\x1b[1;3C"):
                return "word_right", None
            if full in (b"\x1ba", b"\x1bA"):
                return "select_all", None
            if full in (b"\x1bc", b"\x1bC"):
                return "copy", None
            if full in (b"\x1bv", b"\x1bV"):
                return "paste", None
            continue
        if ch >= 32:
            queued, _, _ = select.select([fd], [], [], 0)
            if queued:
                burst = read_pending_burst(bytes((ch,)))
                if len(burst) > 1 or "\n" in burst:
                    return "paste_text", burst
                return "char", burst
            return "char", read_utf8_char(ch)


def move_word_left(query: str, cursor_pos: int) -> int:
    i = max(0, min(cursor_pos, len(query)))
    while i > 0 and query[i - 1].isspace():
        i -= 1
    while i > 0 and not query[i - 1].isspace():
        i -= 1
    return i


def move_word_right(query: str, cursor_pos: int) -> int:
    i = max(0, min(cursor_pos, len(query)))
    n = len(query)
    while i < n and not query[i].isspace():
        i += 1
    while i < n and query[i].isspace():
        i += 1
    return i


def run(
    history: list[HistoryEntry],
    *,
    inline_with_prompt: bool = False,
    history_updates: Optional[queue.Queue[tuple[str, object]]] = None,
    history_client: Optional[HistoryDaemonClient] = None,
) -> Optional[str]:
    global TERM_OUT
    tty_in_file = None
    tty_out_file = None
    current_cwd_text = normalize_cwd_value(os.getcwd())
    current_cwd_path = Path(current_cwd_text)
    fd: Optional[int] = None
    for tty_path in ("/dev/tty", os.ctermid()):
        try:
            tty_in_file = open(tty_path, "r", encoding="utf-8", buffering=1)
            tty_out_file = open(tty_path, "w", encoding="utf-8", buffering=1)
            candidate_fd = tty_in_file.fileno()
            if os.isatty(candidate_fd):
                fd = candidate_fd
                TERM_OUT = tty_out_file
                break
            tty_in_file.close()
            tty_out_file.close()
            tty_in_file = None
            tty_out_file = None
        except OSError:
            if tty_in_file is not None:
                tty_in_file.close()
                tty_in_file = None
            if tty_out_file is not None:
                tty_out_file.close()
                tty_out_file = None

    if fd is None:
        candidate_fd = sys.stdin.fileno()
        if os.isatty(candidate_fd):
            fd = candidate_fd
            TERM_OUT = sys.stdout

    if fd is None:
        print("zsh_flex_history: no usable TTY available for interactive mode", file=sys.stderr)
        return None
    min_result_rows = 3
    min_panel_rows = 1 + min_result_rows

    try:
        with RawTerminal(fd) as rt:
            term_size = tty_terminal_size(fd)
            term_lines = term_size.lines
            pos = query_cursor_position(fd)
            if pos is None:
                start_row = max(1, term_lines - 1)
                start_col = 1
            else:
                start_row = pos[0]
                start_col = pos[1]
            # Keep all row math within the visible terminal bounds even if a
            # terminal reports a transient cursor value during startup.
            start_row = max(1, min(start_row, term_lines))
            space_below = max(0, term_lines - start_row)
            # If there is no room to draw result rows below the prompt area,
            # reserve lines by scrolling a small amount.
            if inline_with_prompt:
                required_below = max(0, min_panel_rows - 1)
            else:
                required_below = min_panel_rows
            scroll_rows = max(0, required_below - space_below)
            if scroll_rows > 0:
                term_write(move_to(term_lines, 1) + ("\n" * scroll_rows))
                term_flush()
                start_row = max(1, start_row - scroll_rows)
                space_below = max(0, term_lines - start_row)
            initial_cursor_row = start_row
            initial_cursor_col = start_col

            # For print-only mode, anchor on the prompt row itself so query
            # input starts on the same line as the prompt.
            # Otherwise, use the row below the prompt when possible.
            if inline_with_prompt:
                anchor_row = max(1, start_row)
                anchor_col = max(1, start_col - 1)
                panel_rows = max(1, term_lines - anchor_row + 1)
            elif space_below >= 1:
                anchor_row = start_row + 1
                anchor_col = 1
                panel_rows = max(1, space_below)
            else:
                anchor_row = max(1, start_row)
                anchor_col = 1
                panel_rows = max(1, term_lines - anchor_row + 1)
            for row in range(anchor_row, anchor_row + panel_rows):
                term_write(move_to(row, anchor_col) + CLEAR_TO_END)
            term_write(move_to(anchor_row, anchor_col))
            term_flush()

            query = ""
            cursor_pos = 0
            sel_anchor: Optional[int] = None
            sel_end: Optional[int] = None
            selected = 0
            offset = 0
            chosen: Optional[str] = None
            query_start = 0
            query_rows_used = 1
            results_visible = max(1, panel_rows - 1)
            render_width = 1
            all_indices: list[int] = []
            initial_matched_indices: Optional[list[int]] = None
            initial_matched_count: Optional[int] = None
            if history_client is not None:
                loaded = history_client.search_history(
                    "",
                    limit=MAX_RETURNED_RESULTS,
                    cwd=current_cwd_text,
                )
                if loaded is None:
                    initial_results = []
                    history_load_error = True
                else:
                    history_matches, initial_matched_indices, initial_matched_count = loaded
                    initial_results = merge_runtime_and_rank(
                        "",
                        cursor_pos,
                        history_matches,
                        limit=MAX_RETURNED_RESULTS,
                        cwd=current_cwd_path,
                    )
                    history_load_error = False
            else:
                all_indices = list(range(len(history)))
                initial_results, _ = search(
                    "",
                    history,
                    cursor_pos=cursor_pos,
                    candidate_indices=all_indices,
                    limit=MAX_RETURNED_RESULTS,
                    cwd=current_cwd_path,
                )
                initial_matched_indices = all_indices
                initial_matched_count = len(all_indices)
                history_load_error = False
            last_query = ""
            last_matched_indices = initial_matched_indices
            initial_total_count = max(len(initial_results), initial_matched_count or 0)
            match_cache: dict[str, tuple[Optional[list[int]], list[MatchResult], Optional[int], int]] = {
                "": (initial_matched_indices, initial_results, initial_matched_count, initial_total_count)
            }
            cache_order: list[str] = [""]
            cache_limit = 128
            history_loading = history_updates is not None and history_client is None
            displayed_results = initial_results
            displayed_matched_indices = initial_matched_indices
            displayed_matched_count = initial_matched_count
            displayed_total_count = initial_total_count
            mouse_selecting = False
            mouse_enabled = False
            last_left_click_time = 0.0
            last_left_click_row = -1
            last_left_click_col = -1
            left_click_count = 0
            last_drawn_panel_rows = panel_rows
            search_requests: queue.Queue[Optional[tuple[str, Optional[list[int]], str]]] = queue.Queue()
            search_updates: queue.Queue[
                tuple[str, Optional[list[int]], list[MatchResult], Optional[int], int, bool]
            ] = queue.Queue()
            search_stop = threading.Event()
            queued_search_key: Optional[str] = None

            def search_candidates_for(query_text: str) -> Optional[list[int]]:
                if history_client is not None:
                    prefix = query_text[:-1]
                    while prefix:
                        cached = match_cache.get(prefix)
                        if cached is not None and cached[0] is not None:
                            return cached[0]
                        prefix = prefix[:-1]
                    return None
                candidate_indices = all_indices
                prefix = query_text[:-1]
                while prefix:
                    cached = match_cache.get(prefix)
                    if cached is not None and cached[0] is not None:
                        candidate_indices = cached[0]
                        break
                    prefix = prefix[:-1]
                return candidate_indices

            def run_search_request(
                query_text: str,
                candidate_indices: Optional[list[int]],
                cwd_text: str,
            ) -> tuple[Optional[list[int]], list[MatchResult], Optional[int], int, bool]:
                search_error = False
                if history_client is not None:
                    remote = history_client.search_history(
                        query_text,
                        candidate_indices=candidate_indices,
                        limit=MAX_RETURNED_RESULTS,
                        cwd=cwd_text,
                    )
                    if remote is None:
                        history_results = []
                        matched_indices = None
                        matched_count = None
                        search_error = True
                    else:
                        history_results, matched_indices, matched_count = remote
                    resolved_results = merge_runtime_and_rank(
                        query_text,
                        len(query_text),
                        history_results,
                        limit=MAX_RETURNED_RESULTS,
                        cwd=current_cwd_path,
                    )
                else:
                    resolved_results, matched_indices = search(
                        query_text,
                        history,
                        cursor_pos=len(query_text),
                        candidate_indices=candidate_indices,
                        limit=MAX_RETURNED_RESULTS,
                        cwd=current_cwd_path,
                    )
                    matched_count = len(matched_indices) if matched_indices is not None else None
                total_count = max(len(resolved_results), matched_count or 0)
                return matched_indices, resolved_results, matched_count, total_count, search_error

            def search_worker() -> None:
                while not search_stop.is_set():
                    try:
                        request = search_requests.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if request is None:
                        break
                    query_text, candidate_indices, cwd_text = request
                    matched_indices, resolved_results, matched_count, total_count, search_error = run_search_request(
                        query_text,
                        candidate_indices,
                        cwd_text,
                    )
                    search_updates.put(
                        (query_text, matched_indices, resolved_results, matched_count, total_count, search_error)
                    )

            search_thread = threading.Thread(target=search_worker, daemon=True)
            search_thread.start()

            def refresh_anchor_from_cursor() -> None:
                nonlocal start_row, start_col, anchor_row, anchor_col, panel_rows, last_drawn_panel_rows
                nonlocal initial_cursor_row, initial_cursor_col
                old_anchor_row = anchor_row
                old_anchor_col = anchor_col
                old_panel_rows = max(panel_rows, last_drawn_panel_rows)
                clear_panel_area = True

                term_size = tty_terminal_size(fd)
                term_lines = term_size.lines
                pos = query_cursor_position(fd)
                if pos is None:
                    next_start_row = max(1, term_lines - 1)
                    next_start_col = 1
                elif pos[0] == 1:
                    next_start_row = pos[0]
                    next_start_col = pos[1]
                    initial_cursor_row = next_start_row
                    initial_cursor_col = next_start_col
                else:
                    next_start_row = initial_cursor_row
                    next_start_col = initial_cursor_col
                    clear_panel_area = False
                    term_write(move_to(initial_cursor_row, initial_cursor_col))
                    term_flush()

                next_start_row = max(1, min(next_start_row, term_lines))
                space_below = max(0, term_lines - next_start_row)
                if inline_with_prompt:
                    required_below = max(0, min_panel_rows - 1)
                else:
                    required_below = min_panel_rows
                scroll_rows = max(0, required_below - space_below)
                if scroll_rows > 0:
                    term_write(move_to(term_lines, 1) + ("\n" * scroll_rows))
                    term_flush()
                    next_start_row = max(1, next_start_row - scroll_rows)
                    space_below = max(0, term_lines - next_start_row)

                if inline_with_prompt:
                    next_anchor_row = max(1, next_start_row)
                    next_anchor_col = max(1, next_start_col - 1)
                    next_panel_rows = max(1, term_lines - next_anchor_row + 1)
                elif space_below >= 1:
                    next_anchor_row = next_start_row + 1
                    next_anchor_col = 1
                    next_panel_rows = max(1, space_below)
                else:
                    next_anchor_row = max(1, next_start_row)
                    next_anchor_col = 1
                    next_panel_rows = max(1, term_lines - next_anchor_row + 1)

                if clear_panel_area:
                    for row in range(old_anchor_row, old_anchor_row + old_panel_rows):
                        term_write(move_to(row, old_anchor_col) + CLEAR_TO_END)

                start_row = next_start_row
                start_col = next_start_col
                anchor_row = next_anchor_row
                anchor_col = next_anchor_col
                panel_rows = next_panel_rows
                last_drawn_panel_rows = panel_rows

                if clear_panel_area:
                    for row in range(anchor_row, anchor_row + panel_rows):
                        term_write(move_to(row, anchor_col) + CLEAR_TO_END)
                term_write(move_to(anchor_row, anchor_col))
                term_flush()

            def clear_panel_and_restore_cursor() -> None:
                # Clear panel content so repeated invocations always start clean.
                for row in range(anchor_row, anchor_row + max(panel_rows, last_drawn_panel_rows)):
                    term_write(move_to(row, anchor_col) + CLEAR_TO_END)
                # Restore cursor to the exact prompt position captured at invocation start.
                term_write(move_to(start_row, start_col))
                term_flush()

            def clear_selection() -> None:
                nonlocal sel_anchor, sel_end
                sel_anchor = None
                sel_end = None

            def move_cursor(new_pos: int, *, select_mode: bool = False) -> None:
                nonlocal cursor_pos, sel_anchor, sel_end
                new_pos = max(0, min(new_pos, len(query)))
                if select_mode:
                    if sel_anchor is None:
                        sel_anchor = cursor_pos
                    cursor_pos = new_pos
                    sel_end = cursor_pos
                    if sel_anchor == sel_end:
                        sel_anchor = None
                        sel_end = None
                    return
                cursor_pos = new_pos
                clear_selection()

            def sync_mouse_mode() -> None:
                nonlocal mouse_enabled, mouse_selecting
                should_enable = len(query) > 0
                if should_enable and not mouse_enabled:
                    term_write(ENABLE_MOUSE)
                    term_flush()
                    mouse_enabled = True
                elif not should_enable and mouse_enabled:
                    term_write(DISABLE_MOUSE)
                    term_flush()
                    mouse_enabled = False
                    mouse_selecting = False

            def select_all_query() -> None:
                nonlocal sel_anchor, sel_end, cursor_pos
                if not query:
                    clear_selection()
                    return
                sel_anchor = 0
                sel_end = len(query)
                cursor_pos = len(query)

            def cache_put(
                key: str,
                indices: Optional[list[int]],
                cached_results: list[MatchResult],
                matched_count: Optional[int],
                total_count: int,
            ) -> None:
                if key in match_cache:
                    return
                if len(cache_order) >= cache_limit:
                    oldest = cache_order.pop(0)
                    match_cache.pop(oldest, None)
                cache_order.append(key)
                match_cache[key] = (indices, cached_results, matched_count, total_count)

            try:
                while True:
                    while True:
                        try:
                            (
                                result_query,
                                result_indices,
                                result_results,
                                result_count,
                                result_total,
                                result_error,
                            ) = search_updates.get_nowait()
                        except queue.Empty:
                            break
                        queued_search_key = None if queued_search_key == result_query else queued_search_key
                        cache_put(result_query, result_indices, result_results, result_count, result_total)
                        if result_error:
                            history_load_error = True
                        if result_query == query:
                            displayed_matched_indices = result_indices
                            displayed_results = result_results
                            displayed_matched_count = result_count
                            displayed_total_count = result_total

                    if history_updates is not None and history_client is None:
                        while True:
                            try:
                                kind, payload = history_updates.get_nowait()
                            except queue.Empty:
                                break
                            if kind == "loaded":
                                loaded_history = payload
                                if isinstance(loaded_history, list):
                                    history = loaded_history
                                    all_indices = list(range(len(history)))
                                    last_query = ""
                                    last_matched_indices = all_indices
                                    initial_results, _ = search(
                                        "",
                                        history,
                                        cursor_pos=cursor_pos,
                                        candidate_indices=all_indices,
                                        limit=MAX_RETURNED_RESULTS,
                                        cwd=current_cwd_path,
                                    )
                                    initial_total_count = max(len(initial_results), len(all_indices))
                                    match_cache = {"": (all_indices, initial_results, len(all_indices), initial_total_count)}
                                    cache_order = [""]
                                    displayed_matched_indices = all_indices
                                    displayed_results = initial_results
                                    displayed_matched_count = len(all_indices)
                                    displayed_total_count = initial_total_count
                                history_loading = False
                            elif kind == "error":
                                history_loading = False
                                history_load_error = True
    
                    term_size = tty_terminal_size(fd)
                    width = term_size.columns
                    term_lines = term_size.lines
                    render_width = max(1, width - max(1, anchor_col) + 1)
                    required_query_rows = max(1, len(build_query_visual_rows(query, render_width)))
                    desired_panel_rows = max(min_panel_rows, required_query_rows + min_result_rows)
                    max_panel_rows = max(1, term_lines - anchor_row + 1)
                    if desired_panel_rows > max_panel_rows and anchor_row > 1:
                        extra_rows = min(desired_panel_rows - max_panel_rows, anchor_row - 1)
                        if extra_rows > 0:
                            term_write(move_to(term_lines, 1) + ("\n" * extra_rows))
                            term_flush()
                            start_row = max(1, start_row - extra_rows)
                            anchor_row = max(1, anchor_row - extra_rows)
                            max_panel_rows = max(1, term_lines - anchor_row + 1)
                    panel_rows = min(desired_panel_rows, max_panel_rows)
                    if max_panel_rows >= min_panel_rows:
                        panel_rows = max(min_panel_rows, panel_rows)

                    _qs, _qvl, _qru, layout_results_visible = wrapped_query_layout(
                        query,
                        cursor_pos,
                        render_width,
                        panel_rows,
                    )
                    visible = max(1, layout_results_visible)
                    cache_key = query
                    if cache_key in match_cache:
                        matched_indices, results, matched_count, total_count = match_cache[cache_key]
                        displayed_matched_indices = matched_indices
                        displayed_results = results
                        displayed_matched_count = matched_count
                        displayed_total_count = total_count
                    else:
                        if queued_search_key != cache_key:
                            search_requests.put((cache_key, search_candidates_for(cache_key), current_cwd_text))
                            queued_search_key = cache_key
                        matched_indices = displayed_matched_indices
                        results = displayed_results
                        matched_count = displayed_matched_count
                        total_count = displayed_total_count
                    last_query = query
                    last_matched_indices = matched_indices
                    status_message = ""
                    debug_note = ""
                    if history_client is not None and history_client.debug:
                        count_text = "?" if matched_count is None else str(matched_count)
                        indices_text = "no-idx" if matched_indices is None else "idx"
                        debug_note = f"matches={count_text} {indices_text}"
                    if history_loading and not results:
                        status_message = "loading history..."
                    elif history_load_error and not results:
                        status_message = "history load failed"
                    if panel_rows < last_drawn_panel_rows:
                        for row in range(anchor_row + panel_rows, anchor_row + last_drawn_panel_rows):
                            term_write(move_to(row, anchor_col) + CLEAR_TO_END)
                    if selected >= len(results):
                        selected = max(0, len(results) - 1)
                    if selected < offset:
                        offset = selected
                    if selected >= offset + visible:
                        offset = selected - visible + 1
    
                    query_start, _query_view_len, query_rows_used, results_visible = draw_panel(
                        anchor_row,
                        anchor_col,
                        query,
                        cursor_pos,
                        sel_anchor,
                        sel_end,
                        results,
                        selected,
                        offset,
                        panel_rows,
                        width,
                        status_message=status_message,
                        debug_note=debug_note,
                        total_count=total_count,
                    )
                    last_drawn_panel_rows = panel_rows
    
                    input_timeout: Optional[float] = 0.03
                    if not history_loading and queued_search_key is None:
                        input_timeout = None
                    ev, payload = read_key(fd, timeout=input_timeout)
                    if ev == "timeout":
                        continue
    
                    if ev == "quit":
                        clear_panel_and_restore_cursor()
                        return None
                    if ev == "enter":
                        chosen = query
                        break
                    if ev == "tab":
                        if 0 <= selected < len(results):
                            query = results[selected].text
                            cursor_pos = len(query)
                            clear_selection()
                            sync_mouse_mode()
                            selected = 0
                            offset = 0
                        continue
                    if ev == "left":
                        move_cursor(cursor_pos - 1)
                        continue
                    if ev == "right":
                        move_cursor(cursor_pos + 1)
                        continue
                    if ev == "shift_left":
                        move_cursor(cursor_pos - 1, select_mode=True)
                        continue
                    if ev == "shift_right":
                        move_cursor(cursor_pos + 1, select_mode=True)
                        continue
                    if ev == "home":
                        move_cursor(0)
                        continue
                    if ev == "shift_home":
                        move_cursor(0, select_mode=True)
                        continue
                    if ev == "end":
                        move_cursor(len(query))
                        continue
                    if ev == "shift_end":
                        move_cursor(len(query), select_mode=True)
                        continue
                    if ev == "word_left":
                        move_cursor(move_word_left(query, cursor_pos))
                        continue
                    if ev == "word_right":
                        move_cursor(move_word_right(query, cursor_pos))
                        continue
                    if ev == "select_all":
                        select_all_query()
                        continue
                    if ev == "up":
                        selected = max(0, selected - 1)
                        continue
                    if ev == "down":
                        selected = min(max(0, len(results) - 1), selected + 1)
                        continue
                    if ev == "pgup":
                        selected = max(0, selected - visible)
                        continue
                    if ev == "pgdn":
                        selected = min(max(0, len(results) - 1), selected + visible)
                        continue
                    if ev == "backspace":
                        sel = selection_bounds(sel_anchor, sel_end)
                        if sel:
                            query = query[: sel[0]] + query[sel[1] :]
                            cursor_pos = sel[0]
                            clear_selection()
                        elif cursor_pos > 0:
                            query = query[: cursor_pos - 1] + query[cursor_pos:]
                            cursor_pos -= 1
                        sync_mouse_mode()
                        selected = 0
                        offset = 0
                        continue
                    if ev == "backspace_word":
                        sel = selection_bounds(sel_anchor, sel_end)
                        if sel:
                            query = query[: sel[0]] + query[sel[1] :]
                            cursor_pos = sel[0]
                            clear_selection()
                        else:
                            new_pos = move_word_left(query, cursor_pos)
                            if new_pos < cursor_pos:
                                query = query[:new_pos] + query[cursor_pos:]
                                cursor_pos = new_pos
                        sync_mouse_mode()
                        selected = 0
                        offset = 0
                        continue
                    if ev == "kill_to_start":
                        sel = selection_bounds(sel_anchor, sel_end)
                        if sel:
                            query = query[: sel[0]] + query[sel[1] :]
                            cursor_pos = sel[0]
                        else:
                            query = query[cursor_pos:]
                            cursor_pos = 0
                        clear_selection()
                        sync_mouse_mode()
                        selected = 0
                        offset = 0
                        continue
                    if ev == "kill_to_end":
                        sel = selection_bounds(sel_anchor, sel_end)
                        if sel:
                            query = query[: sel[0]] + query[sel[1] :]
                            cursor_pos = sel[0]
                        else:
                            query = query[:cursor_pos]
                        clear_selection()
                        sync_mouse_mode()
                        selected = 0
                        offset = 0
                        continue
                    if ev == "delete":
                        sel = selection_bounds(sel_anchor, sel_end)
                        if sel:
                            query = query[: sel[0]] + query[sel[1] :]
                            cursor_pos = sel[0]
                            clear_selection()
                        elif cursor_pos < len(query):
                            query = query[:cursor_pos] + query[cursor_pos + 1 :]
                        sync_mouse_mode()
                        selected = 0
                        offset = 0
                        continue
                    if ev == "char":
                        ch = str(payload)
                        if not query:
                            refresh_anchor_from_cursor()
                        sel = selection_bounds(sel_anchor, sel_end)
                        if sel:
                            query = query[: sel[0]] + ch + query[sel[1] :]
                            cursor_pos = sel[0] + 1
                            clear_selection()
                        else:
                            query = query[:cursor_pos] + ch + query[cursor_pos:]
                            cursor_pos += 1
                        sync_mouse_mode()
                        selected = 0
                        offset = 0
                        continue
                    if ev == "copy":
                        sel = selection_bounds(sel_anchor, sel_end)
                        if sel:
                            write_clipboard(query[sel[0] : sel[1]])
                        elif query:
                            write_clipboard(query)
                        continue
                    if ev == "paste":
                        pasted = normalize_pasted_text(read_clipboard())
                        if not pasted:
                            continue
                        sel = selection_bounds(sel_anchor, sel_end)
                        if sel:
                            query = query[: sel[0]] + pasted + query[sel[1] :]
                            cursor_pos = sel[0] + len(pasted)
                            clear_selection()
                        else:
                            query = query[:cursor_pos] + pasted + query[cursor_pos:]
                            cursor_pos += len(pasted)
                        sync_mouse_mode()
                        selected = 0
                        offset = 0
                        continue
                    if ev == "paste_text":
                        pasted = normalize_pasted_text(str(payload))
                        if not pasted:
                            continue
                        sel = selection_bounds(sel_anchor, sel_end)
                        if sel:
                            query = query[: sel[0]] + pasted + query[sel[1] :]
                            cursor_pos = sel[0] + len(pasted)
                            clear_selection()
                        else:
                            query = query[:cursor_pos] + pasted + query[cursor_pos:]
                            cursor_pos += len(pasted)
                        sync_mouse_mode()
                        selected = 0
                        offset = 0
                        continue
                    if ev == "mouse":
                        bstate, mx, my, action = payload  # type: ignore[misc]
                        if bstate & 64:
                            # Mouse wheel: mirror up/down arrow behavior.
                            if action != "M":
                                continue
                            wheel_button = bstate & 3
                            if wheel_button == 0:
                                selected = max(0, selected - 1)
                            elif wheel_button == 1:
                                selected = min(max(0, len(results) - 1), selected + 1)
                            continue
                        button = bstate & 3
                        is_motion = bool(bstate & 32)
                        is_shift = bool(bstate & 4)
    
                        # SGR mouse uses 'M' for press/motion, 'm' for release.
                        if action == "m":
                            if button in (0, 3):
                                mouse_selecting = False
                            continue
                        if action != "M":
                            continue
    
                        # Query line interactions (including wrapped rows).
                        if anchor_row <= my < (anchor_row + query_rows_used):
                            click_row = my - anchor_row
                            click_col = max(0, mx - anchor_col - 1)
                            click_pos = query_pos_from_visual(
                                query,
                                render_width,
                                query_start,
                                click_row,
                                click_col,
                            )
    
                            if is_motion:
                                if mouse_selecting:
                                    move_cursor(click_pos, select_mode=True)
                                continue
    
                            if button == 0:
                                now = time.monotonic()
                                is_same_click_area = (
                                    (now - last_left_click_time) <= 0.35
                                    and my == last_left_click_row
                                    and abs(mx - last_left_click_col) <= 1
                                )
                                if is_same_click_area:
                                    left_click_count += 1
                                else:
                                    left_click_count = 1
                                last_left_click_time = now
                                last_left_click_row = my
                                last_left_click_col = mx
    
                                if left_click_count >= 3 and query:
                                    select_all_query()
                                    mouse_selecting = False
                                elif left_click_count == 2 and query:
                                    # Select the contiguous run under the cursor:
                                    # either non-whitespace ("word") or whitespace.
                                    left = click_pos
                                    right = click_pos
                                    if click_pos < len(query):
                                        select_whitespace = query[click_pos].isspace()
                                        while left > 0 and query[left - 1].isspace() == select_whitespace:
                                            left -= 1
                                        right = click_pos + 1
                                        while right < len(query) and query[right].isspace() == select_whitespace:
                                            right += 1
                                    else:
                                        while left > 0 and not query[left - 1].isspace():
                                            left -= 1
                                    if left != right:
                                        sel_anchor = left
                                        sel_end = right
                                        cursor_pos = right
                                    else:
                                        move_cursor(click_pos, select_mode=is_shift)
                                else:
                                    move_cursor(click_pos, select_mode=is_shift)
                                    mouse_selecting = True
                                continue
    
                        # Ignore result-line clicks; selection/accept remains
                        # keyboard-driven (arrows + Enter/Tab).
                        if my >= (anchor_row + query_rows_used) and my < anchor_row + panel_rows and not is_motion and button == 0:
                            continue
            except KeyboardInterrupt:
                clear_panel_and_restore_cursor()
                return None
            finally:
                search_stop.set()
                search_requests.put(None)

            clear_panel_and_restore_cursor()
            return chosen
    finally:
        TERM_OUT = sys.stdout
        if tty_in_file is not None:
            tty_in_file.close()
        if tty_out_file is not None:
            tty_out_file.close()


def main() -> int:
    parser = ArgumentParser(add_help=True)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print selected command to stdout instead of executing it.",
    )
    parser.add_argument("--daemon", action="store_true", help=SUPPRESS)
    parser.add_argument("--socket-path", default="", help=SUPPRESS)
    parser.add_argument("--history-file", default="", help=SUPPRESS)
    parser.add_argument(
        "--history-length",
        default="10k",
        help="Maximum SQLite history rows to keep when the daemon starts (for example: 10000 or 10k).",
    )
    parser.add_argument(
        "--no-shared-daemon",
        action="store_true",
        help="Error out because local fallback mode is disabled.",
    )
    parser.add_argument(
        "--debug-daemon",
        action="store_true",
        help="Print daemon connection/startup diagnostics to stderr.",
    )
    parser.add_argument(
        "--use-custom-history",
        action="store_true",
        help="Use local SQLite history at ./history.db (command, cwd, timestamp).",
    )
    args = parser.parse_args()
    try:
        history_length = parse_history_length_arg(str(args.history_length))
    except ValueError as exc:
        print(f"zsh_flex_history: {exc}", file=sys.stderr)
        return 2

    if args.use_custom_history:
        history_path = default_custom_history_path()
        try:
            ensure_custom_history_file(history_path)
        except OSError as exc:
            print(f"zsh_flex_history: failed to initialize custom history file: {exc}", file=sys.stderr)
            return 1
    else:
        history_path_value = args.history_file or os.environ.get("HISTFILE", str(Path.home() / ".zsh_history"))
        history_path = Path(history_path_value).expanduser()
    socket_path = (
        Path(args.socket_path).expanduser()
        if args.socket_path
        else default_daemon_socket_path(use_custom_history=args.use_custom_history)
    )

    if args.daemon:
        return run_history_daemon(
            history_path,
            socket_path,
            debug=args.debug_daemon,
            history_length=history_length,
            use_custom_history=args.use_custom_history,
        )

    history_client: Optional[HistoryDaemonClient] = None
    history_updates: Optional[queue.Queue[tuple[str, object]]] = None
    if not args.no_shared_daemon:
        daemon_client = HistoryDaemonClient(
            socket_path,
            history_path,
            Path(__file__).resolve(),
            debug=args.debug_daemon,
            history_length=history_length,
            use_custom_history=args.use_custom_history,
        )
        if daemon_client.ensure_running():
            history_client = daemon_client
            daemon_debug_log(args.debug_daemon, "shared daemon mode enabled")
        else:
            print("zsh_flex_history: daemon unavailable (no local fallback enabled)", file=sys.stderr)
            return 1
    else:
        print("zsh_flex_history: --no-shared-daemon is not supported (no local fallback enabled)", file=sys.stderr)
        return 1

    selected = run(
        [],
        inline_with_prompt=args.print_only,
        history_updates=history_updates,
        history_client=history_client,
    )
    if selected:
        selected = selected.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        if not selected.strip():
            return 1
        if args.use_custom_history:
            append_custom_history_entry(
                history_path,
                selected,
                os.getcwd(),
                datetime.now(timezone.utc).isoformat(),
            )
        if args.print_only:
            print(selected)
            return 0
        shell = os.environ.get("SHELL", "/bin/zsh")
        print(f"$ {selected}")
        completed = subprocess.run([shell, "-lc", selected])
        return completed.returncode
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
