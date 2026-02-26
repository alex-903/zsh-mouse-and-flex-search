#!/usr/bin/env python3
"""Interactive zsh history search with Emacs-like flex matching."""

from __future__ import annotations

import os
import re
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


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


@dataclass
class MatchResult:
    text: str
    score: int
    positions: List[int]
    exact: bool = False
    recency: int = 0


def base16_ansi(name: str) -> int:
    return BASE16_TO_ANSI[name]


def fg_code(slot: int) -> str:
    if 0 <= slot <= 7:
        return str(30 + slot)
    if 8 <= slot <= 15:
        return str(90 + (slot - 8))
    return "39"


def style(*, fg: Optional[int] = None, bold: bool = False, underline: bool = False) -> str:
    codes: list[str] = []
    if bold:
        codes.append("1")
    if underline:
        codes.append("4")
    if fg is not None:
        codes.append(fg_code(fg))
    if not codes:
        return ""
    return f"\x1b[{';'.join(codes)}m"


RESET = "\x1b[0m"
REVERSE = "\x1b[7m"
CLEAR_LINE = "\x1b[2K"
CLEAR_TO_END = "\x1b[K"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
ENABLE_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1006h"
DISABLE_MOUSE = "\x1b[?1000l\x1b[?1002l\x1b[?1006l"

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
    return last_match


def load_history(path: Path) -> list[str]:
    entries: list[str] = []
    if not path.exists():
        return entries

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # zsh extended history format: ': 1700012345:0;command'
            if line.startswith(": "):
                semicolon = line.find(";")
                if semicolon != -1:
                    cmd = line[semicolon + 1 :].strip()
                    if cmd:
                        entries.append(cmd)
                        continue
            entries.append(line)

    dedup: list[str] = []
    seen = set()
    for cmd in reversed(entries):
        if cmd in seen:
            continue
        seen.add(cmd)
        dedup.append(cmd)
    return dedup


def flex_match(query: str, candidate: str) -> Optional[MatchResult]:
    if not query:
        return MatchResult(candidate, 0, [])

    q = query.lower()
    c = candidate.lower()

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
    compact_bonus = max(0, 20 - (span - len(query)))

    score += contiguous + boundary_bonus + start_bonus + compact_bonus
    score -= gap_penalty
    score -= len(candidate) // 8

    return MatchResult(candidate, score, positions)


def search(query: str, history: Iterable[str], limit: int = 200) -> list[MatchResult]:
    results: list[MatchResult] = []
    normalized_query = query.strip().lower()
    for i, cmd in enumerate(history):
        m = flex_match(query, cmd)
        if m is not None:
            m.exact = bool(normalized_query) and cmd.strip().lower() == normalized_query
            # load_history() already returns commands in most-recent-first order.
            m.recency = -i
            results.append(m)

    results.sort(key=lambda m: (m.exact, m.recency), reverse=True)
    return results[:limit]


def truncate_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    return text[:width]


def query_window(query: str, cursor_pos: int, available: int) -> tuple[int, str]:
    if available <= 0:
        return 0, ""
    max_start = max(0, len(query) - available)
    start = min(max(0, cursor_pos - available + 1), max_start)
    return start, query[start : start + available]


def selection_bounds(sel_anchor: Optional[int], sel_end: Optional[int]) -> Optional[tuple[int, int]]:
    if sel_anchor is None or sel_end is None:
        return None
    if sel_anchor == sel_end:
        return None
    return (min(sel_anchor, sel_end), max(sel_anchor, sel_end))


def render_result_line(item: MatchResult, selected: bool, width: int, *, unselected_white: bool = False) -> str:
    if width <= 0:
        return ""

    body_width = width
    text = truncate_text(item.text, body_width)
    pos_set = set(item.positions)

    sel_fg = base16_ansi("base0E")
    match_fg = base16_ansi("base0D")

    if selected:
        normal_style = style(fg=sel_fg, bold=True)
    elif unselected_white:
        normal_style = style(fg=base16_ansi("base05"))
    else:
        normal_style = ""
    match_style = style(fg=match_fg, bold=True)

    out = [normal_style]
    for i, ch in enumerate(text):
        out.append(match_style if i in pos_set else normal_style)
        out.append(ch)
    out.append(RESET)
    return "".join(out)


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
) -> tuple[int, int]:
    anchor_col = max(1, anchor_col)
    render_width = max(1, width - anchor_col + 1)
    visible = max(1, panel_rows - 1)
    muted = style(fg=base16_ansi("base03"))

    lines: list[str] = []
    cursor_pos = max(0, min(cursor_pos, len(query)))
    query_width = render_width
    query_start, query_view = query_window(query, cursor_pos, query_width)
    sel = selection_bounds(sel_anchor, sel_end)
    query_parts: list[str] = []
    for i, ch in enumerate(query_view):
        qidx = query_start + i
        if sel and sel[0] <= qidx < sel[1]:
            query_parts.append(f"{REVERSE}{ch}{RESET}")
        else:
            query_parts.append(ch)
    lines.append("".join(query_parts))

    for i in range(visible):
        idx = offset + i
        if idx >= len(results):
            lines.append("")
            continue
        remaining = max(0, len(results) - (offset + visible))
        is_last_visible_row = i == (visible - 1)
        more_text = f"{remaining} more" if (is_last_visible_row and remaining > 0) else ""
        suffix = f"    {muted}{more_text}{RESET}" if more_text else ""
        result_width = max(0, render_width - len(more_text) - (4 if more_text else 0))
        base_line = render_result_line(
            results[idx],
            idx == selected,
            result_width,
            unselected_white=True,
        )
        lines.append(f"{base_line}{suffix}")

    for i, line in enumerate(lines[:panel_rows]):
        term_write(move_to(anchor_row + i, anchor_col) + CLEAR_TO_END + line)

    # Put cursor on query input field.
    cursor_in_view = max(0, min(len(query_view), cursor_pos - query_start))
    query_col = anchor_col + cursor_in_view
    if query_width > 0:
        query_col = min(query_col, anchor_col + query_width - 1)
    else:
        query_col = anchor_col
    term_write(move_to(anchor_row, query_col))
    term_flush()
    return query_start, len(query_view)


def read_key(fd: int) -> tuple[str, object]:
    def read_escape_tail() -> bytes:
        # Read an escape sequence incrementally; macOS Terminal can split
        # bytes across small intervals, so a single short poll is unreliable.
        seq = b""
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            rdy, _, _ = select.select([fd], [], [], 0.01)
            if not rdy:
                if seq:
                    break
                continue
            chunk = os.read(fd, 64)
            if not chunk:
                break
            seq += chunk
            # Common CSI/SS3 final-byte termination.
            if seq[-1:] and (64 <= seq[-1] <= 126):
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

    while True:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            continue
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
            continue
        if 32 <= ch < 127:
            return "char", chr(ch)


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


def run(history: list[str], *, inline_with_prompt: bool = False) -> Optional[str]:
    global TERM_OUT
    tty_in_file = None
    tty_out_file = None
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
    panel_rows = 4

    try:
        with RawTerminal(fd) as rt:
            term_lines = shutil.get_terminal_size((120, 24)).lines
            pos = query_cursor_position(fd)
            desired_rows = max(1, min(panel_rows, term_lines))
            if pos is None:
                start_row = max(1, term_lines - 1)
                start_col = 1
            else:
                start_row = pos[0]
                start_col = pos[1]
            # Never scroll the terminal to create room.
            # For print-only mode, anchor on the prompt row itself so query
            # input starts on the same line as the prompt.
            # Otherwise, use the row below the prompt when possible.
            space_below = max(0, term_lines - start_row)
            if inline_with_prompt:
                anchor_row = max(1, start_row)
                anchor_col = max(1, start_col)
                panel_rows = max(1, min(desired_rows, term_lines - anchor_row + 1))
            elif space_below >= 2:
                anchor_row = start_row + 1
                anchor_col = 1
                panel_rows = max(1, min(desired_rows, space_below))
            else:
                anchor_row = max(1, start_row)
                anchor_col = 1
                panel_rows = max(1, min(desired_rows, term_lines - anchor_row + 1))
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
            mouse_selecting = False
            mouse_enabled = False
            last_left_click_time = 0.0
            last_left_click_row = -1
            last_left_click_col = -1

            while True:
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

                width = shutil.get_terminal_size((120, 24)).columns
                visible = max(1, panel_rows - 1)

                results = search(query, history, limit=500)
                if selected >= len(results):
                    selected = max(0, len(results) - 1)
                if selected < offset:
                    offset = selected
                if selected >= offset + visible:
                    offset = selected - visible + 1

                query_start, _query_view_len = draw_panel(
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
                )

                ev, payload = read_key(fd)

                if ev == "quit":
                    break
                if ev == "enter":
                    if 0 <= selected < len(results):
                        chosen = results[selected].text
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
                if ev == "up":
                    selected = min(max(0, len(results) - 1), selected + 1)
                    continue
                if ev == "down":
                    selected = max(0, selected - 1)
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
                if ev == "mouse":
                    bstate, mx, my, action = payload  # type: ignore[misc]
                    # Ignore wheel events so mouse wheel does not navigate results.
                    if bstate & 64:
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

                    # Query line interactions.
                    if my == anchor_row:
                        click_col = max(anchor_col, mx)
                        click_pos = query_start + max(0, click_col - anchor_col)
                        click_pos = max(0, min(click_pos, len(query)))

                        if is_motion:
                            if mouse_selecting:
                                move_cursor(click_pos, select_mode=True)
                            continue

                        if button == 0:
                            now = time.monotonic()
                            is_double_click = (
                                (now - last_left_click_time) <= 0.35
                                and my == last_left_click_row
                                and abs(mx - last_left_click_col) <= 1
                            )
                            last_left_click_time = now
                            last_left_click_row = my
                            last_left_click_col = mx

                            if is_double_click and query:
                                # Select a "word" delimited by whitespace.
                                left = click_pos
                                while left > 0 and not query[left - 1].isspace():
                                    left -= 1
                                right = click_pos
                                while right < len(query) and not query[right].isspace():
                                    right += 1
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

                    # Result line click: select and accept current item.
                    if my == anchor_row + 1 and results and not is_motion and button == 0:
                        selected = min(len(results) - 1, max(offset, 0))
                        chosen = results[selected].text
                        break
        # Clear panel content so repeated invocations always start clean.
        for row in range(anchor_row, anchor_row + panel_rows):
            term_write(move_to(row, anchor_col) + CLEAR_TO_END)

        # Restore cursor to the exact prompt position captured at invocation start.
        term_write(move_to(start_row, start_col))
        term_flush()
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
    args = parser.parse_args()

    history_path = Path(os.environ.get("HISTFILE", str(Path.home() / ".zsh_history"))).expanduser()
    history = load_history(history_path)
    if not history:
        print(f"No zsh history found at {history_path}", file=sys.stderr)
        return 1

    selected = run(history, inline_with_prompt=args.print_only)
    if selected:
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
