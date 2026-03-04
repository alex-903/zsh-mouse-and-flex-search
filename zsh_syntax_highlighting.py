#!/usr/bin/env python3
"""Basic zsh syntax highlighting helpers for interactive query rendering."""

from __future__ import annotations

from functools import lru_cache
import os
import re
import shutil


ANSI_STYLE_BY_TOKEN = {
    "default": "",
    "command": "\x1b[32m",      # green
    "keyword": "\x1b[34m",      # blue
    "option": "\x1b[36m",       # cyan
    "string": "\x1b[33m",       # yellow
    "variable": "\x1b[35m",     # magenta
    "operator": "",             # default
    "comment": "\x1b[90m",      # bright black
    "assignment": "",           # default
    "error": "",                # default (no special error coloring)
}


KEYWORDS = {
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "for",
    "while",
    "until",
    "do",
    "done",
    "in",
    "case",
    "esac",
    "select",
    "function",
    "time",
    "coproc",
    "repeat",
    "noglob",
    "builtin",
    "command",
    "exec",
    "eval",
    "source",
    ".",
}


OPERATORS = tuple(
    sorted(
        [
            "&&",
            "||",
            ";;",
            "<<-",
            "<<",
            ">>",
            "<&",
            ">&",
            "|",
            ";",
            "&",
            "(",
            ")",
            "{",
            "}",
            "<",
            ">",
        ],
        key=len,
        reverse=True,
    )
)


COMMAND_SEPARATORS = {"&&", "||", "|", ";", "&", "(", ")"}
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
AMBIGUOUS_COMMAND_RE = re.compile(r"[\\$*?[\]{}()'\"`=]")

# Common zsh builtins and shell-resolved command words.
BUILTINS = {
    "alias",
    "autoload",
    "bg",
    "bindkey",
    "break",
    "builtin",
    "bye",
    "cd",
    "chdir",
    "command",
    "compgen",
    "complete",
    "continue",
    "declare",
    "dirs",
    "disable",
    "disown",
    "echo",
    "echotc",
    "emulate",
    "enable",
    "eval",
    "exec",
    "exit",
    "export",
    "false",
    "fc",
    "fg",
    "functions",
    "getopts",
    "hash",
    "history",
    "jobs",
    "kill",
    "let",
    "limit",
    "local",
    "logout",
    "popd",
    "print",
    "printf",
    "pushd",
    "pwd",
    "read",
    "readonly",
    "rehash",
    "return",
    "set",
    "setopt",
    "shift",
    "source",
    "suspend",
    "test",
    "times",
    "trap",
    "true",
    "type",
    "typeset",
    "ulimit",
    "umask",
    "unalias",
    "unfunction",
    "unset",
    "unsetopt",
    "wait",
    "whence",
    "where",
    "which",
    "zmodload",
}


def ansi_for_token(token: str) -> str:
    return ANSI_STYLE_BY_TOKEN.get(token, "")


def highlight_tokens(query: str) -> list[str]:
    tokens = ["default"] * len(query)
    if not query:
        return tokens

    i = 0
    expect_command = True

    while i < len(query):
        ch = query[i]
        if ch.isspace():
            i += 1
            continue

        if _is_comment_start(query, i):
            _mark(tokens, i, len(query), "comment")
            break

        op = _match_operator(query, i)
        if op is not None:
            _mark(tokens, i, i + len(op), "operator")
            if op in COMMAND_SEPARATORS:
                expect_command = True
            i += len(op)
            continue

        if ch in ("'", '"', "`"):
            end = _scan_quoted(query, i, ch)
            _mark(tokens, i, end, "string")
            i = end
            expect_command = False
            continue

        if ch == "$":
            end = _scan_dollar_expr(query, i)
            _mark(tokens, i, end, "variable")
            i = end
            expect_command = False
            continue

        start = i
        while i < len(query):
            if query[i].isspace():
                break
            if _match_operator(query, i) is not None:
                break
            if query[i] in ("'", '"', "`"):
                i = _scan_quoted(query, i, query[i])
                continue
            if query[i] == "$":
                i = _scan_dollar_expr(query, i)
                continue
            if query[i] == "\\" and i + 1 < len(query):
                i += 2
                continue
            i += 1

        word = query[start:i]
        word_complete = False
        if i < len(query):
            next_ch = query[i]
            if next_ch.isspace():
                word_complete = True
            elif _match_operator(query, i) is not None:
                word_complete = True
            elif _is_comment_start(query, i):
                word_complete = True
        kind = _classify_word(word, expect_command, word_complete)
        _mark(tokens, start, i, kind)

        if kind == "assignment":
            expect_command = True
        elif kind == "keyword":
            # `in` (for/case lists) expects patterns/words, not a command.
            expect_command = word in {"then", "do", "else", "elif", "time"}
        else:
            expect_command = False

    return tokens


def _mark(tokens: list[str], start: int, end: int, kind: str) -> None:
    start = max(0, start)
    end = min(len(tokens), end)
    for idx in range(start, end):
        tokens[idx] = kind


def _is_comment_start(text: str, index: int) -> bool:
    if text[index] != "#":
        return False
    if index == 0:
        return True
    prev = text[index - 1]
    return prev.isspace() or prev in ";|&(){}"


def _match_operator(text: str, index: int) -> str | None:
    for op in OPERATORS:
        if text.startswith(op, index):
            return op
    return None


def _scan_quoted(text: str, start: int, quote: str) -> int:
    i = start + 1
    while i < len(text):
        ch = text[i]
        if quote != "'" and ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == quote:
            return i + 1
        i += 1
    # Incomplete quote while typing is not necessarily an error;
    # keep highlighting the rest as a string.
    return len(text)


def _scan_dollar_expr(text: str, start: int) -> int:
    if start + 1 >= len(text):
        return start + 1

    nxt = text[start + 1]
    if nxt == "{":
        return _scan_braced(text, start + 2)
    if nxt == "(":
        if start + 2 < len(text) and text[start + 2] == "(":
            return _scan_arithmetic(text, start + 3)
        return _scan_subshell(text, start + 2)

    if nxt.isdigit() or nxt in "@*#?$!-_":
        return start + 2

    if nxt.isalpha() or nxt == "_":
        i = start + 2
        while i < len(text) and (text[i].isalnum() or text[i] == "_"):
            i += 1
        return i

    return start + 1


def _scan_braced(text: str, start: int) -> int:
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def _scan_subshell(text: str, start: int) -> int:
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch in ("'", '"', "`"):
            i = _scan_quoted(text, i, ch)
            continue
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def _scan_arithmetic(text: str, start: int) -> int:
    depth = 1
    i = start
    while i < len(text):
        if text.startswith("((", i):
            depth += 1
            i += 2
            continue
        if text.startswith("))", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
            continue
        if text[i] == "\\" and i + 1 < len(text):
            i += 2
            continue
        i += 1
    return len(text)


def _classify_word(word: str, expect_command: bool, word_complete: bool) -> str:
    if not word:
        return "default"
    if word in KEYWORDS:
        return "keyword"
    if ASSIGNMENT_RE.match(word):
        return "assignment"
    if word.startswith("-") and len(word) > 1:
        return "option"
    if expect_command:
        state = _command_state(word, word_complete)
        if state == "valid":
            return "command"
        if state == "error":
            return "error"
        return "default"
    return "default"


def _command_state(word: str, word_complete: bool) -> str:
    # Avoid false positives for ambiguous/incomplete shell forms while typing.
    if not word or AMBIGUOUS_COMMAND_RE.search(word):
        return "pending"

    if _is_valid_command(word):
        return "valid"

    # For in-progress typing, unknown non-prefix commands are likely errors.
    # Keep known prefixes neutral until command token is complete.
    if not word_complete:
        if _is_known_command_prefix(word) or _is_existing_path_prefix(word):
            return "pending"
        return "error"

    if word_complete:
        return "error"
    return "pending"


@lru_cache(maxsize=4096)
def _which_cached(path_env: str, word: str) -> str | None:
    return shutil.which(word, path=path_env)


def _is_valid_command(word: str) -> bool:
    if word in KEYWORDS or word in BUILTINS:
        return True
    if "/" in word:
        path = os.path.expanduser(word)
        return os.path.isfile(path) and os.access(path, os.X_OK)
    return _which_cached(os.environ.get("PATH", ""), word) is not None


def _is_known_command_prefix(prefix: str) -> bool:
    if not prefix:
        return False
    if any(k.startswith(prefix) for k in KEYWORDS):
        return True
    if any(b.startswith(prefix) for b in BUILTINS):
        return True
    path_env = os.environ.get("PATH", "")
    return _path_has_prefix(path_env, prefix)


def _is_existing_path_prefix(text: str) -> bool:
    if not text:
        return False
    if "/" not in text and not text.startswith("~"):
        return False

    expanded = os.path.expanduser(text)
    parent_part, sep, name_prefix = expanded.rpartition("/")
    if sep:
        base_dir = parent_part or "/"
    else:
        base_dir = "."
        name_prefix = expanded

    try:
        with os.scandir(base_dir) as entries:
            for entry in entries:
                if entry.name.startswith(name_prefix):
                    return True
    except OSError:
        return False
    return False


@lru_cache(maxsize=4096)
def _path_has_prefix(path_env: str, prefix: str) -> bool:
    if not prefix:
        return False
    for path_dir in path_env.split(os.pathsep):
        if not path_dir:
            continue
        try:
            with os.scandir(path_dir) as entries:
                for entry in entries:
                    if not entry.name.startswith(prefix):
                        continue
                    try:
                        if entry.is_file() and os.access(entry.path, os.X_OK):
                            return True
                    except OSError:
                        continue
        except OSError:
            continue
    return False
