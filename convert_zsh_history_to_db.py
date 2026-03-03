#!/usr/bin/env python3
"""Convert zsh history (plain + extended/mixed) into SQLite history.db."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


HEADER_RE = re.compile(r"^: (\d+):\d+;(.*)$")


def default_input_history_path() -> Path:
    return Path(os.environ.get("HISTFILE", str(Path.home() / ".zsh_history"))).expanduser()


def default_output_db_path() -> Path:
    return Path(__file__).resolve().parent / "history.db"


def ensure_schema(conn: sqlite3.Connection) -> None:
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


def epoch_to_iso(epoch_text: str) -> str:
    try:
        epoch = int(epoch_text)
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return ""


def normalize_command(command: str) -> str:
    return command.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip("\n")


def parse_mixed_zsh_history(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8", errors="replace")
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")

    entries: list[tuple[str, str]] = []
    current_extended_command: str | None = None
    current_extended_timestamp = ""

    def flush_extended() -> None:
        nonlocal current_extended_command, current_extended_timestamp
        if current_extended_command is None:
            return
        cmd = normalize_command(current_extended_command)
        if cmd.strip():
            entries.append((cmd, current_extended_timestamp))
        current_extended_command = None
        current_extended_timestamp = ""

    for line in normalized.split("\n"):
        match = HEADER_RE.match(line)
        if match:
            flush_extended()
            current_extended_timestamp = epoch_to_iso(match.group(1))
            current_extended_command = match.group(2)
            continue

        if current_extended_command is not None:
            # In mixed files, only treat the next line as continuation when
            # the previous extended command line explicitly continues.
            if current_extended_command.endswith("\\"):
                current_extended_command += "\n" + line
                continue
            flush_extended()

        plain = normalize_command(line)
        if plain.strip():
            entries.append((plain, ""))

    flush_extended()
    return entries


def import_history_to_db(
    source_history: Path,
    target_db: Path,
    *,
    append: bool,
) -> tuple[int, int]:
    entries = parse_mixed_zsh_history(source_history)
    target_db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(target_db) as conn:
        ensure_schema(conn)
        if not append:
            conn.execute("DELETE FROM custom_history")
        if entries:
            conn.executemany(
                "INSERT INTO custom_history(command, cwd, timestamp) VALUES(?, ?, ?)",
                [(command, "", timestamp) for command, timestamp in entries],
            )
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM custom_history").fetchone()
    row_count = int(total[0]) if total else 0
    return len(entries), row_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert zsh history (plain+extended mixed) into SQLite history.db.",
    )
    parser.add_argument(
        "--source",
        default=str(default_input_history_path()),
        help="Path to zsh history file (default: $HISTFILE or ~/.zsh_history).",
    )
    parser.add_argument(
        "--target",
        default=str(default_output_db_path()),
        help="Path to output SQLite DB (default: ./history.db).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append into existing DB rows instead of replacing table contents.",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    target = Path(args.target).expanduser()

    imported, total = import_history_to_db(source, target, append=args.append)
    print(f"Imported {imported} entries into {target} (total rows: {total}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
