# zsh mouse and flex history search

![zsh flex history screenshot](./screenshot.png)

A modernized terminal UI for searching zsh history with Emacs-style `flex` fuzzy matching, full mouse support for setting point and related interactions, and a Base16-driven color palette; in `.zshrc`, wire it via a `zle-line-init` hook (for example, `fh-line-init`) to run at prompt initialization.

## Run

```bash
./zsh_flex_history.py
```

Or:

```bash
python3 zsh_flex_history.py
```

## Behavior

- Loads history from `$HISTFILE` (or `~/.zsh_history`) on launch.
- Uses in-order flexible fuzzy matching (similar to Emacs `flex`).
- Shows a completing-read style vertical completion menu with highlighted match chars.
- Prioritizes first-token matches (command completion and matching command prefixes) ahead of deeper in-string matches, then scores by recency and query fit.
- For directory-aware prioritization, use `--use-custom-history` so history scoring can include current `cwd`, which improves relevance for repeated workflows per folder.
- Takes over mouse `x` from the native terminal app only when there is any text in the prompt.
- Syntax highlighting is "good enough" but incomplete

## Options


- `--use-custom-history`
  - Uses an alternate history backend at `history.db` (SQLite) in this project directory.
  - Stores commands as UTF-8 text by default, unlike zsh
  - Includes extra metadata per entry (`command`, `cwd`, `timestamp`).
- `--history-length <N>`
  - Maximum number of SQLite history rows to keep when the daemon starts (default: `10k`).
  - Accepts values like `10000` or `10k`.
  - Applies only to `--use-custom-history` and only when a daemon instance is starting; normal `~/.zsh_history` is not trimmed.
  - Existing daemon processes keep their current DB until restarted (new trim happens only on next startup).
- `--print-only`
  - Prints the selected command to stdout instead of executing it.




## Keys

- `Up` / `Down` / Scroll: move selection
- `Tab`: inserts selected command
- `PageUp` / `PageDown`: move faster
- `Backspace`: delete query char
- `Enter`: print and optionally runs the selected command
- `Esc` or `Ctrl-C`: quit
