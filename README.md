# zsh flex history search

Terminal UI for searching zsh history with Emacs-style `flex` fuzzy matching and a Base16-driven color palette.

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
- `thbesre` can match phrases like `the best restaurants in the town`.
- Shows a Corfu-style vertical completion menu with highlighted match chars.
- Uses Base16 env colors (`BASE16_COLOR_00`..`BASE16_COLOR_0F`) when available, else falls back to a default Base16 scheme.

## Keys

- `Up` / `Down`: move selection
- `PageUp` / `PageDown`: move faster
- `Backspace`: delete query char
- `Enter`: print selected command to stdout
- `Esc` or `Ctrl-C`: quit
- `q`: quit when query is empty
