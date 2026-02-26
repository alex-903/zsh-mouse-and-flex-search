fh-line-init() {
  [[ -n ${widgets[fh-orig-line-init]} ]] && zle fh-orig-line-init

  local cmd
  cmd="$("/Users/alex/Desktop/git/untitled folder/zsh_flex_history.py" --print-only 2>/dev/null)" || return
  [[ -z "$cmd" ]] && return

  BUFFER="$cmd"
  CURSOR=${#BUFFER}
  zle redisplay      # show prompt + command
  zle -U $'\n'       # auto-press Enter after prompt is shown
}

zle -N zle-line-init fh-line-init