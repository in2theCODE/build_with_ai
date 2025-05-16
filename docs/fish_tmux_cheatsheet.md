# Fish Shell and Tmux Cheat Sheet

## Fish Shell Basics
Command | Purpose
--------|--------
`set -Ux VAR value` | Set a universal (persistent) variable
`set -gx VAR value` | Set and export global variable (like PATH)
`abbr -a gs 'git status'` | Create a global shortcut/alias
`funced fish_greeting` | Edit a fish function
`fish_config` | Launch Fish config UI in browser

## Common Fish Config (`~/.config/fish/config.fish`)
Command | Value
--------|------
`set -gx EDITOR` | `nvim`
`set -gx LANG` | `en_US.UTF-8`
`set -gx PATH` | `$HOME/.local/bin $PATH`
`set -g fish_key_bindings` | `fish_vi_key_bindings`

## Tmux Core Commands
Command | Description
--------|------------
`tmux` | Start tmux session
`tmux attach` | Reattach to last session
`tmux ls` | List all sessions
`tmux kill-session -t name` | Kill a session
`tmux source-file ~/.tmux.conf` | Reload config

## Tmux Keybindings (Prefix: `Ctrl-A`)
Action | Keybinding
--------|-----------
Prefix key | `Ctrl-A`
New window | `Ctrl-A c`
Next window | `Ctrl-A n`
Previous window | `Ctrl-A p`
Split pane vertical | `Ctrl-A |`
Split pane horizontal | `Ctrl-A -`
Switch panes | `Ctrl-A Arrow`
Resize pane | `Ctrl-A Ctrl + Arrow`
Close pane | `Ctrl-A x`
Reload config | `Ctrl-A r`

## Tmux Plugin Manager (TPM)
Setting | Value
--------|------
Plugin directory | `~/.tmux/plugins/`
Plugin setup | `set -g @plugin 'tmux-plugins/tpm'`
Sensible defaults | `set -g @plugin 'tmux-plugins/tmux-sensible'`
Run TPM | `run '~/.tmux/plugins/tpm/tpm'`
Install plugins | `Prefix + I`

## Troubleshooting
Problem | Fix
--------|----
Config not reloading | `tmux source-file ~/.tmux.conf`
Keybindings don’t work | Check prefix key & escape delay
Plugin manager not working | Ensure `~/.tmux/plugins/tpm/` exists
Color issues | `set -g default-terminal "screen-256color"`