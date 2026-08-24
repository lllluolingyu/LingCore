Work in small, verifiable steps. Before editing a file, read it. After changing
code, run the relevant build or tests to confirm it works. When you make
`edit_file` calls, the `old` text must be unique in the file. Prefer running
commands over guessing at their output.
Prefer `search` for workspace grep and recursive filename lookup because shell
commands require confirmation in this profile.
