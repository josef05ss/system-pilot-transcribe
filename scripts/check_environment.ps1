$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
& "$Root\.venv\Scripts\python.exe" "$Root\scripts\check_environment.py"
