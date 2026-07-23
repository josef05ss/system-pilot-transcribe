$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location "$Root\backend"
$env:PYTHONPATH = "$Root\backend"
& "$Root\.venv\Scripts\python.exe" -m celery -A app.tasks.celery_app:celery_app worker --pool=solo -Q cpu --hostname=cpu@%h --concurrency=1 --loglevel=INFO
