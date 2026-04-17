# Pull the SQLite chat/feedback DB from prod (labus_api container) and
# materialise it into this repo under _dbdump/<YYYY-MM-DD>/.
#
# Strategy: docker cp the .db (+ optional -wal / -shm sidecars) directly.
# No sqlite3 CLI needed inside the container. On first open, SQLite
# checkpoints WAL into the main DB on the Windows side.
#
# Usage:   powershell -ExecutionPolicy Bypass -File scripts\pull_chats_from_prod.ps1
# Env overrides:
#   $env:PROD_HOST      (default: root@62.217.178.117)
#   $env:PROD_PORT      (default: 22)
#   $env:API_CONTAINER  (default: labus_api)
#   $env:CONTAINER_DB   (default: /app/data/labus_rag.db)

$ErrorActionPreference = "Stop"

$ProdHost     = if ($env:PROD_HOST)     { $env:PROD_HOST }     else { "root@62.217.178.117" }
$ProdPort     = if ($env:PROD_PORT)     { $env:PROD_PORT }     else { "22" }
$ApiContainer = if ($env:API_CONTAINER) { $env:API_CONTAINER } else { "labus_api" }
$ContainerDb  = if ($env:CONTAINER_DB)  { $env:CONTAINER_DB }  else { "/app/data/labus_rag.db" }

$Today    = Get-Date -Format "yyyy-MM-dd"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$OutDir   = Join-Path $RepoRoot "_dbdump\$Today"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$RemoteDir = "/tmp/labus_rag_$Today"

Write-Host "[1/4] Copying DB file(s) out of container $ApiContainer..."
# Use a single shell call: create tmp dir, docker cp main db, and sidecars (wal/shm) if they exist.
$remoteSetup = "mkdir -p $RemoteDir && docker cp ${ApiContainer}:${ContainerDb} $RemoteDir/labus_rag.db && (docker cp ${ApiContainer}:${ContainerDb}-wal $RemoteDir/labus_rag.db-wal 2>/dev/null || true) && (docker cp ${ApiContainer}:${ContainerDb}-shm $RemoteDir/labus_rag.db-shm 2>/dev/null || true) && ls -la $RemoteDir"
& ssh -p $ProdPort $ProdHost $remoteSetup
if ($LASTEXITCODE -ne 0) { throw "docker cp step failed (exit $LASTEXITCODE)" }

Write-Host "[2/4] scp to $OutDir..."
& scp -P $ProdPort "${ProdHost}:$RemoteDir/labus_rag.db" (Join-Path $OutDir "labus_rag.db")
if ($LASTEXITCODE -ne 0) { throw "scp .db failed (exit $LASTEXITCODE)" }

# WAL/SHM are optional - SQLite only creates them when a writer is holding a
# transaction at the moment we snapshotted. Pre-check existence on the remote
# so we don't make scp emit "No such file or directory" noise for the common
# case where they don't exist.
$remoteListing = & ssh -p $ProdPort $ProdHost "ls -1 $RemoteDir"
$remoteFiles = @($remoteListing -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($side in @("labus_rag.db-wal", "labus_rag.db-shm")) {
    if ($remoteFiles -contains $side) {
        & scp -P $ProdPort "${ProdHost}:$RemoteDir/$side" (Join-Path $OutDir $side)
        if ($LASTEXITCODE -ne 0) { throw "scp $side failed (exit $LASTEXITCODE)" }
    } else {
        Write-Host "  (skip $side - not present on remote; WAL already checkpointed)"
    }
}

Write-Host "[3/4] Checkpointing WAL locally (if sidecars present)..."
$dbLocal = (Join-Path $OutDir "labus_rag.db").Replace('\','/')
# Python's sqlite3 auto-consolidates WAL on first write connection.
& python -c "import sqlite3; c=sqlite3.connect(r'$dbLocal'); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.commit(); c.close(); print('checkpoint ok')"

Write-Host "[4/4] Cleaning remote tmp..."
& ssh -p $ProdPort $ProdHost "rm -rf $RemoteDir"

Write-Host ""
Write-Host "OK. Pulled to $OutDir"
Get-ChildItem $OutDir | Format-Table Name, Length, LastWriteTime
Write-Host ""
Write-Host "Next:"
Write-Host "  python RAG_RUNTIME\scripts\export_chats.py $OutDir\labus_rag.db"
