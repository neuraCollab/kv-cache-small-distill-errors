# =====================================================================
# backup_to_desktop.ps1
# Backup of critical KV-quant research data before Windows reinstall.
# Creates backup_kv folder on Desktop and copies items NOT tracked in
# the git repository (recomputing them costs hours of CPU/GPU).
#
# Run:
#   powershell -ExecutionPolicy Bypass -File C:\Users\morro\prog\files\backup_to_desktop.ps1
# or simply:
#   & "C:\Users\morro\prog\files\backup_to_desktop.ps1"
# =====================================================================

$ErrorActionPreference = 'Continue'

# ---- Paths ----
$Source  = "C:\Users\morro\prog\files"
$Desktop = [Environment]::GetFolderPath("Desktop")
$Backup  = Join-Path $Desktop "backup_kv"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Backup KV-cache research data"               -ForegroundColor Cyan
Write-Host "  Source : $Source"                            -ForegroundColor Gray
Write-Host "  Target : $Backup"                            -ForegroundColor Gray
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# ---- Create root folder ----
if (-not (Test-Path $Backup)) {
    New-Item -ItemType Directory -Path $Backup -Force | Out-Null
    Write-Host "[+] Created folder $Backup" -ForegroundColor Green
}

# =====================================================================
# CRITICAL (~155 MB) - must be saved
# =====================================================================

Write-Host ""
Write-Host "[1/5] outputs\fdps  (FDP labels for 80 problems x 4 models x 4 quants)" -ForegroundColor Yellow
Copy-Item -Path (Join-Path $Source "outputs\fdps") `
          -Destination $Backup -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      done" -ForegroundColor Green

Write-Host ""
Write-Host "[2/5] outputs\traces  (bf16/quant token streams, ~111 MB)" -ForegroundColor Yellow
Copy-Item -Path (Join-Path $Source "outputs\traces") `
          -Destination $Backup -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      done" -ForegroundColor Green

Write-Host ""
Write-Host "[3/5] outputs\judgments  (LLM-judge error classifications)" -ForegroundColor Yellow
Copy-Item -Path (Join-Path $Source "outputs\judgments") `
          -Destination $Backup -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      done" -ForegroundColor Green

Write-Host ""
Write-Host "[4/5] outputs\paper  (final plots + supervisor_report.md)" -ForegroundColor Yellow
Copy-Item -Path (Join-Path $Source "outputs\paper") `
          -Destination $Backup -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      done" -ForegroundColor Green

# ---- outputs/kv_capture: only analysis/, NOT raw safetensors ----
Write-Host ""
Write-Host "[5/5] outputs\kv_capture\*\analysis\  (JSON/NPZ/plots only, no safetensors)" -ForegroundColor Yellow
$KvSrc = Join-Path $Source "outputs\kv_capture"
$KvDst = Join-Path $Backup "outputs\kv_capture"
if (-not (Test-Path $KvDst)) {
    New-Item -ItemType Directory -Path $KvDst -Force | Out-Null
}
Get-ChildItem -Path $KvSrc -Directory | ForEach-Object {
    $modelDir    = $_.Name
    $analysisSrc = Join-Path $_.FullName "analysis"
    if (Test-Path $analysisSrc) {
        $modelDst = Join-Path $KvDst $modelDir
        if (-not (Test-Path $modelDst)) {
            New-Item -ItemType Directory -Path $modelDst -Force | Out-Null
        }
        Copy-Item -Path $analysisSrc -Destination $modelDst -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "      $modelDir\analysis" -ForegroundColor Gray
    }
    # _run_metadata.json (if exists)
    $meta = Join-Path $_.FullName "_run_metadata.json"
    if (Test-Path $meta) {
        $modelDstMeta = Join-Path $KvDst $modelDir
        if (-not (Test-Path $modelDstMeta)) {
            New-Item -ItemType Directory -Path $modelDstMeta -Force | Out-Null
        }
        Copy-Item -Path $meta -Destination $modelDstMeta -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "      done" -ForegroundColor Green

# ---- Total size ----
$totalMB = (Get-ChildItem -Path $Backup -Recurse -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ("  Total critical data copied: {0:N1} MB" -f $totalMB) -ForegroundColor Green
Write-Host "  Path: $Backup" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Cyan

# =====================================================================
# OPTIONAL (large; recoverable by recomputation, but slow)
#
# Uncomment blocks below if you have desktop/external space.
# Replace $Backup with external drive path for large items.
# =====================================================================

# ---- Raw safetensors Qwen3-1.7B (~75 GB) ----
# Recompute = ~6-8 h CPU. Save if you have the room.
#
# Write-Host ""
# Write-Host "[optional] outputs\kv_capture\qwen3-1.7b raw safetensors (~75 GB)" -ForegroundColor Magenta
# robocopy (Join-Path $Source "outputs\kv_capture\qwen3-1.7b") `
#          (Join-Path $Backup "outputs\kv_capture\qwen3-1.7b") /E /XD analysis /NFL /NDL /NP /R:1 /W:1

# ---- Multi-seed (~32 GB) ----
# Write-Host ""
# Write-Host "[optional] outputs\kv_capture\qwen3-1.7b_multiseed (~32 GB)" -ForegroundColor Magenta
# robocopy (Join-Path $Source "outputs\kv_capture\qwen3-1.7b_multiseed") `
#          (Join-Path $Backup "outputs\kv_capture\qwen3-1.7b_multiseed") /E /XD analysis /NFL /NDL /NP /R:1 /W:1

# ---- Qwen3-4B + DeepSeek raw (~8 GB) ----
# Write-Host ""
# Write-Host "[optional] outputs\kv_capture\qwen3-4b + deepseek-r1-distill-qwen-1.5b (~8 GB)" -ForegroundColor Magenta
# robocopy (Join-Path $Source "outputs\kv_capture\qwen3-4b") `
#          (Join-Path $Backup "outputs\kv_capture\qwen3-4b") /E /XD analysis /NFL /NDL /NP /R:1 /W:1
# robocopy (Join-Path $Source "outputs\kv_capture\deepseek-r1-distill-qwen-1.5b") `
#          (Join-Path $Backup "outputs\kv_capture\deepseek-r1-distill-qwen-1.5b") /E /XD analysis /NFL /NDL /NP /R:1 /W:1

# ---- HuggingFace cache (~28 GB) ----
# All downloaded models + MATH-500 dataset.
# Recompute = re-download (size + internet).
#
# Write-Host ""
# Write-Host "[optional] HuggingFace cache (~28 GB)" -ForegroundColor Magenta
# robocopy "$env:USERPROFILE\.cache\huggingface" `
#          (Join-Path $Backup "huggingface_cache") /E /NFL /NDL /NP /R:1 /W:1

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "To also save raw safetensors or HF cache - uncomment the" -ForegroundColor Gray
Write-Host "corresponding blocks at the bottom of this script and rerun." -ForegroundColor Gray
Write-Host ""
