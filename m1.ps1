# ── 系统 PATH ──────────────────────────────────────────────
Write-Host "`n===== 系统 PATH (Machine) =====" -ForegroundColor Cyan
[System.Environment]::GetEnvironmentVariable("Path", "Machine") -split ";" |
    Where-Object { $_ -ne "" } | ForEach-Object { Write-Host "  $_" }

# ── 用户 PATH ──────────────────────────────────────────────
Write-Host "`n===== 用户 PATH (User) =====" -ForegroundColor Green
[System.Environment]::GetEnvironmentVariable("Path", "User") -split ";" |
    Where-Object { $_ -ne "" } | ForEach-Object { Write-Host "  $_" }

# ── CUDA / cuDNN 相关环境变量 ──────────────────────────────
Write-Host "`n===== CUDA / cuDNN 环境变量 =====" -ForegroundColor Yellow

$sources = @(
    @{ Label = "Machine"; Level = "Machine" },
    @{ Label = "User";    Level = "User"    }
)

foreach ($src in $sources) {
    $vars = [System.Environment]::GetEnvironmentVariables($src.Level).GetEnumerator() |
        Where-Object { $_.Key -match "CUDA|CUDNN|NVCUDA|NVTOOLSEXT|CUPTI" } |
        Sort-Object Key
    if ($vars) {
        Write-Host "`n  [$($src.Label)]" -ForegroundColor Magenta
        foreach ($v in $vars) {
            Write-Host "  $($v.Key) = $($v.Value)"
        }
    }
}

# ── PATH 中含 cuda/cudnn 的条目单独列出 ───────────────────
Write-Host "`n===== PATH 中含 CUDA/cuDNN 的条目 =====" -ForegroundColor Yellow
$env:PATH -split ";" |
    Where-Object { $_ -match "cuda|cudnn" -and $_ -ne "" } |
    ForEach-Object { Write-Host "  $_" }

Write-Host "`n完成。" -ForegroundColor Gray
