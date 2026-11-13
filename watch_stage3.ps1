$Host.UI.RawUI.WindowTitle = "Stage 3 - Cross-fit training (live)"
Set-Location d:\Samruddhi\driving-label-auditor
while ($true) {
  Clear-Host
  Write-Host "=== STAGE 3: training YOLO on NOISY labels (2-fold cross-fitting) ===" -ForegroundColor Cyan
  Write-Host (Get-Date -Format "HH:mm:ss") " (refreshes every 10 s; close window any time - training keeps running)`n"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
  foreach ($k in 0,1) {
    $f = "work\runs\crossfit_model$k\results.csv"
    Write-Host "`n--- model $k (train on half $k, predict the other half) ---" -ForegroundColor Yellow
    if (Test-Path $f) {
      $d = Import-Csv $f
      Write-Host ("epochs done: {0}/30" -f $d.Count)
      $d | Select-Object -Last 8 | ForEach-Object { "epoch {0,2}  train box/cls loss {1:N3}/{2:N3}   val mAP50 {3:N3}   mAP50-95 {4:N3}" -f [int]$_.epoch, [double]$_.'train/box_loss', [double]$_.'train/cls_loss', [double]$_.'metrics/mAP50(B)', [double]$_.'metrics/mAP50-95(B)' }
    } else { Write-Host "not started yet" }
  }
  Start-Sleep 10
}
