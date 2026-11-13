$Host.UI.RawUI.WindowTitle = "Stage 6 - 3 detectors: clean vs noisy vs cleaned (live)"
Set-Location d:\Samruddhi\driving-label-auditor
while ($true) {
  Clear-Host
  Write-Host "=== STAGE 6: clean vs noisy vs cleaned labels (same settings/seed) ===" -ForegroundColor Cyan
  Write-Host ((Get-Date -Format "HH:mm:ss") + "  refreshes every 10 s; closing this window does not stop training`n")
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
  foreach ($n in "a_clean","b_noisy","c_cleaned") {
    $f = "work\runs\downstream_$n\results.csv"
    Write-Host "`n--- detector $n ---" -ForegroundColor Yellow
    if (Test-Path $f) { $d = Import-Csv $f; Write-Host ("epochs done: {0}/30" -f $d.Count)
      $d | Select-Object -Last 5 | ForEach-Object { "epoch {0,2}  train box loss {1:N3}   val mAP50 {2:N3}   mAP50-95 {3:N3}" -f [int]$_.epoch,[double]$_.'train/box_loss',[double]$_.'metrics/mAP50(B)',[double]$_.'metrics/mAP50-95(B)' } }
    else { Write-Host "waiting" }
  }
  if (Test-Path results\downstream_results.json) { Write-Host "`nFINAL so far:" -ForegroundColor Green; Get-Content results\downstream_results.json | Select-String "mAP50"}
  Start-Sleep 10
}
