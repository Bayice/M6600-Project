$ErrorActionPreference = "Stop"

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "logs\round2_final_visible_$Timestamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path "models\final" | Out-Null

$TranscriptPath = "$LogDir\FULL_CONSOLE_TRANSCRIPT.log"
Start-Transcript -Path $TranscriptPath -Append | Out-Null

function Run-Step {
    param (
        [string]$Name,
        [string]$Command
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "START: $Name"
    Write-Host "TIME:  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "COMMAND:"
    Write-Host $Command
    Write-Host "============================================================"
    Write-Host ""

    cmd /c "$Command"

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "FAILED: $Name"
        Write-Host "Exit code: $LASTEXITCODE"
        Stop-Transcript | Out-Null
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "DONE: $Name"
    Write-Host "TIME: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "============================================================"
}

$Activate = 'call "C:\Users\hexyw\anaconda3\Scripts\activate.bat" accent'

# HF-style config:
# learning_rate=5e-5
# epochs=10
# effective batch size=8 via batch_size=4 and gradient_accumulation_steps=2
# warmup_ratio=0.1
# AdamW + linear scheduler are handled by Hugging Face Trainer defaults / args.
$CommonArgs = "--model-name ntu-spml/distilhubert --label-mode union --epochs 10 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --learning-rate 5e-5 --warmup-ratio 0.1 --weight-decay 0.0 --max-seconds 8 --freeze-feature-encoder --logging-steps 1 --seed 42"

for ($Fold = 0; $Fold -le 3; $Fold++) {
    $TrainCsv = "data/processed/round2_balanced/fold$Fold/train_l2_only.csv"
    $DevCsv = "data/processed/round2_balanced/fold$Fold/dev_l2_only.csv"
    $TestCsv = "data/processed/round2_balanced/fold$Fold/test_l2_only.csv"
    $OutDir = "models/final/round2_exp2_l2_only_fold$Fold" + "_distilhubert_hfstyle"

    $Cmd = "$Activate && python -u -m src.training.train_distilhubert_flexible $CommonArgs --train-csv $TrainCsv --dev-csv $DevCsv --test-csv $TestCsv --output-dir $OutDir"

    Run-Step "Round2 Exp2 L2-only fold$Fold" $Cmd
}

for ($Fold = 0; $Fold -le 3; $Fold++) {
    $TrainCsv = "data/processed/round2_balanced/fold$Fold/train_round2_expanded_balanced.csv"
    $DevCsv = "data/processed/round2_balanced/fold$Fold/dev_round2_expanded_balanced.csv"
    $TestCsv = "data/processed/round2_balanced/fold$Fold/test_round2_expanded_combined.csv"
    $OutDir = "models/final/round2_exp5_l2_sg_british_fold$Fold" + "_distilhubert_hfstyle"

    $Cmd = "$Activate && python -u -m src.training.train_distilhubert_flexible $CommonArgs --train-csv $TrainCsv --dev-csv $DevCsv --test-csv $TestCsv --output-dir $OutDir"

    Run-Step "Round2 Exp5 L2+SG+British fold$Fold" $Cmd
}

Write-Host ""
Write-Host "============================================================"
Write-Host "ALL ROUND2 FINAL TRAINING FINISHED"
Write-Host "Logs saved to: $TranscriptPath"
Write-Host "Models saved under: models/final"
Write-Host "============================================================"

Stop-Transcript | Out-Null