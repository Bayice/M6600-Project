$ErrorActionPreference = "Stop"

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "logs\semi_supervised_$Timestamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path "models\final" | Out-Null
New-Item -ItemType Directory -Force -Path "data\processed\semi_supervised" | Out-Null

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

# ============================================================
# Choose teacher models here.
# After Round2 training, adjust fold if another fold is better.
# ============================================================

$SixTeacher = "models/final/round2_exp2_l2_only_fold0_distilhubert_hfstyle/best_model"
$EightTeacher = "models/final/round2_exp5_l2_sg_british_fold0_distilhubert_hfstyle/best_model"

$Pool = "data/processed/external_metadata_plus_targeted.csv"

$SixLabels = "Arabic,Hindi,Korean,Mandarin,Spanish,Vietnamese"
$EightLabels = "Arabic,Hindi,Korean,Mandarin,Spanish,Vietnamese,OOD_British_Isles_English,OOD_Singaporean_English"

$Conf = "0.95"
$MaxPerLabel = "500"

# Exclude supervised data from pseudo-label pool.
$SixExclude = "data/processed/round2_balanced/fold0/train_l2_only.csv data/processed/round2_balanced/fold0/dev_l2_only.csv data/processed/round2_balanced/fold0/test_l2_only.csv"
$EightExclude = "data/processed/round2_balanced/fold0/train_round2_expanded_balanced.csv data/processed/round2_balanced/fold0/dev_round2_expanded_balanced.csv data/processed/round2_balanced/fold0/test_round2_expanded_combined.csv"

# ============================================================
# 1. Generate six-class pseudo labels.
# ============================================================

$Cmd = "$Activate && python -u -m src.semi_supervised.generate_pseudo_labels --teacher-model $SixTeacher --pool-metadata $Pool --output-dir data/processed/semi_supervised --name six_conf095_max500 --allowed-labels $SixLabels --exclude-csvs $SixExclude --confidence-threshold $Conf --max-per-label $MaxPerLabel --top-k 6 --seed 42"

Run-Step "Generate six-class pseudo labels" $Cmd

# ============================================================
# 2. Generate eight-class pseudo labels.
# ============================================================

$Cmd = "$Activate && python -u -m src.semi_supervised.generate_pseudo_labels --teacher-model $EightTeacher --pool-metadata $Pool --output-dir data/processed/semi_supervised --name eight_conf095_max500 --allowed-labels $EightLabels --exclude-csvs $EightExclude --confidence-threshold $Conf --max-per-label $MaxPerLabel --top-k 8 --seed 42"

Run-Step "Generate eight-class pseudo labels" $Cmd

# ============================================================
# 3. Build semi-supervised train CSVs.
# ============================================================

$Cmd = "$Activate && python -u -m src.semi_supervised.make_semi_supervised_train --base-train-csv data/processed/round2_balanced/fold0/train_l2_only.csv --pseudo-train-csv data/processed/semi_supervised/six_conf095_max500/pseudo_train_rows.csv --output-train-csv data/processed/semi_supervised/fold0_train_semi_six_conf095_max500.csv --allowed-labels $SixLabels --shuffle --seed 42"

Run-Step "Make six-class semi-supervised train CSV" $Cmd

$Cmd = "$Activate && python -u -m src.semi_supervised.make_semi_supervised_train --base-train-csv data/processed/round2_balanced/fold0/train_round2_expanded_balanced.csv --pseudo-train-csv data/processed/semi_supervised/eight_conf095_max500/pseudo_train_rows.csv --output-train-csv data/processed/semi_supervised/fold0_train_semi_eight_conf095_max500.csv --allowed-labels $EightLabels --shuffle --seed 42"

Run-Step "Make eight-class semi-supervised train CSV" $Cmd

# ============================================================
# 4. Train student models.
# HF-style config:
# lr=5e-5, epochs=10, warmup_ratio=0.1, effective batch size=8.
# ============================================================

$TrainArgs = "--model-name ntu-spml/distilhubert --label-mode union --epochs 10 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --learning-rate 5e-5 --warmup-ratio 0.1 --weight-decay 0.0 --max-seconds 8 --freeze-feature-encoder --logging-steps 1 --seed 42"

$Cmd = "$Activate && python -u -m src.training.train_distilhubert_flexible $TrainArgs --train-csv data/processed/semi_supervised/fold0_train_semi_six_conf095_max500.csv --dev-csv data/processed/round2_balanced/fold0/dev_l2_only.csv --test-csv data/processed/round2_balanced/fold0/test_l2_only.csv --output-dir models/final/semi_exp7_six_pseudo_conf095_max500_fold0_distilhubert_hfstyle"

Run-Step "Train six-class semi-supervised student" $Cmd

$Cmd = "$Activate && python -u -m src.training.train_distilhubert_flexible $TrainArgs --train-csv data/processed/semi_supervised/fold0_train_semi_eight_conf095_max500.csv --dev-csv data/processed/round2_balanced/fold0/dev_round2_expanded_balanced.csv --test-csv data/processed/round2_balanced/fold0/test_round2_expanded_combined.csv --output-dir models/final/semi_exp7_eight_pseudo_conf095_max500_fold0_distilhubert_hfstyle"

Run-Step "Train eight-class semi-supervised student" $Cmd

Write-Host ""
Write-Host "============================================================"
Write-Host "ALL SEMI-SUPERVISED PIPELINE FINISHED"
Write-Host "Logs saved to: $TranscriptPath"
Write-Host "Outputs:"
Write-Host "  data/processed/semi_supervised"
Write-Host "  models/final/semi_exp7_*"
Write-Host "============================================================"

Stop-Transcript | Out-Null