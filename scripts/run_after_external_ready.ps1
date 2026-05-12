# Run remaining external-augmented experiments after external data is already prepared.
# This script does NOT crawl external datasets again.
#
# It will:
#   1. check external_metadata.csv
#   2. generate augmented CSV files
#   3. check augmented data size / label distribution
#   4. train external-augmented models
#   5. save full logs

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\hexyw\Desktop\M6600_Project"
$CondaRoot = "C:\Users\hexyw\anaconda3"
$EnvName = "accent"

Set-Location $ProjectRoot

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "logs\after_external_ready_$Timestamp"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$MasterLog = "$LogDir\MASTER_after_external_ready.log"

function Write-Log {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $MasterLog -Value $line
}

function Run-Step {
    param(
        [string]$Name,
        [string]$Command,
        [string]$LogFile
    )

    Write-Log "============================================================"
    Write-Log "START: $Name"
    Write-Log "COMMAND: $Command"
    Write-Log "LOG: $LogFile"
    Write-Log "============================================================"

    Add-Content -Path $LogFile -Value "============================================================"
    Add-Content -Path $LogFile -Value "START: $Name"
    Add-Content -Path $LogFile -Value "TIME: $(Get-Date)"
    Add-Content -Path $LogFile -Value "COMMAND: $Command"
    Add-Content -Path $LogFile -Value "============================================================"

    # Important:
    # Some Python / Transformers / CUDA messages are written to stderr
    # even when training is normal. We should log stderr, not treat it as
    # a PowerShell fatal error.
    $OldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    cmd /c "$Command" 2>&1 | ForEach-Object {
        $_
        Add-Content -Path $LogFile -Value $_
    }

    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $OldErrorActionPreference

    if ($ExitCode -ne 0) {
        Write-Log "FAILED: $Name"
        Write-Log "Exit code: $ExitCode"
        throw "Step failed: $Name"
    }
 
    Add-Content -Path $LogFile -Value "============================================================"
    Add-Content -Path $LogFile -Value "END: $Name"
    Add-Content -Path $LogFile -Value "TIME: $(Get-Date)"
    Add-Content -Path $LogFile -Value "============================================================"

    Write-Log "DONE: $Name"
}

Write-Log "After-external-ready experiment started"
Write-Log "Project root: $ProjectRoot"
Write-Log "Log dir: $LogDir"

# Prevent sleep / hibernate while training.
powercfg /change monitor-timeout-ac 10
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

$Activate = "call `"$CondaRoot\Scripts\activate.bat`" $EnvName"

# ------------------------------------------------------------
# 00. Check environment
# ------------------------------------------------------------
Run-Step `
    -Name "00 Check CUDA and imports" `
    -Command "$Activate && python -u -c `"import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')`"" `
    -LogFile "$LogDir\00_check_env.log"

# ------------------------------------------------------------
# 01. Check external metadata
# ------------------------------------------------------------
Run-Step `
    -Name "01 Check external metadata" `
    -Command "$Activate && python -u -c `"import pandas as pd; df=pd.read_csv('data/processed/external_metadata.csv'); print('external rows:', len(df)); print('\nDataset counts:'); print(df['dataset'].value_counts()); print('\nLabel counts:'); print(df['label'].value_counts()); mapped=df[df['is_mapped_to_baseline'].astype(str).str.lower().isin(['true','1'])]; print('\nMapped rows:', len(mapped)); print(mapped['label'].value_counts())`"" `
    -LogFile "$LogDir\01_check_external_metadata.log"

# ------------------------------------------------------------
# 02. Make augmented split CSVs
# ------------------------------------------------------------
Run-Step `
    -Name "02 Make external augmented splits" `
    -Command "$Activate && python -u -m src.data.make_external_augmented_splits" `
    -LogFile "$LogDir\02_make_external_augmented_splits.log"

# ------------------------------------------------------------
# 03. Check augmented CSVs
# ------------------------------------------------------------
Run-Step `
    -Name "03 Check augmented CSVs" `
    -Command "$Activate && python -u -c `"import pandas as pd; files=['data/processed/augmented/external_six_train.csv','data/processed/augmented/external_expanded_train.csv','data/processed/augmented/fold0/train_six_l2_plus_external.csv','data/processed/augmented/fold0/train_six_external_only.csv','data/processed/augmented/fold0/train_expanded_l2_plus_external.csv','data/processed/augmented/fold0/train_expanded_external_only.csv']; [print('\n'+f, pd.read_csv(f).shape, '\nnum labels=', pd.read_csv(f)['label'].nunique(), '\n', pd.read_csv(f)['label'].value_counts().head(30)) for f in files]`"" `
    -LogFile "$LogDir\03_check_augmented_csvs.log"

# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------
$CommonTrainArgs = "--epochs 3 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --max-seconds 8 --freeze-feature-encoder --logging-steps 10"

# ------------------------------------------------------------
# 04-19. Train 16 augmented models
# ------------------------------------------------------------
for ($Fold = 0; $Fold -lt 4; $Fold++) {
    $FoldDir = "data/processed/augmented/fold$Fold"
    $DevCsv = "$FoldDir/dev.csv"
    $TestCsv = "$FoldDir/test.csv"

    Run-Step `
        -Name "fold$Fold A1 six-label L2 plus external" `
        -Command "$Activate && python -u -m src.training.train_distilhubert_flexible --label-mode baseline --train-csv $FoldDir/train_six_l2_plus_external.csv --dev-csv $DevCsv --test-csv $TestCsv --output-dir models/checkpoints/exp3_fold${Fold}_six_l2_plus_external_distilhubert $CommonTrainArgs" `
        -LogFile "$LogDir\04_fold${Fold}_six_l2_plus_external.log"

    Run-Step `
        -Name "fold$Fold A2 six-label external only" `
        -Command "$Activate && python -u -m src.training.train_distilhubert_flexible --label-mode baseline --train-csv $FoldDir/train_six_external_only.csv --dev-csv $DevCsv --test-csv $TestCsv --output-dir models/checkpoints/exp4_fold${Fold}_six_external_only_distilhubert $CommonTrainArgs" `
        -LogFile "$LogDir\05_fold${Fold}_six_external_only.log"

    Run-Step `
        -Name "fold$Fold B1 expanded-label L2 plus external" `
        -Command "$Activate && python -u -m src.training.train_distilhubert_flexible --label-mode union --train-csv $FoldDir/train_expanded_l2_plus_external.csv --dev-csv $DevCsv --test-csv $TestCsv --output-dir models/checkpoints/exp5_fold${Fold}_expanded_l2_plus_external_distilhubert $CommonTrainArgs" `
        -LogFile "$LogDir\06_fold${Fold}_expanded_l2_plus_external.log"

    Run-Step `
        -Name "fold$Fold B2 expanded-label external only" `
        -Command "$Activate && python -u -m src.training.train_distilhubert_flexible --label-mode union --train-csv $FoldDir/train_expanded_external_only.csv --dev-csv $DevCsv --test-csv $TestCsv --output-dir models/checkpoints/exp6_fold${Fold}_expanded_external_only_distilhubert $CommonTrainArgs" `
        -LogFile "$LogDir\07_fold${Fold}_expanded_external_only.log"
}

# ------------------------------------------------------------
# 20. Collect summaries
# ------------------------------------------------------------
Run-Step `
    -Name "20 Collect summaries" `
    -Command "$Activate && python -u -c `"from pathlib import Path; paths=sorted(Path('models/checkpoints').glob('exp*_distilhubert/test_summary.json')); print('Found summaries:', len(paths)); [print('\n'+str(p)+'\n'+p.read_text(encoding='utf-8')) for p in paths]`"" `
    -LogFile "$LogDir\20_collect_summaries.log"

Write-Log "============================================================"
Write-Log "ALL AFTER-EXTERNAL-READY EXPERIMENTS FINISHED"
Write-Log "Logs saved to: $LogDir"
Write-Log "============================================================"