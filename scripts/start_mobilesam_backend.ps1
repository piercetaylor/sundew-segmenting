$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".tools\label-studio-venv\Scripts\python.exe"
$BackendDirectory = Join-Path $ProjectRoot ".tools\label-studio-ml-backend\label_studio_ml\examples\segment_anything_model"
$Checkpoint = Join-Path $ProjectRoot ".tools\MobileSAM\weights\mobile_sam.pt"
$CredentialsPath = Join-Path $ProjectRoot ".tools\label-studio-credentials.json"
$LogDirectory = Join-Path $ProjectRoot ".tools\logs"
$BackendDataDirectory = Join-Path $ProjectRoot ".tools\mobilesam-data"
$ImageDirectory = Join-Path $ProjectRoot "data\curated"

foreach ($RequiredPath in @($Python, $BackendDirectory, $Checkpoint, $CredentialsPath)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Missing required local setup path: $RequiredPath"
    }
}

$Credentials = Get-Content -LiteralPath $CredentialsPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $LogDirectory, $BackendDataDirectory | Out-Null

$env:LABEL_STUDIO_URL = "http://127.0.0.1:8080"
$env:LABEL_STUDIO_HOST = $env:LABEL_STUDIO_URL
$env:LABEL_STUDIO_API_KEY = $Credentials.token
$env:LABEL_STUDIO_ACCESS_TOKEN = $Credentials.token
$env:LABEL_STUDIO_BASE_DATA_DIR = $BackendDataDirectory
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $ImageDirectory
$env:SAM_CHOICE = "MobileSAM"
$env:MOBILESAM_CHECKPOINT = $Checkpoint

$Arguments = @(
    "_wsgi.py",
    "--host", "127.0.0.1",
    "--port", "9090",
    "--log-level", "WARNING"
)

Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $BackendDirectory `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDirectory "mobilesam.stdout.log") `
    -RedirectStandardError (Join-Path $LogDirectory "mobilesam.stderr.log")

Write-Output "MobileSAM is starting at http://127.0.0.1:9090"
