$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $ProjectRoot ".tools\label-studio-venv\Scripts\label-studio.exe"
$CredentialsPath = Join-Path $ProjectRoot ".tools\label-studio-credentials.json"
$DataDirectory = Join-Path $ProjectRoot "data\label-studio"
$ImageDirectory = Join-Path $ProjectRoot "data\curated"
$LogDirectory = Join-Path $ProjectRoot ".tools\logs"

if (-not (Test-Path -LiteralPath $Executable)) {
    throw "Label Studio is not installed. Run scripts/setup_label_studio.py first."
}
if (-not (Test-Path -LiteralPath $CredentialsPath)) {
    throw "Local credentials are missing. Run scripts/setup_label_studio.py first."
}

$Credentials = Get-Content -LiteralPath $CredentialsPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $DataDirectory, $LogDirectory | Out-Null

$env:LABEL_STUDIO_BASE_DATA_DIR = $DataDirectory
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $ImageDirectory
$env:LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK = "true"

$Arguments = @(
    "start",
    "--no-browser",
    "--internal-host", "127.0.0.1",
    "--host", "http://127.0.0.1:8080",
    "--port", "8080",
    "--data-dir", $DataDirectory,
    "--username", $Credentials.username,
    "--password", $Credentials.password,
    "--user-token", $Credentials.token,
    "--enable-legacy-api-token",
    "--log-level", "WARNING"
)

Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDirectory "label-studio.stdout.log") `
    -RedirectStandardError (Join-Path $LogDirectory "label-studio.stderr.log")

Write-Output "Label Studio is starting at http://127.0.0.1:8080"
