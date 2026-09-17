param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ToolsDirectory = Join-Path $ProjectRoot ".tools"
$EnvironmentDirectory = Join-Path $ToolsDirectory "label-studio-venv"
$EnvironmentPython = Join-Path $EnvironmentDirectory "Scripts\python.exe"
$BackendDirectory = Join-Path $ToolsDirectory "label-studio-ml-backend"
$MobileSamDirectory = Join-Path $ToolsDirectory "MobileSAM"

New-Item -ItemType Directory -Force -Path $ToolsDirectory | Out-Null
if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    & $Python -m venv $EnvironmentDirectory
}

& $EnvironmentPython -m pip install --upgrade pip
& $EnvironmentPython -m pip install "label-studio==1.23.0"
& $EnvironmentPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

function Install-PinnedRepository {
    param(
        [string]$Url,
        [string]$Path,
        [string]$Commit
    )
    if (-not (Test-Path -LiteralPath (Join-Path $Path ".git"))) {
        git clone --filter=blob:none $Url $Path
    }
    git -C $Path fetch --depth 1 origin $Commit
    git -C $Path checkout --detach $Commit
}

Install-PinnedRepository `
    -Url "https://github.com/HumanSignal/label-studio-ml-backend.git" `
    -Path $BackendDirectory `
    -Commit "7b0d01f3593e327ad9cd2a92dbd08aa89f6403f4"
Install-PinnedRepository `
    -Url "https://github.com/ChaoningZhang/MobileSAM.git" `
    -Path $MobileSamDirectory `
    -Commit "f706ad9c4eb7f219c00d9050e46328518ffb65d2"

& $EnvironmentPython -m pip install -e $MobileSamDirectory
& $EnvironmentPython -m pip install -e $BackendDirectory
# Label Studio 1.23 requires a newer ijson than the converter's declared range;
# both imports and the end-to-end annotation smoke test are verified with 3.5.1.
& $EnvironmentPython -m pip install "ijson==3.5.1"
& $EnvironmentPython (Join-Path $PSScriptRoot "setup_label_studio.py")

Write-Output "Annotation environment installed in $EnvironmentDirectory"
