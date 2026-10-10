# Runs a command (default: an interactive shell) in the SAVA dev container,
# with this repository mounted at /workspace. Builds the image on first use.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Command)

$Repo = Split-Path -Parent $PSScriptRoot
$Image = if ($env:SAVA_DEV_IMAGE) { $env:SAVA_DEV_IMAGE } else { "sava-dev" }

docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
    docker build -t $Image -f "$Repo/docker/Dockerfile" $Repo
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if (-not $Command) { $Command = @("bash") }
docker run --rm -it -v "${Repo}:/workspace" -w /workspace $Image @Command
exit $LASTEXITCODE
