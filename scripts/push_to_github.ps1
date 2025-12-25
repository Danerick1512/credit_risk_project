<#
Usage: .\scripts\push_to_github.ps1 -RepoUrl "https://github.com/usuario/repo.git" -Message "Mi commit"

This script will:
 - initialize a git repo if needed
 - create a .gitignore if missing (does not overwrite)
 - add and commit changes
 - add remote origin if RepoUrl provided and not set
 - set branch to main and push to origin

Security: It's safer to create the GitHub repo first and then run this script with the repo URL.
#>
param(
    [string]$RepoUrl = '',
    [string]$Message = 'Commit desde push_to_github.ps1',
    [switch]$Force
)

Set-Location -Path (Split-Path -Path $MyInvocation.MyCommand.Definition -Parent)\..\
$cwd = Get-Location
Write-Host "Trabajo en: $cwd"

function RunGit([string]$args){
    Write-Host "git $args"
    $p = Start-Process -FilePath git -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput stdout.txt -RedirectStandardError stderr.txt
n    $out = Get-Content stdout.txt -Raw -ErrorAction SilentlyContinue
    $err = Get-Content stderr.txt -Raw -ErrorAction SilentlyContinue
    if ($out) { Write-Host $out }
    if ($err) { Write-Host $err }
}

# 1) Init repo if needed
if (-not (Test-Path .git)) {
    Write-Host "No encuentro .git — inicializando repo"
    git init
} else {
    Write-Host ".git ya existe"
}

# 2) Ensure .gitignore exists (do not overwrite)
$gitignore = Join-Path $cwd '.gitignore'
if (-not (Test-Path $gitignore)) {
    Write-Host "No existe .gitignore — creando uno básico"
    @"# Python
__pycache__/
.venv/
.env
output/models/
"@ | Out-File -FilePath $gitignore -Encoding utf8
} else {
    Write-Host ".gitignore ya existe"
}

# 3) Stage and commit
git add .
try {
    git commit -m "$Message"
} catch {
    Write-Host "No se pudo commitear (¿ya había cambios sin staged?). Intentando forzar staging y commit."
    git add .
    git commit -m "$Message"
}

# 4) Add remote if provided and not present
if ($RepoUrl -ne '') {
    $remotes = git remote
    if ($remotes -notmatch 'origin') {
        git remote add origin $RepoUrl
        Write-Host "Remote 'origin' agregado: $RepoUrl"
    } else {
        Write-Host "Remote 'origin' ya existe"
    }
}

# 5) Set branch main and push
git branch -M main
if ($RepoUrl -ne '') {
    Write-Host "Pushing a origin main..."
    git push -u origin main
} else {
    Write-Host "No se proporcionó RepoUrl — repository inicializado localmente. Agrega un remote y push manualmente."
}

Write-Host "Hecho."