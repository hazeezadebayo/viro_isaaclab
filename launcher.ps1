#!/usr/bin/env pwsh
<#
.SYNOPSIS
Isaac RL Studio launcher for Windows PowerShell.
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("build","up","logs","clean","kill")]
    [string]$Command,
    
    [string]$Head = "template",
    [string]$EnvFile,
    [switch]$NoGui,
    [switch]$Headless,
    [switch]$Viz,
    [string]$Task
)

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ComposeFile = Join-Path $ScriptDir "docker\docker-compose.yml"

$HeadTaskMap = @{
    "humanoid" = "Isaac-Humanoid-Imitation-v0"
    "anymal"   = "Isaac-Anymal-C-v0"
    "amr"      = "Isaac-AMR-Navigation-v0"
    "cobot"    = "Isaac-Cobot-Reaching-v0"
}


function Write-Info($msg) { Write-Host $msg -ForegroundColor Green }
function Write-Warn($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host $msg -ForegroundColor Red }

function Show-Usage {
    Write-Host @"
Usage: .\launcher.ps1 <command> [options]

Commands:
  build                        Build Docker images (main + viz)
  up                          Start simulation container
  logs [service]              Show logs (default: isaac-sim)
  clean                       Remove containers, images, volumes
  kill                        Stop and remove containers

Options for 'up':
  -Head <name>                Head/environment name (default: template)
  -EnvFile <path>             Path to environment config file
  -NoGui                      Run headless (no Isaac Sim GUI)
  -Headless                   Alias for -NoGui
  -Viz                        Run with browser visualization (noVNC) instead
                              of the plain sim container. Open:
                              http://localhost:6080/vnc.html
  -Task <task>                IsaacLab task to auto-start inside the -Viz
                              container (e.g. Isaac-Humanoid-v0). Falls back
                              to the head's default task when not given.

Examples:
  .\launcher.ps1 build
  .\launcher.ps1 up -Head anymal
  .\launcher.ps1 up -Head spider -NoGui
  .\launcher.ps1 up -Head spider -Headless
  .\launcher.ps1 up -Head humanoid -Viz
  .\launcher.ps1 up -Head humanoid -Viz -Task Isaac-Humanoid-v0
  .\launcher.ps1 up -EnvFile .env.custom
  .\launcher.ps1 logs
  .\launcher.ps1 kill
"@
    exit 1
}

if (-not $Command) { Show-Usage }

# Load env file if specified
if ($EnvFile) {
    if (-not (Test-Path $EnvFile)) {
        Write-Err "Env file not found: $EnvFile"
        exit 1
    }
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

function Invoke-Build {
    Write-Info "Building Isaac RL Studio images (main + viz)..."
    Push-Location (Join-Path $ScriptDir "docker")
    try {
        docker compose -f $ComposeFile build isaac-sim
        if ($LASTEXITCODE -ne 0) { throw "docker compose build isaac-sim failed" }
        docker compose -f $ComposeFile build isaac-viz
        if ($LASTEXITCODE -ne 0) { throw "docker compose build isaac-viz failed" }
    }
    finally {
        Pop-Location
    }
    Write-Info "Build complete."
}

function Invoke-Up {
    if (-not $Head) {
        $Head = "template"
        Write-Warn "No -Head specified, using default: $Head"
    }

    $headPath = Join-Path $ScriptDir "core\source\$Head"
    if (-not (Test-Path $headPath)) {
        Write-Err "Head not found: $Head"
        Write-Host "Available heads:"
        Get-ChildItem (Join-Path $ScriptDir "heads") -Directory | ForEach-Object { Write-Host "  $($_.Name)" }
        exit 1
    }

    $env:HEAD_NAME = $Head
    $noGuiEffective = $NoGui -or $Headless
    if ($noGuiEffective) { $env:NO_GUI = "1" }
    else { Remove-Item Env:NO_GUI -ErrorAction SilentlyContinue }

    if ($Viz) {
        # Ensure the viz container always runs in GUI mode and does not inherit headless settings.
        $env:NO_GUI = "0"
        $env:VIZ_MODE = "1"

        $vizPort = if ($env:VIZ_PORT) { $env:VIZ_PORT } else { "6080" }
        $taskEffective = if ($Task) { $Task } elseif ($HeadTaskMap[$Head]) { $HeadTaskMap[$Head] } else { "" }
        if ($taskEffective) {
            $env:VIZ_TASK = $taskEffective
            Write-Info "Auto-starting task: $taskEffective (headless training)"
        }
        else {
            Remove-Item Env:VIZ_TASK -ErrorAction SilentlyContinue
            Write-Warn "No task mapped for head '$Head' - visualization only (no robot)"
        }
        Write-Info "Starting visualization container with head: $Head"
        Write-Warn "Open http://localhost:$vizPort/vnc.html in your browser (virtual desktop only)"
        Write-Warn "Training metrics: http://localhost:6006 (TensorBoard, auto-started)"
        Push-Location (Join-Path $ScriptDir "docker")
        try {
            docker compose -f $ComposeFile stop isaac-sim 2>&1 | Out-Null
            docker compose -f $ComposeFile up -d isaac-viz
        }
        finally {
            Pop-Location
        }

        $containerId = docker compose -f $ComposeFile ps -q isaac-viz
        if ($containerId) {
            Write-Info "Viz container started: $containerId"
            Write-Host "Enter: docker exec -it $containerId bash"
            Write-Host "Then:  source /usr/local/bin/isaac-ros-entrypoint.sh"
            Write-Host "Logs:  .\launcher.ps1 logs isaac-viz"
        }
        return
    }

    Write-Info "Starting with head: $Head"
    if ($noGuiEffective) { Write-Warn "Running headless mode" }

    Push-Location (Join-Path $ScriptDir "docker")
    try {
        docker compose -f $ComposeFile stop isaac-viz 2>&1 | Out-Null
        docker compose -f $ComposeFile up -d isaac-sim
    }
    finally {
        Pop-Location
    }

    $containerId = docker compose -f $ComposeFile ps -q isaac-sim
    if ($containerId) {
        Write-Info "Container started: $containerId"
        Write-Host "Enter: docker exec -it $containerId bash"
        Write-Host "Logs:  .\launcher.ps1 logs"
    }
}

function Invoke-Logs {
    param([string]$Service = "isaac-sim")
    docker compose -f $ComposeFile logs -f $Service
}

function Invoke-Clean {
    Write-Warn "Cleaning up containers, images, and volumes..."
    Push-Location (Join-Path $ScriptDir "docker")
    try {
        docker compose -f $ComposeFile down -v --rmi local --remove-orphans 2>&1 | Out-Null
    }
    finally {
        Pop-Location
    }
    Write-Info "Clean complete."
}

function Invoke-Kill {
    Write-Warn "Stopping containers..."
    Push-Location (Join-Path $ScriptDir "docker")
    try {
        docker compose -f $ComposeFile down --timeout 5
    }
    finally {
        Pop-Location
    }
    Write-Info "Containers stopped."
}

switch ($Command) {
    "build" { Invoke-Build }
    "up" { Invoke-Up }
    "logs" { Invoke-Logs -Service (if ($args[0]) { $args[0] } else { "isaac-sim" }) }
    "clean" { Invoke-Clean }
    "kill" { Invoke-Kill }
    default { Show-Usage }
}
