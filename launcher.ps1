#!/usr/bin/env pwsh
<#
.SYNOPSIS
Isaac RL Studio launcher for Windows PowerShell.
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("build","up","logs","clean","kill")]
    [string]$Command,
    
    [string]$Head = "humanoid",
    [string]$EnvFile,
    [switch]$NoGui,
    [switch]$Headless,
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
  build                        Build Docker simulation image
  up                           Start headless simulation container
  logs [service]               Show logs (default: isaac-sim)
  clean                        Remove containers, images, volumes
  kill                         Stop and remove containers

Options for 'up':
  -Head <name>                 Head name (humanoid | anymal | amr | cobot)
  -EnvFile <path>              Path to environment config file
  -Headless                    Run in headless container mode (Camera & ROS2 active)
  -Task <task>                 IsaacLab task override

Examples:
  .\launcher.ps1 build
  .\launcher.ps1 up -Head humanoid -Headless
  .\launcher.ps1 up -Head anymal -Headless
  .\launcher.ps1 up -Head amr -Headless
  .\launcher.ps1 up -Head cobot -Headless
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
    Write-Info "Building Isaac RL Studio Docker image..."
    Push-Location (Join-Path $ScriptDir "docker")
    try {
        docker compose -f $ComposeFile build isaac-sim
        if ($LASTEXITCODE -ne 0) { throw "docker compose build isaac-sim failed" }
    }
    finally {
        Pop-Location
    }
    Write-Info "Build complete."
}

function Invoke-Up {
    if (-not $Head) {
        $Head = "humanoid"
        Write-Warn "No -Head specified, using default: $Head"
    }

    $headPath = Join-Path $ScriptDir "core\source\$Head"
    if (-not (Test-Path $headPath)) {
        Write-Err "Head not found: $Head"
        Write-Host "Available heads: humanoid, anymal, amr, cobot"
        exit 1
    }

    $env:HEAD_NAME = $Head
    $env:NO_GUI = "1"

    Write-Info "Starting headless simulation with head: $Head"
    Write-Info "In-scene Camera Active -> Saving MP4 to /workspace/data/videos & streaming live to ROS2 /camera/rgb/image_raw"

    Push-Location (Join-Path $ScriptDir "docker")
    try {
        docker compose -f $ComposeFile up -d isaac-sim
    }
    finally {
        Pop-Location
    }

    $containerId = docker compose -f $ComposeFile ps -q isaac-sim
    if ($containerId) {
        Write-Info "Container started: $containerId"
        Write-Host "Enter container: docker exec -it $containerId bash"
        Write-Host "View ROS2 stream: python3 core/ros2_ws/image_listener.py"
        Write-Host "View logs:       .\launcher.ps1 logs"
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
