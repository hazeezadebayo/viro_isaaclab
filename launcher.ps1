#!/usr/bin/env pwsh
<#
.SYNOPSIS
Isaac RL Studio launcher for Windows PowerShell.
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("build","up","logs","clean","kill","train","play","export")]
    [string]$Command,
    
    [string]$Head = "humanoid",
    [string]$EnvFile,
    [switch]$NoGui,
    [switch]$Headless,
    [string]$Task,
    [int]$NumEnvs = 16,
    [int]$MaxIterations = 0,
    [string]$Checkpoint,
    [bool]$Video = $true,
    [double]$VideoLengthMin = 1.0,
    [double]$VideoIntervalMin = 30.0,
    [switch]$UsdExport,
    [double]$UsdIntervalSec = 1800.0,
    [double]$UsdLengthSec = 10.0,
    [switch]$RealTime,
    [double]$ExportSeconds = 5.0,
    [string]$ExportFormat = "usda"
)

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ComposeFile = Join-Path $ScriptDir "docker\docker-compose.yml"

$HeadTaskMap = @{
    "humanoid" = "Isaac-Humanoid-Imitation-v0"
    "anymal"   = "Isaac-Anymal-C-v0"
    "amr"      = "Isaac-AMR-Navigation-v0"
    "cobot"    = "Isaac-Lift-Cylinder-Cobot-Play-v0" # "Isaac-Lift-Cylinder-Cobot-Play-v0" | "Isaac-Lift-Cylinder-Cobot-v0"
}

# Simulation step time (sim.dt * decimation) per head, used to convert video clip
# lengths/intervals from seconds to simulation steps for the --video flags.
$SimDtMap = @{
    "humanoid" = (1.0 / 60.0)
    "anymal"   = (1.0 / 50.0)
    "amr"      = (1.0 / 25.0)
    "cobot"    = (1.0 / 50.0)
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
  train                        Run RL training in-container (built-in --video recording)
  play                         Run trained-policy play in-container (built-in --video recording)
  export                       Bake a trained-policy rollout into an animated USD file
  logs [service]               Show logs (default: isaac-sim)
  clean                        Remove containers, images, volumes
  kill                         Stop and remove containers

Options for 'up':
  -Head <name>                 Head name (humanoid | anymal | amr | cobot)
  -EnvFile <path>              Path to environment config file
  -Headless                    Run in headless container mode (Camera & ROS2 active)
  -Task <task>                 IsaacLab task override

Options for 'train' / 'play':
  -Head <name>                 Head name (humanoid | anymal | amr | cobot)
  -Task <task>                 IsaacLab task override (default: task mapped to -Head)
  -NumEnvs <int>               Number of parallel environments (default: 16)
  -MaxIterations <int>         Training iterations (train; default: runner cfg value)
  -Checkpoint <path>           Checkpoint relative to /workspace (play)
  -Video <bool>                Enable built-in --video recording (default: true)
  -VideoLengthMin <minutes>    Clip length (default: 1 minute)
  -VideoIntervalMin <minutes>  Steps between clip starts (default: 30 minutes)
  -RealTime                    Run play in real time

Options for 'export':
  -Head <name>                 Head name (humanoid | anymal | amr | cobot)
  -Task <task>                 IsaacLab task override (default: task mapped to -Head)
  -Checkpoint <path>           Checkpoint relative to /workspace (required)
  -ExportSeconds <seconds>     Rollout duration to export (default: 5)
  -ExportFormat <fmt>          USD format: usda | usdc | usd (default: usda)

Examples:
  .\launcher.ps1 build
  .\launcher.ps1 up -Head humanoid -Headless
  .\launcher.ps1 up -Head anymal -Headless
  .\launcher.ps1 up -Head amr -Headless
  .\launcher.ps1 up -Head cobot -Headless
  .\launcher.ps1 train -Head humanoid
  .\launcher.ps1 train -Head humanoid -VideoLengthMin 2 -VideoIntervalMin 15
  .\launcher.ps1 play -Head humanoid -Checkpoint ./logs/rsl_rl/humanoid/<run>/model_1000.pt
  .\launcher.ps1 export -Head humanoid -Checkpoint ./logs/rsl_rl/humanoid/<run>/model_1000.pt -ExportSeconds 5
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
    Write-Info "In-scene camera available via built-in '--video' recording (see launcher.ps1 train/play)"

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
        Write-Host "Train (with video): .\launcher.ps1 train -Head humanoid"
        Write-Host "View logs:          .\launcher.ps1 logs"
    }
}

function Get-StepCount {
    param([double]$Minutes)
    $headKey = if ($Head) { $Head.ToLower() } else { "humanoid" }
    $dt = $SimDtMap[$headKey]
    if (-not $dt) { $dt = 1.0 / 60.0 }
    return [int][Math]::Round($Minutes * 60.0 / $dt)
}

function Resolve-Task {
    $headKey = if ($Head) { $Head.ToLower() } else { "humanoid" }
    if ($Task) { return $Task }
    if (-not $HeadTaskMap.ContainsKey($headKey)) {
        Write-Err "No default task for head: $Head"
        exit 1
    }
    return $HeadTaskMap[$headKey]
}

function Get-VideoArgs {
    $videoArgs = @()
    if ($Video) {
        $videoArgs += @(
            "--video",
            "--video_length", "$(Get-StepCount $VideoLengthMin)",
            "--video_interval", "$(Get-StepCount $VideoIntervalMin)",
            "--enable_cameras"
        )
    }
    return $videoArgs
}

function Invoke-Train {
    $task = Resolve-Task
    $python = "/isaac-sim/python.sh"
    $script = "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py"

    $cmdArgs = @("--task", $task, "--headless", "--num_envs", "$NumEnvs")
    if ($MaxIterations -gt 0) { $cmdArgs += @("--max_iterations", "$MaxIterations") }
    if ($Checkpoint) { $cmdArgs += @("--checkpoint", $Checkpoint) }

    $execEnv = @()
    if ($UsdExport) {
        $intervalSec = if ($UsdIntervalSec -ne 1800.0) { $UsdIntervalSec } else { [int][Math]::Round($VideoIntervalMin * 60.0) }
        $lengthSec = if ($UsdLengthSec -ne 10.0) { $UsdLengthSec } else { [int][Math]::Round($VideoLengthMin * 60.0) }
        $execEnv = @("-e", "USD_EXPORT=1", "-e", "USD_INTERVAL=$intervalSec", "-e", "USD_LENGTH=$lengthSec")
        Write-Info "Automated USD & MP4 Export Enabled: Interval=${intervalSec}s (${VideoIntervalMin}min), Length=${lengthSec}s (${VideoLengthMin}min)"
    } else {
        $cmdArgs += Get-VideoArgs
    }

    Write-Info "Training task '$task' (num_envs=$NumEnvs)"
    Write-Info "Running from /workspace so logs/videos persist to core/logs/rsl_rl/..."

    Push-Location (Join-Path $ScriptDir "docker")
    try {
        docker compose -f $ComposeFile exec @execEnv -w /workspace isaac-sim $python $script @cmdArgs
    }
    finally {
        Pop-Location
    }
}

function Invoke-Play {
    if (-not $Checkpoint) {
        Write-Err "Play requires -Checkpoint <path> (e.g. ./core/logs/rsl_rl/humanoid/<run>/model_1000.pt)"
        exit 1
    }
    $task = Resolve-Task
    $python = "/isaac-sim/python.sh"
    $script = "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py"

    $cmdArgs = @("--task", $task, "--num_envs", "$NumEnvs", "--checkpoint", $Checkpoint)
    if ($RealTime) { $cmdArgs += "--real-time" }

    $execEnv = @()
    if ($UsdExport) {
        $intervalSec = if ($UsdIntervalSec -ne 1800.0) { $UsdIntervalSec } else { [int][Math]::Round($VideoIntervalMin * 60.0) }
        $lengthSec = if ($UsdLengthSec -ne 10.0) { $UsdLengthSec } else { [int][Math]::Round($VideoLengthMin * 60.0) }
        $execEnv = @("-e", "USD_EXPORT=1", "-e", "USD_INTERVAL=$intervalSec", "-e", "USD_LENGTH=$lengthSec")
        Write-Info "Automated USD & MP4 Export Enabled: Interval=${intervalSec}s (${VideoIntervalMin}min), Length=${lengthSec}s (${VideoLengthMin}min)"
    } else {
        $cmdArgs += Get-VideoArgs
    }

    Write-Info "Playing task '$task' from checkpoint '$Checkpoint'"

    Push-Location (Join-Path $ScriptDir "docker")
    try {
        docker compose -f $ComposeFile exec @execEnv -w /workspace isaac-sim $python $script @cmdArgs
    }
    finally {
        Pop-Location
    }
}

function Invoke-Export {
    param([string]$UsdPath)
    if (-not $UsdPath) {
        Write-Err "Export requires -UsdPath <path> (e.g. ./core/logs/usd/trajectory_t1.usda)"
        exit 1
    }
    $python = "/isaac-sim/python.sh"
    $script = "/workspace/core/utils/usd_to_mp4.py"

    Write-Info "Converting USD trajectory '$UsdPath' to MP4 video..."

    Push-Location (Join-Path $ScriptDir "docker")
    try {
        docker compose -f $ComposeFile exec -w /workspace isaac-sim $python $script $UsdPath
    }
    finally {
        Pop-Location
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

$logsService = if ($args[0]) { $args[0] } else { "isaac-sim" }

switch ($Command) {
    "build" { Invoke-Build }
    "up" { Invoke-Up }
    "train" { Invoke-Train }
    "play" { Invoke-Play }
    "export" { Invoke-Export }
    "logs" { Invoke-Logs -Service $logsService }
    "clean" { Invoke-Clean }
    "kill" { Invoke-Kill }
    default { Show-Usage }
}
