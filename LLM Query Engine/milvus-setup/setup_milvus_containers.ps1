# setup_milvus_containers.ps1
# Script to set up Milvus containers for RAG system
# This script should be placed in the milvus-setup directory

Write-Host "Setting up Milvus containers for RAG system..." -ForegroundColor Green

# Get the current directory (should be the milvus-setup directory)
$currentDir = Get-Location
Write-Host "Running from directory: $currentDir" -ForegroundColor Cyan

# Check if Docker is installed and running
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker Desktop is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check Docker is running
try {
    $dockerStatus = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# Check if docker-compose.yml exists in current directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "docker-compose.yml not found in current directory." -ForegroundColor Red
    Write-Host "Please run this script from the milvus-setup directory." -ForegroundColor Yellow
    exit 1
}

# Check if containers exist and their status
Write-Host "Checking existing Milvus containers..." -ForegroundColor Cyan
$containersExist = $false
$containersRunning = $true

foreach ($container in @("milvus-standalone", "milvus-etcd", "milvus-minio")) {
    $containerStatus = docker ps -a --filter "name=$container" --format "{{.Status}}"
    
    if ($containerStatus) {
        $containersExist = $true
        Write-Host "$container exists. Status: $containerStatus" -ForegroundColor Cyan
        
        if (-not ($containerStatus -match "Up")) {
            $containersRunning = $false
        }
    } else {
        Write-Host "$container does not exist." -ForegroundColor Yellow
        $containersExist = $false
        $containersRunning = $false
    }
}

# Handle different scenarios
if ($containersExist -and $containersRunning) {
    Write-Host "All Milvus containers are already running." -ForegroundColor Green
} elseif ($containersExist -and -not $containersRunning) {
    # Containers exist but not running - start them
    Write-Host "Starting existing Milvus containers..." -ForegroundColor Cyan
    foreach ($container in @("milvus-etcd", "milvus-minio", "milvus-standalone")) {
        docker start $container
        Write-Host "Started $container" -ForegroundColor Green
    }
} else {
    # Containers don't exist or some are missing - recreate all
    Write-Host "Creating Milvus containers using docker-compose..." -ForegroundColor Cyan
    
    # Remove any existing containers first
    docker-compose down -v
    
    # Create and start containers
    docker-compose up -d
    
    Write-Host "Waiting for containers to initialize..." -ForegroundColor Cyan
    Start-Sleep -Seconds 10
}

# Verify containers are running
$allRunning = $true
foreach ($container in @("milvus-standalone", "milvus-etcd", "milvus-minio")) {
    $containerStatus = docker ps --filter "name=$container" --format "{{.Status}}"
    
    if (-not ($containerStatus -match "Up")) {
        Write-Host "$container is not running." -ForegroundColor Red
        $allRunning = $false
    } else {
        Write-Host "$container is running: $containerStatus" -ForegroundColor Green
    }
}

if ($allRunning) {
    Write-Host "All Milvus containers are now running successfully!" -ForegroundColor Green
    
    # Display container information
    Write-Host "`nContainer Details:" -ForegroundColor Cyan
    docker ps --filter "name=milvus"
} else {
    Write-Host "Some containers failed to start. Please check Docker logs:" -ForegroundColor Red
    Write-Host "docker logs milvus-standalone" -ForegroundColor Yellow
}

Write-Host "`nMilvus container setup completed!" -ForegroundColor Green
Write-Host "You can now run the API server with: python api_server.py --with-rag --port 8002" -ForegroundColor Yellow

# Keep the window open until user presses a key
Write-Host "`nPress any key to exit..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
