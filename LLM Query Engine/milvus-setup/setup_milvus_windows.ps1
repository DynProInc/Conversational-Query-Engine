# Milvus RAG System Setup Script for Windows
# PowerShell version of setup_milvus.sh for Windows users

Write-Host "=================================="
Write-Host "Milvus RAG System Setup (Windows)"  
Write-Host "=================================="

# Check if Docker is installed and running
Write-Host "Checking Docker installation..."
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Docker Desktop for Windows from: https://docs.docker.com/desktop/windows/" -ForegroundColor Yellow
    exit 1
}

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop" -ForegroundColor Red
    exit 1
}

# Check if Docker Compose is available
try {
    $composeVersion = docker-compose --version
    Write-Host "✓ Docker Compose found: $composeVersion" -ForegroundColor Green
    $composeCmd = "docker-compose"
} catch {
    try {
        $composeVersion = docker compose version
        Write-Host "✓ Docker Compose found: $composeVersion" -ForegroundColor Green  
        $composeCmd = "docker compose"
    } catch {
        Write-Host "❌ Docker Compose not found" -ForegroundColor Red
        exit 1
    }
}

# Create necessary directories
Write-Host "Creating data directories..."
New-Item -ItemType Directory -Force -Path "volumes\etcd" | Out-Null
New-Item -ItemType Directory -Force -Path "volumes\minio" | Out-Null  
New-Item -ItemType Directory -Force -Path "volumes\milvus" | Out-Null
Write-Host "✓ Data directories created" -ForegroundColor Green

# Stop any existing containers
Write-Host "Stopping any existing Milvus containers..."
& $composeCmd down 2>$null

# Start Milvus containers
Write-Host "Starting Milvus containers..."
& $composeCmd up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Milvus containers started successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to start Milvus containers" -ForegroundColor Red
    Write-Host "Check Docker logs with: $composeCmd logs" -ForegroundColor Yellow
    exit 1
}

# Wait for services to be ready
Write-Host "Waiting for services to be ready (30 seconds)..."
Start-Sleep -Seconds 30

# Check service health
Write-Host "Checking service health..."

# Test Milvus health
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9091/healthz" -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Milvus is healthy" -ForegroundColor Green
    } else {
        Write-Host "⚠ Milvus health check returned: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Milvus might still be starting up" -ForegroundColor Yellow
}

# Test MinIO health  
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ MinIO is healthy" -ForegroundColor Green
    } else {
        Write-Host "⚠ MinIO health check returned: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ MinIO might still be starting up" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=================================="
Write-Host "Setup Complete!"
Write-Host "=================================="
Write-Host ""
Write-Host "Services running:"
Write-Host "• Milvus Vector Database: http://localhost:19530"
Write-Host "• Milvus Web UI: http://localhost:9091"
Write-Host "• MinIO Console: http://localhost:9001 (admin/minioadmin)"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Install Python dependencies: pip install -r requirements.txt"
Write-Host "2. Copy .env.template to .env and add your OpenAI API key"  
Write-Host "3. Run the RAG pipeline: python database_schema_rag_pipeline.py"
Write-Host ""
Write-Host "To stop services: $composeCmd down"
Write-Host "To restart services: $composeCmd up -d"
Write-Host "To view logs: $composeCmd logs -f"
