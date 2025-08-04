@echo off
REM Milvus RAG System Setup Script for Windows (Batch version)

echo ==================================
echo Milvus RAG System Setup (Windows)
echo ==================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not installed or not in PATH
    echo Please install Docker Desktop for Windows from: https://docs.docker.com/desktop/windows/
    pause
    exit /b 1
)
echo ✓ Docker is installed

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop
    pause
    exit /b 1
)
echo ✓ Docker is running

REM Create directories
echo Creating data directories...
if not exist "volumes" mkdir volumes
if not exist "volumes\etcd" mkdir volumes\etcd
if not exist "volumes\minio" mkdir volumes\minio
if not exist "volumes\milvus" mkdir volumes\milvus
echo ✓ Data directories created

REM Try docker-compose first, then docker compose
docker-compose --version >nul 2>&1
if not errorlevel 1 (
    set COMPOSE_CMD=docker-compose
) else (
    docker compose version >nul 2>&1
    if not errorlevel 1 (
        set COMPOSE_CMD=docker compose
    ) else (
        echo Error: Docker Compose not found
        pause
        exit /b 1
    )
)
echo ✓ Docker Compose found

REM Start Milvus containers
echo Starting Milvus containers...
%COMPOSE_CMD% up -d

if errorlevel 1 (
    echo Failed to start Milvus containers
    echo Check logs with: %COMPOSE_CMD% logs
    pause
    exit /b 1
)
echo ✓ Milvus containers started

REM Wait for services
echo Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak >nul

echo.
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo Services running:
echo • Milvus Vector Database: http://localhost:19530
echo • Milvus Web UI: http://localhost:9091
echo • MinIO Console: http://localhost:9001 (admin/minioadmin)
echo.
echo Next steps:
echo 1. Install Python: pip install -r requirements.txt  
echo 2. Configure .env with your OpenAI API key
echo 3. Run: python quick_start.py
echo.
echo To stop: %COMPOSE_CMD% down
pause
