@echo off
echo ====================================
echo Milvus Multi-Client RAG Setup Script
echo ====================================
echo.

REM Check if Docker is installed
where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed or not in PATH
    echo Please install Docker Desktop for Windows from https://www.docker.com/products/docker-desktop
    exit /b 1
)

echo [INFO] Docker is installed. Checking if Docker is running...

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    exit /b 1
)

echo [INFO] Docker is running. Setting up Milvus environment...

REM Create volumes directory if not exists
if not exist "..\volumes" mkdir "..\volumes"
if not exist "..\logs" mkdir "..\logs"

REM Set Docker volume directory
set DOCKER_VOLUME_DIRECTORY=%cd%\..

echo [INFO] Starting Milvus containers with Docker Compose...
cd ..
docker-compose down
docker-compose up -d

echo.
echo [INFO] Waiting for Milvus to start up...
timeout /t 10 /nobreak

REM Check if containers are running
echo [INFO] Checking container status...
docker ps --filter "name=milvus"

echo.
echo [INFO] Installing Python dependencies...
pip install -r ..\requirements.txt

echo.
echo ==============================================
echo Milvus Multi-Client RAG Setup Complete!
echo ==============================================
echo.
echo Milvus is now running at localhost:19530
echo Minio Console: http://localhost:9001 (minioadmin/minioadmin)
echo.
echo To initialize the RAG system:
echo python multi_client_rag.py
echo.
echo To stop Milvus:
echo docker-compose down
echo ==============================================
