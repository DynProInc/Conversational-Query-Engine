@echo off
echo Stopping existing API server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8002 ^| findstr LISTENING') do (
    echo Found process: %%a
    taskkill /F /PID %%a
    echo Process %%a terminated
)

echo Starting API server...
start cmd /k "python api_server.py"
echo API server started on port 8002
