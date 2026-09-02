@echo off
setlocal enabledelayedexpansion

echo [Ensemble Boot Sequence Initiated]

:: Check if the virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [Creating Python Virtual Environment...]
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Install requirements silently
echo [Verifying Dependencies...]
pip install -r requirements.txt -q

:: Start the FastAPI server in a new minimized command window
echo [Igniting Cognitive Loop on Port 8000...]
start /MIN cmd /c "uvicorn server:app --reload --port 8000"

:: Wait a few seconds for the server to bind to the port
timeout /t 3 /nobreak > nul

:: Open the UI in the default web browser
echo [Launching Interface...]
start index.html

echo [Boot Sequence Complete]
exit