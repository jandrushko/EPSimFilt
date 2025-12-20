@echo off
REM MEP Filter Tool Launcher Script for Windows
REM This script checks dependencies and launches the application

echo ==========================================
echo MEP Filter Testing Tool
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python is not installed.
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo + Python found
python --version

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ! Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
    
    if %errorlevel% neq 0 (
        echo X Failed to install dependencies.
        echo Please try manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo + Dependencies installed
)

REM Launch the application
echo.
echo Starting MEP Filter Testing Tool...
echo.
echo The tool will open in your default web browser.
echo If it doesn't, manually navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py
