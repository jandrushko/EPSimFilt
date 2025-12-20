#!/bin/bash

# MEP Filter Tool Launcher Script
# This script checks dependencies and launches the application

echo "=========================================="
echo "MEP Filter Testing Tool"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo ""
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies."
        echo "Please try manually: pip install -r requirements.txt"
        exit 1
    fi
    echo "✓ Dependencies installed"
fi

# Launch the application
echo ""
echo "🚀 Launching MEP Filter Testing Tool..."
echo ""
echo "The tool will open in your default web browser."
echo "If it doesn't, manually navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
