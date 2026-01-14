@echo off
echo ========================================
echo        AI Derm - Skin Lesion Analysis
echo ========================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>NUL
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo Starting AI Derm...
streamlit run Home.py --server.headless true

pause
