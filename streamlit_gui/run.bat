@echo off
echo ========================================
echo        DermAI - Skin Lesion Analysis
echo ========================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>NUL
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo Starting DermAI...
streamlit run Home.py --server.headless true

pause
