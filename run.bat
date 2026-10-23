@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting SmartTopic_LDA Streamlit Application...
echo ===================================================
call .venv\Scripts\activate.bat
streamlit run app.py
pause
