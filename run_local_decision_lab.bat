@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo BGS Local Decision Lab
echo URL: http://127.0.0.1:8787
echo ================================================
start "" http://127.0.0.1:8787
python -m uvicorn local_decision_lab:app --host 127.0.0.1 --port 8787 --reload
if errorlevel 1 (
  echo.
  echo Failed to start. Run: pip install -r requirements.txt
  pause
)
endlocal
