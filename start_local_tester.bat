@echo off
setlocal
cd /d %~dp0
start "BGS Browser" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765"
python -m uvicorn local_road_test_app:app --host 127.0.0.1 --port 8765
if errorlevel 1 (
  echo.
  echo 啟動失敗。請先執行: python -m pip install -r requirements.txt
  pause
)
