@echo off
cd /d "%~dp0"
start "Walnut Backend" cmd /k "cd backend && python main.py"
start "Walnut Frontend" cmd /k "cd frontend && npm run dev"
echo Application starting...
timeout /t 5
start http://localhost:5173
