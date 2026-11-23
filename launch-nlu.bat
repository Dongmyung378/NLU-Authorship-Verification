@echo off
setlocal
set "ROOT=%~dp0"
if "%PYTHONPATH%"=="" (
    set "PYTHONPATH=%ROOT%"
) else (
    set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
)
if exist "%ROOT%.venv\Scripts\python.exe" (
    "%ROOT%.venv\Scripts\python.exe" -m nlu_authorship %*
) else (
    python -m nlu_authorship %*
)
set "ERR=%ERRORLEVEL%"
endlocal & exit /b %ERR%
