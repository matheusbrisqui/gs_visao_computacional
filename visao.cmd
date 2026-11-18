@echo off
rem Atalho do Windows: acha o Python do projeto e repassa os argumentos para visao.py.
rem Exemplos:  visao camera   |   visao video rua.mp4   |   visao testes
setlocal
set "RAIZ=%~dp0"
set "PY="
if exist "%RAIZ%.dados_local" (
  for /f "usebackq delims=" %%L in ("%RAIZ%.dados_local") do if not defined PY if exist "%%L\.venv\Scripts\python.exe" set "PY=%%L\.venv\Scripts\python.exe"
)
if not defined PY if exist "%RAIZ%.venv\Scripts\python.exe" set "PY=%RAIZ%.venv\Scripts\python.exe"
if not defined PY set "PY=python"
cd /d "%RAIZ%"
if /i "%~1"=="testes" (
  "%PY%" -X utf8 -m pytest -q
) else (
  "%PY%" -X utf8 "%RAIZ%visao.py" %*
)
exit /b %ERRORLEVEL%
