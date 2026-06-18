@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 添加 NVIDIA CUDA 12 DLL 到 PATH
set "NVROOT=%APPDATA%\Python\Python313\site-packages\nvidia"
for /d %%d in ("%NVROOT%\*") do (
    if exist "%%d\bin" set "PATH=%%d\bin;%PATH%"
)

"C:\ProgramData\miniconda3\python.exe" "%~dp0video_subtitle_ocr.py" %*
