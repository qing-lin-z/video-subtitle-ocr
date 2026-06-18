@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

REM 添加 NVIDIA CUDA 12 DLL 到 PATH（遍历所有 nvidia wheel 的 bin 目录）
set "NVROOT=%APPDATA%\Python\Python313\site-packages\nvidia"
if exist "%NVROOT%" (
    for /d %%d in ("%NVROOT%\*") do (
        if exist "%%d\bin" set "PATH=%%d\bin;!PATH!"
    )
)

"C:\ProgramData\miniconda3\pythonw.exe" "%~dp0video_subtitle_ocr_gui.py"
