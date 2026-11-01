@echo off
chcp 65001 >nul
title 人机协同图像标注系统
cd /d %~dp0

set "CONDA=C:\Users\c1083\miniforge3"
if not exist "%CONDA%\Scripts\activate.bat" set "CONDA=%USERPROFILE%\miniforge3"

if not exist "%CONDA%\Scripts\activate.bat" (
    echo 没找到 Miniforge，请确认它装在 %USERPROFILE%\miniforge3
    pause
    exit /b 1
)

call "%CONDA%\Scripts\activate.bat" imglabel
echo 正在启动，浏览器会自动打开 http://127.0.0.1:5000
python app.py
pause
