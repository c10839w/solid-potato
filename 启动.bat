@echo off
chcp 65001 >nul
title 人机协同图像标注系统
cd /d %~dp0

rem 第二轮：如果存在自己训练的权重，就优先用它做预标注
set "MODEL_ARGS="
if exist "model\best.pt" set "MODEL_ARGS=--model model/best.pt"

rem 先找 Miniforge / Anaconda；找不到就直接用系统里的 python（别人拿到仓库也能跑）
set "CONDA="
if exist "C:\Users\c1083\miniforge3\Scripts\activate.bat" set "CONDA=C:\Users\c1083\miniforge3"
if not defined CONDA if exist "%USERPROFILE%\miniforge3\Scripts\activate.bat" set "CONDA=%USERPROFILE%\miniforge3"
if not defined CONDA if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA=%USERPROFILE%\anaconda3"
if not defined CONDA if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA=%USERPROFILE%\miniconda3"

if defined CONDA (
    call "%CONDA%\Scripts\activate.bat" imglabel 2>nul
)

where python >nul 2>nul
if errorlevel 1 (
    echo 没找到 Python。请先安装 Miniforge/Anaconda，或把 python 加入 PATH。
    echo 装好后重新双击本文件即可。
    pause
    exit /b 1
)

echo 正在启动，浏览器会自动打开 http://127.0.0.1:5000
echo 关闭这个黑窗口即停止运行
python app.py %MODEL_ARGS%
echo.
echo 服务已退出。如果是报错，把上面的红字截图发我。
pause
