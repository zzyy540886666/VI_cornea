# 快速启动脚本

@echo off
chcp 65001 >nul
echo ========================================
echo   2026 角膜地形图分类系统 - 启动菜单
echo ========================================
echo.

# 检查虚拟环境
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ 虚拟环境不存在！
    echo 请先运行：scripts\setup_env.bat
    echo.
    pause
    exit /b 1
)

# 激活虚拟环境
call .venv\Scripts\activate.bat

:menu
echo 请选择要启动的服务：
echo.
echo   1. Streamlit Web 界面 (推荐)
echo   2. FastAPI 服务
echo   3. Docker 一键部署
echo   4. 退出
echo.
set /p choice=请输入选项 (1-4): 

if "%choice%"=="1" goto web
if "%choice%"=="2" goto api
if "%choice%"=="3" goto docker
if "%choice%"=="4" goto end
goto menu

:web
echo.
echo 正在启动 Streamlit Web 界面...
echo 浏览器访问：http://localhost:8501
echo.
streamlit run app.py
goto menu

:api
echo.
echo 正在启动 FastAPI 服务...
echo API 文档：http://localhost:8000/docs
echo.
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
goto menu

:docker
echo.
echo 正在启动 Docker 服务...
echo Web 界面：http://localhost:8501
echo API 服务：http://localhost:8000
echo.
docker-compose up -d
echo 按任意键查看日志...
pause
docker-compose logs -f
goto menu

:end
echo.
echo 感谢使用！再见！
echo.
