# 2026 角膜地形图分类系统 - 启动脚本 (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  2026 角膜地形图分类系统 - 启动菜单" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查虚拟环境
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "❌ 虚拟环境不存在！" -ForegroundColor Red
    Write-Host "请先运行：.\scripts\setup_env.ps1" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

# 激活虚拟环境
& ".venv\Scripts\Activate.ps1"

function Show-Menu {
    Write-Host "请选择要启动的服务：" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1. Streamlit Web 界面 (推荐)" -ForegroundColor White
    Write-Host "  2. FastAPI 服务" -ForegroundColor White
    Write-Host "  3. Docker 一键部署" -ForegroundColor White
    Write-Host "  4. 退出" -ForegroundColor White
    Write-Host ""
}

do {
    Show-Menu
    $choice = Read-Host "请输入选项 (1-4)"
    
    switch ($choice) {
        '1' {
            Write-Host ""
            Write-Host "正在启动 Streamlit Web 界面..." -ForegroundColor Green
            Write-Host "浏览器访问：http://localhost:8501" -ForegroundColor Cyan
            Write-Host ""
            streamlit run app.py
        }
        '2' {
            Write-Host ""
            Write-Host "正在启动 FastAPI 服务..." -ForegroundColor Green
            Write-Host "API 文档：http://localhost:8000/docs" -ForegroundColor Cyan
            Write-Host ""
            uvicorn api:app --host 0.0.0.0 --port 8000 --reload
        }
        '3' {
            Write-Host ""
            Write-Host "正在启动 Docker 服务..." -ForegroundColor Green
            Write-Host "Web 界面：http://localhost:8501" -ForegroundColor Cyan
            Write-Host "API 服务：http://localhost:8000" -ForegroundColor Cyan
            Write-Host ""
            docker-compose up -d
            Write-Host "按任意键查看日志..." -ForegroundColor Yellow
            pause | Out-Null
            docker-compose logs -f
        }
        '4' {
            Write-Host ""
            Write-Host "感谢使用！再见！" -ForegroundColor Green
            Write-Host ""
        }
    }
} while ($choice -ne '4')
