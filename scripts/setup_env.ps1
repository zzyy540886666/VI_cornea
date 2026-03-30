# 2026 角膜地形图分类系统 - 环境配置脚本 (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  2026 角膜地形图分类系统 - 环境配置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 D 盘 Python
Write-Host "📍 检查 D 盘 Python..." -ForegroundColor Yellow
$pythonPath = "D:\Python\Python313\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "❌ 错误：未找到 $pythonPath" -ForegroundColor Red
    Write-Host "请确认 Python 安装路径是否正确" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✅ D 盘 Python 存在" -ForegroundColor Green
& $pythonPath --version
Write-Host ""

# 创建虚拟环境
if (Test-Path ".venv") {
    Write-Host "✅ 虚拟环境已存在，跳过创建" -ForegroundColor Green
} else {
    Write-Host "📦 创建虚拟环境..." -ForegroundColor Yellow
    & $pythonPath -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ 虚拟环境创建失败" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host "✅ 虚拟环境创建成功" -ForegroundColor Green
}
Write-Host ""

# 激活虚拟环境
Write-Host "🔄 激活虚拟环境..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# 升级 pip
Write-Host "🔄 升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
Write-Host ""

# 安装依赖
Write-Host "📥 安装依赖包..." -ForegroundColor Yellow
Write-Host "使用清华镜像源加速下载..." -ForegroundColor Cyan
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ 依赖安装失败" -ForegroundColor Red
    Write-Host "请检查网络连接或手动安装" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "可以运行以下命令手动安装：" -ForegroundColor Cyan
    Write-Host "  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple" -ForegroundColor White
    pause
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✅ 环境配置完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "下次使用时，只需运行：" -ForegroundColor Cyan
Write-Host "  1. .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "或直接运行 .\scripts\start.ps1 启动 Web 界面" -ForegroundColor Cyan
Write-Host ""

pause
