#!/bin/bash

echo "========================================"
echo "  2026 角膜地形图分类系统 - 环境配置"
echo "========================================"
echo ""

echo "📍 检查 D 盘 Python..."
if [ ! -f "D:/Python/Python313/python.exe" ]; then
    echo "❌ 错误：未找到 D:/Python/Python313/python.exe"
    echo "请确认 Python 安装路径是否正确"
    exit 1
fi

echo "✅ D 盘 Python 存在"
D:/Python/Python313/python.exe --version
echo ""

echo "📦 创建虚拟环境..."
if [ -d ".venv" ]; then
    echo "⚠️  虚拟环境已存在，将先删除"
    rm -rf .venv
fi

D:/Python/Python313/python.exe -m venv .venv
if [ $? -ne 0 ]; then
    echo "❌ 虚拟环境创建失败"
    exit 1
fi

echo "✅ 虚拟环境创建成功"
echo ""

echo "🔄 激活虚拟环境..."
source .venv/Scripts/activate

echo "🔄 升级 pip..."
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

echo "📥 安装依赖包..."
echo "使用清华镜像源加速下载..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    echo "请检查网络连接或手动安装"
    exit 1
fi

echo ""
echo "========================================"
echo "  ✅ 环境配置完成！"
echo "========================================"
echo ""
echo "下次使用时，只需运行："
echo "  1. source .venv/Scripts/activate"
echo "  2. streamlit run app.py"
echo ""
echo "或直接运行 ./scripts/start.sh 启动 Web 界面"
echo ""
