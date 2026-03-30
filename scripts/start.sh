#!/bin/bash

echo "========================================"
echo "  2026 角膜地形图分类系统 - 启动菜单"
echo "========================================"
echo ""

# 检查虚拟环境
if [ ! -f ".venv/Scripts/activate" ]; then
    echo "❌ 虚拟环境不存在！"
    echo "请先运行：./scripts/setup_env.sh"
    echo ""
    exit 1
fi

# 激活虚拟环境
source .venv/Scripts/activate

select choice in "Streamlit Web 界面" "FastAPI 服务" "Docker 一键部署" "退出"; do
    case $choice in
        "Streamlit Web 界面")
            echo ""
            echo "正在启动 Streamlit Web 界面..."
            echo "浏览器访问：http://localhost:8501"
            echo ""
            streamlit run app.py
            break
            ;;
        "FastAPI 服务")
            echo ""
            echo "正在启动 FastAPI 服务..."
            echo "API 文档：http://localhost:8000/docs"
            echo ""
            uvicorn api:app --host 0.0.0.0 --port 8000 --reload
            break
            ;;
        "Docker 一键部署")
            echo ""
            echo "正在启动 Docker 服务..."
            echo "Web 界面：http://localhost:8501"
            echo "API 服务：http://localhost:8000"
            echo ""
            docker-compose up -d
            echo "按 Ctrl+C 查看日志..."
            docker-compose logs -f
            break
            ;;
        "退出")
            echo ""
            echo "感谢使用！再见！"
            echo ""
            break
            ;;
        *)
            echo "无效选项，请重新选择"
            ;;
    esac
done
