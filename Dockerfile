# 多阶段构建 - 优化镜像大小
FROM python:3.10-slim as builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir --user -r requirements.txt

# 运行时镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 从 builder 阶段复制已安装的包
COPY --from=builder /root/.local /root/.local

# 确保脚本路径在 PATH 中
ENV PATH=/root/.local/bin:$PATH

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 复制应用代码
COPY model_service.py .
COPY app.py .
COPY api.py .
COPY predict.py .
COPY checkpoints/ ./checkpoints/

# 创建数据目录
RUN mkdir -p /app/data

# 暴露端口
# Streamlit 默认端口
EXPOSE 8501
# FastAPI 默认端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# 启动命令（默认启动 Streamlit）
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
