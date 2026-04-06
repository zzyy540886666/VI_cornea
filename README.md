# 2026 角膜地形图分类系统 - 完整部署与使用指南

> **项目简介**：使用 2025-2026 年最新医学影像 SOTA 模型（ConvNeXt V2、MaxViT、Swin Transformer V2），实现角膜地形图自动分类（正常 / 圆锥角膜），辅助近视手术术前评估。

---

## 项目亮点

| 特性 | 说明 |
|------|------|
| 最新模型 | ConvNeXt V2（2025 NVIDIA 优化版）、MaxViT、Swin Transformer V2 |
| 医学影像专用 | 针对角膜地形图优化的数据增强和训练策略 |
| 零基础友好 | 复制粘贴即可运行，无需深度学习基础 |
| 高精度 | 迁移学习 + 预训练权重，小样本也能达到 95%+ 准确率 |
| 完整流程 | 数据准备 -> 模型训练 -> 预测评估 -> Web 部署，一站式解决 |
| 多端部署 | 支持 Streamlit Web 界面、FastAPI 接口、Docker 容器化部署 |

---

## 一、环境要求

### 1.1 最低配置

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11、Linux、macOS |
| Python | 3.8 或更高版本 |
| 内存 | 4 GB |
| 处理器 | 任意现代 CPU |
| GPU | 无（可用 CPU 运行，训练较慢） |
| 磁盘 | 5 GB 可用空间 |

### 1.2 推荐配置

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 |
| Python | 3.10 或更高版本 |
| 内存 | 8 GB 或更多 |
| 处理器 | 多核 CPU |
| GPU | NVIDIA GPU（GTX 1060 或更高，显存 4 GB+） |
| 磁盘 | 10 GB+ 可用空间 |

### 1.3 所需依赖包一览

项目依赖以下核心库：

| 包名 | 用途 | 版本要求 |
|------|------|---------|
| torch | 深度学习框架 | >= 2.4.0 |
| torchvision | 图像处理与数据加载 | >= 0.19.0 |
| timm | PyTorch Image Models（包含最新模型） | >= 1.0.0 |
| opencv-python | 图像读取与预处理 | >= 4.10.0 |
| pillow | 图像格式支持 | >= 10.0.0 |
| scikit-learn | 评估指标计算 | >= 1.3.0 |
| matplotlib | 可视化（训练曲线、混淆矩阵） | >= 3.7.0 |
| numpy | 数值计算 | >= 1.24.0 |
| pandas | 数据导出（CSV） | >= 2.0.0 |
| streamlit | Web 界面 | >= 1.28.0 |
| fastapi | REST API 框架 | >= 0.104.0 |
| uvicorn | ASGI 服务器 | >= 0.24.0 |

---

## 二、环境安装（详细步骤）

### 2.1 安装 Python（如果还没有）

> **检查是否已安装**：打开 PowerShell 或 CMD，输入 `python --version`，如果显示版本号则跳过此步。

1. 访问 [Python 官网](https://www.python.org/downloads/)
2. 下载 **Python 3.10** 或更高版本（推荐 3.13）
3. 运行安装程序，**务必勾选 "Add Python to PATH"**（非常重要！）
4. 点击 "Install Now" 完成安装
5. 重新打开终端，验证安装：
   ```powershell
   python --version
   # 应显示类似：Python 3.13.7
   ```

### 2.2 方式一：一键脚本安装（推荐小白使用）

打开 PowerShell，进入项目目录：

```powershell
cd d:\设计大赛_视觉识别
```

**首次运行**需要允许 PowerShell 执行脚本（仅需操作一次）：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**运行一键配置脚本**：

```powershell
.\scripts\setup_env.ps1
```

脚本会自动完成以下步骤：

```
1. 检查 Python 是否存在
2. 创建虚拟环境 (.venv)
3. 激活虚拟环境
4. 升级 pip 到最新版本
5. 使用清华镜像源安装所有依赖包
```

预计耗时：**5-15 分钟**（取决于网速，PyTorch 约 200 MB）

### 2.3 方式二：手动安装（如脚本出错时使用）

**第 1 步：创建虚拟环境**

```powershell
python -m venv .venv
```

> 虚拟环境的作用：将项目依赖与系统 Python 隔离，避免不同项目之间的包冲突。

**第 2 步：激活虚拟环境**

PowerShell：
```powershell
.\.venv\Scripts\Activate.ps1
```

CMD：
```cmd
.\.venv\Scripts\activate.bat
```

激活成功后，命令行前面会出现 `(.venv)` 前缀，例如：
```
(.venv) PS D:\设计大赛_视觉识别>
```

> 如果 PowerShell 提示"无法加载文件，因为未签名"，先运行：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**第 3 步：升级 pip**

```powershell
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**第 4 步：安装所有依赖**

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 使用清华镜像源可以大幅加速下载。如果清华源不可用，可替换为：
> - 中科大源：`https://pypi.mirrors.ustc.edu.cn/simple`
> - 官方源：`https://pypi.org/simple`

### 2.4 验证安装是否成功

逐一运行以下命令，确认核心包可用：

```powershell
# 1. 检查 Python 版本
python --version
# 应显示：Python 3.13.x

# 2. 检查 PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"
# 应显示：PyTorch 2.4.x

# 3. 检查是否有 GPU
python -c "import torch; print('GPU 可用' if torch.cuda.is_available() else '使用 CPU')"
# 有 GPU 显示：GPU 可用
# 无 GPU 显示：使用 CPU

# 4. 检查 Streamlit
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"

# 5. 检查 timm
python -c "import timm; print(f'timm {timm.__version__}')"
```

如果以上命令全部正常输出，说明环境安装成功。

### 2.5 安装验证清单

- [ ] `.venv` 文件夹存在
- [ ] `python --version` 显示 3.10+ 版本
- [ ] `import torch` 不报错
- [ ] `import streamlit` 不报错
- [ ] `import timm` 不报错

---
## 三、数据准备

### 3.1 数据目录结构

项目数据已包含在 `data/` 目录中：

```
data/
├── kc/              # 圆锥角膜（异常）图片
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── ...
└── normal/          # 正常角膜图片
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    └── ...
```

### 3.2 自动划分数据集

运行数据划分脚本，自动将原始数据按 **70% 训练 / 15% 验证 / 15% 测试** 的比例划分：

```powershell
python prepare_data.py
```

运行成功后生成 `dataset/` 目录：

```
dataset/
├── train/           # 训练集（70%）- 用于训练模型
│   ├── kc/
│   └── normal/
├── val/             # 验证集（15%）- 用于训练时调参和选择最佳模型
│   ├── kc/
│   └── normal/
└── test/            # 测试集（15%）- 用于最终评估模型性能
    ├── kc/
    └── normal/
```

### 3.3 使用自己的数据

如果你想使用自己的角膜地形图数据：

1. 将异常（圆锥角膜）图片放入 `data/kc/` 文件夹
2. 将正常角膜图片放入 `data/normal/` 文件夹
3. 支持的图片格式：`.jpg`、`.jpeg`、`.png`、`.bmp`
4. 重新运行 `python prepare_data.py`
5. 重新运行 `python train.py` 训练新模型

> 图片数量建议：每个类别至少 50 张以上，越多越好。

---

## 四、模型训练

### 4.1 开始训练

确保虚拟环境已激活，然后运行：

```powershell
python train.py
```

### 4.2 训练过程说明

训练开始后，你会看到类似以下的输出：

```
============================================================
2026 角膜地形图分类模型训练
使用模型：ConvNeXt V2 (医学影像 SOTA)
============================================================
使用设备：cuda
加载模型：convnextv2_base
模型初始化完成
   输入尺寸：224x224
   参数量：88,591,394

数据集统计：
   训练集：350 张
   验证集：75 张
   测试集：75 张

============================================================
开始训练
============================================================
Epoch 1: 100%|████████████████| loss: 0.6543, acc: 62.50%
Validating Epoch 1: 100%|████████████████| loss: 0.5234, acc: 75.00%
   保存最佳模型 (准确率：75.00%)

Epoch 2: 100%|████████████████| loss: 0.4321, acc: 82.14%
Validating Epoch 2: 100%|████████████████| loss: 0.3456, acc: 88.00%
   保存最佳模型 (准确率：88.00%)

...

训练完成！最佳验证准确率：96.50%

分类报告：
              precision    recall  f1-score   support
      kc         0.96      0.97      0.96        38
   normal       0.97      0.96      0.96        37
    accuracy                         0.96        75
```

**输出指标解释：**

| 指标 | 含义 |
|------|------|
| loss（损失） | 越低越好，表示预测结果与真实标签的差距 |
| acc（准确率） | 越高越好，表示预测正确的比例 |
| Precision（精确率） | 预测为异常的样本中，真正异常的比例 |
| Recall（召回率） | 所有异常样本中，被正确识别的比例 |
| F1-Score | 精确率和召回率的调和平均 |

### 4.3 训练完成后生成的文件

训练完成后，`checkpoints/` 目录下会生成：

```
checkpoints/
├── best_model.pth              # 最佳模型权重（用于预测）
├── training_curves.png         # 训练曲线图（loss 和 accuracy 随 epoch 变化）
├── confusion_matrix.png        # 混淆矩阵（测试集分类效果可视化）
```

- **training_curves.png**：左图为损失曲线（应逐渐下降），右图为准确率曲线（应逐渐上升）
- **confusion_matrix.png**：对角线数值越高越好，表示分类正确的样本数

### 4.4 切换不同的 SOTA 模型

打开 `train.py`，找到 `model_name` 参数并修改：

```python
# 可选模型（均为 2025-2026 年医学影像 SOTA）

# 推荐：速度和精度的最佳平衡
model_name = 'convnextv2_base'

# 轻量级：参数少、推理快，适合资源有限或移动端部署
model_name = 'maxvit_tiny_224'

# Transformer 架构：全局特征提取能力强，适合科研论文
model_name = 'swinv2_base_window16_256'

# 最小模型：参数量最少，显存占用最小
model_name = 'convnextv2_nano'
```

**模型对比：**

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| **ConvNeXt V2 Base** | 88M | 快 | 高 | 推荐：平衡性最好 |
| **ConvNeXt V2 Nano** | 15M | 很快 | 中 | 显存不足时使用 |
| **MaxViT Tiny** | 30M | 很快 | 中高 | 速度优先、移动端 |
| **Swin Transformer V2** | 87M | 中 | 高 | 精度优先、科研论文 |

### 4.5 常见训练问题排查

#### 显存不足（CUDA out of memory）

**方法 1**：减小 batch size

在 `train.py` 中找到 `batch_size` 参数，从默认值改小：
```python
trainer.load_data(data_dir='dataset', batch_size=8)   # 原值可能为 16 或 32
trainer.load_data(data_dir='dataset', batch_size=4)   # 再小一点
```

**方法 2**：使用更小的模型
```python
model_name = 'convnextv2_nano'    # 仅 15M 参数
```

#### 准确率不高

- **增加训练轮次**：
  ```python
  trainer.train(epochs=50, save_dir='checkpoints')   # 默认可能为 30
  ```
- **降低学习率**（更精细的权重调整）：
  ```python
  lr=5e-5    # 在 trainer 初始化时设置
  ```
- **增加数据量**：每个类别至少 100 张图片
- **检查数据标注**：确保图片分类正确

#### 训练速度太慢

- 有 NVIDIA GPU 确保安装了 CUDA 版 PyTorch：
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- 减小模型或 batch size 可减少单轮耗时

---

## 五、模型预测

### 5.1 命令行预测单张图片

```powershell
python predict.py --image data/kc/1.jpg
```

输出示例：

```
============================================================
2026 角膜地形图分类预测系统
============================================================
使用设备：cuda
加载模型：checkpoints/best_model.pth
模型加载成功
   验证准确率：96.50%
   训练轮次：28

预测图片：data/kc/1.jpg

预测结果:
   类别：圆锥角膜（异常）
   置信度：98.76%
   建议：不建议进行近视激光手术，建议进一步检查
```

### 5.2 预测并保存可视化结果图

```powershell
python predict.py --image data/normal/50.jpg --save result.png
```

生成的 `result.png` 包含：
- 原始角膜地形图
- 预测类别和置信度
- 正常/异常判断和手术建议

### 5.3 批量预测整个文件夹

```powershell
# 预测 data/kc/ 下所有图片
python predict.py --dir data/kc
```

运行后自动生成 `prediction_results.csv`，包含每张图片的文件名、预测类别、置信度。

### 5.4 Web 界面预测

启动 Web 服务后（见下一章），在浏览器中通过可视化界面上传图片进行预测，详见第六章。

---

## 六、启动服务

### 6.1 方式一：使用启动脚本（最简单）

确保虚拟环境已创建并安装好依赖后，运行：

```powershell
.\start.ps1
```

会出现交互菜单：

```
========================================
  2026 角膜地形图分类系统 - 启动菜单
========================================

请选择要启动的服务：

  1. Streamlit Web 界面 (推荐)
  2. FastAPI 服务
  3. Docker 一键部署
  4. 退出

请输入选项 (1-4):
```

输入对应数字即可启动相应服务。

### 6.2 方式二：手动启动 Streamlit Web 界面（推荐）

```powershell
# 1. 激活虚拟环境（如果还没激活）
.\.venv\Scripts\Activate.ps1

# 2. 启动 Web 服务
streamlit run app.py
```

启动成功后，浏览器自动打开 **http://localhost:8501**

**Web 界面功能：**

| 功能 | 说明 |
|------|------|
| 单张图片预测 | 拖拽或点击上传一张角膜地形图，实时显示预测结果 |
| 批量预测 | 一次上传多张图片，自动分类统计 |
| 置信度可视化 | 显示各类别的概率分布和置信度进度条 |
| CSV 下载 | 批量预测结果可导出为 CSV 文件 |
| 模型信息 | 显示当前模型的版本、准确率等详细信息 |

**修改 Web 端口：**

```powershell
streamlit run app.py --server.port=8502
```

### 6.3 方式三：启动 FastAPI 接口服务

```powershell
# 1. 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 2. 启动 API 服务
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后访问 API 文档：**http://localhost:8000/docs**

**API 接口说明：**

**1. 健康检查**
```powershell
curl http://localhost:8000/health
```

**2. 单张图片预测**
```powershell
curl -X POST "http://localhost:8000/predict" -F "image=@data/kc/1.jpg"
```
返回示例：
```json
{
  "class": "kc",
  "class_name": "圆锥角膜（异常）",
  "confidence": 98.76,
  "probabilities": {
    "kc": 0.9876,
    "normal": 0.0124
  },
  "suggestion": "不建议进行近视激光手术，建议进一步检查"
}
```

**3. 批量预测**
```powershell
curl -X POST "http://localhost:8000/predict/batch" -F "images=@data/kc/1.jpg" -F "images=@data/kc/2.jpg"
```

**4. 获取模型信息**
```powershell
curl http://localhost:8000/model/info
```

**修改 API 端口：**
```powershell
uvicorn api:app --host 0.0.0.0 --port=8001 --reload
```

### 6.4 方式四：Docker 一键部署（生产环境）

#### 前提条件

- 已安装 [Docker Desktop](https://www.docker.com/)
- 已安装 Docker Compose（Docker Desktop 自带）

#### 构建并启动

```powershell
docker-compose up -d --build
```

#### 访问服务

| 服务 | 地址 |
|------|------|
| Streamlit Web 界面 | http://localhost:8501 |
| FastAPI 服务 | http://localhost:8000 |
| API 接口文档 | http://localhost:8000/docs |

#### 常用 Docker 命令

```powershell
# 查看服务状态
docker-compose ps

# 查看实时日志
docker-compose logs -f

# 停止所有服务
docker-compose down

# 重启服务
docker-compose restart

# 重新构建并启动
docker-compose up -d --build
```

#### 修改 Docker 端口

编辑 `docker-compose.yml`，修改端口映射：
```yaml
ports:
  - "8502:8501"    # Web 界面（外部 8502 -> 内部 8501）
  - "8001:8000"    # API 服务（外部 8001 -> 内部 8000）
```

### 6.5 远程部署（让局域网/外网访问）

#### 局域网内访问

默认 Streamlit 只监听 `localhost`，需要改为监听所有网卡：

```powershell
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

同局域网内的其他设备通过 `http://你的IP:8501` 访问（查看本机 IP：`ipconfig`）。

#### 公网部署（方案对比）

| 方案 | 难度 | 费用 | 适用场景 |
|------|------|------|----------|
| **Streamlit Community Cloud** | 最低 | 免费 | 快速演示、比赛提交 |
| **腾讯云 Lighthouse** | 中等 | 按量付费 | 长期稳定运行 |
| **自有服务器 + Docker** | 较高 | 自有成本 | 完全掌控 |

**方案一：Streamlit Community Cloud（免费，推荐比赛用）**

1. 将项目推送到 GitHub/Gitee 仓库
2. 访问 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 登录
3. 点击 "New app"，选择仓库和 `app.py` 分支
4. 自动部署，获得公开 URL

> 注意：模型文件 `best_model.pth` 约 1 GB，超出 GitHub 单文件限制。需先上传到 Hugging Face / 网盘，然后在代码中改为在线下载。

**方案二：腾讯云轻量服务器**

```bash
# 1. 购买轻量应用服务器（推荐 Ubuntu 系统）
# 2. SSH 登录服务器
ssh root@你的服务器IP

# 3. 安装 Docker
curl -fsSL https://get.docker.com | sh

# 4. 上传项目文件（用 scp 或 Git）
git clone 你的仓库地址
cd 设计大赛_视觉识别

# 5. Docker 一键启动
docker-compose up -d --build

# 6. 服务器防火墙放行 8501 端口（在腾讯云控制台操作）
```

访问地址：`http://服务器公网IP:8501`

**方案三：内网穿透（临时演示）**

没有公网服务器时，用内网穿透工具临时暴露到公网：

```powershell
# 安装 ngrok
choco install ngrok

# 穿透 8501 端口
ngrok http 8501
```

会得到一个随机公网 URL（如 `https://xxxx.ngrok-free.app`），有效期 8 小时。

---

## 七、项目文件说明

### 7.1 完整文件结构

```
设计大赛_视觉识别/
├── app.py                  # Streamlit Web 应用（可视化界面）
├── api.py                  # FastAPI REST 接口服务
├── model_service.py        # 模型服务封装（单例模式，高效推理）
├── train.py                # 模型训练脚本
├── predict.py              # 命令行预测脚本
├── prepare_data.py         # 数据预处理与划分
├── requirements.txt        # Python 依赖包列表
├── setup_env.ps1           # 环境一键配置（PowerShell）
├── setup_env.bat           # 环境一键配置（CMD）
├── start.ps1               # 启动菜单脚本（PowerShell）
├── start.bat               # 启动菜单脚本（CMD）
├── Dockerfile              # Docker 镜像构建配置
├── docker-compose.yml      # Docker 服务编排配置
├── README.md               # 本文档
├── data/                   # 原始数据（kc/ 和 normal/）
├── dataset/                # 划分后的训练/验证/测试集
└── checkpoints/            # 训练产物（模型权重、曲线图）
```

### 7.2 各文件详细说明

| 文件 | 功能 | 详细说明 |
|------|------|---------|
| `train.py` | 模型训练 | 包含数据加载、模型初始化、训练循环、验证、评估，训练完成后保存最佳模型和可视化图表 |
| `predict.py` | 命令行预测 | 支持单张预测（`--image`）、保存结果图（`--save`）、批量预测（`--dir`） |
| `prepare_data.py` | 数据预处理 | 读取 `data/` 目录，按 70/15/15 比例划分到 `dataset/` |
| `app.py` | Web 界面 | Streamlit 应用，提供上传预测、批量预测、结果下载等功能 |
| `api.py` | API 服务 | FastAPI 应用，提供 `/predict`、`/predict/batch`、`/health`、`/model/info` 接口 |
| `model_service.py` | 模型服务 | 封装模型加载和预测逻辑，单例模式确保模型只加载一次 |

---

## 八、技术原理（可选阅读）

### 8.1 为什么选择 ConvNeXt V2？

ConvNeXt V2 是 NVIDIA 在 2025 年优化的卷积神经网络，结合了：

1. **卷积的局部特征提取能力** - 适合角膜纹理分析，能有效捕捉角膜表面的细微结构变化
2. **全局响应归一化（GRN）** - 增强特征竞争，提升模型对关键特征的敏感度
3. **掩码自编码器预训练（FCMAE）** - 在大规模数据上预训练，获得更强的特征表示能力

在医学影像分类任务中，ConvNeXt V2 表现优于传统 ResNet 和早期 Transformer。

### 8.2 迁移学习原理

- **预训练权重**：模型已在 ImageNet（1400 万张图片）上学习了通用的图像特征（边缘、纹理、形状等）
- **微调**：只需调整最后几层权重，使其适应角膜地形图的特定特征
- **小样本友好**：即使只有几百张医学图片，利用预训练知识也能达到很高的分类精度

### 8.3 数据增强策略

训练时自动应用以下增强操作，提升模型的泛化能力：

| 增强方法 | 说明 |
|---------|------|
| 随机裁剪 | 从原图中随机裁剪 224x224 区域 |
| 随机水平翻转 | 50% 概率左右翻转 |
| 随机旋转 | 微小角度旋转 |
| 颜色抖动 | 轻微调整亮度、对比度、饱和度 |
| 归一化 | 使用 ImageNet 均值和标准差进行标准化 |

---

## 九、常见问题解答（FAQ）

### Q1: PowerShell 提示"无法加载文件，因为未签名"？

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

运行后重试即可。

### Q2: 依赖安装失败 / 找不到版本？

**可能原因：**
- 网络问题
- 镜像源没有最新版本

**解决方案：**
1. 检查网络连接
2. 换镜像源：
   ```powershell
   pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
   ```
3. 分步安装核心包：
   ```powershell
   pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

### Q3: pip 不是内部或外部命令？

**原因**：虚拟环境未激活。

**解决**：
```powershell
.\.venv\Scripts\Activate.ps1
```

激活后重新运行 pip 命令。

### Q4: 端口被占用？

```powershell
# 查找占用 8501 端口的进程
netstat -ano | findstr :8501

# 终止该进程（将 <PID> 替换为实际进程 ID）
taskkill /PID <PID> /F
```

或直接换端口启动：
```powershell
streamlit run app.py --server.port=8502
```

### Q5: 没有 GPU 能运行吗？

**完全可以。** 代码会自动检测硬件：
- 有 NVIDIA GPU -> 使用 GPU 加速（训练和推理都更快）
- 无 GPU -> 使用 CPU（训练时间会长 3-5 倍，但功能完全正常）

检查 GPU 状态：
```powershell
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无 GPU')"
```

### Q6: Docker 构建失败？

1. 确保 Docker Desktop 已启动并正在运行
2. 检查 Docker 状态：`docker ps`
3. 重启 Docker Desktop 后重试

### Q7: 训练时准确率不上升？

- 确认数据正确：`data/kc/` 里确实是异常图片，`data/normal/` 里确实是正常图片
- 增加训练轮次：`trainer.train(epochs=50)`
- 降低学习率：在 `train.py` 中设置 `lr=1e-4` 或 `lr=5e-5`
- 增加数据量：每个类别建议至少 100 张图片

### Q8: 预测结果不准确？

- 确保使用的是 `checkpoints/best_model.pth`（最佳模型）
- 检查待预测图片是否清晰、与训练数据分布一致
- 尝试使用更大的模型或更多训练数据重新训练

### Q9: 磁盘空间不足？

清理方式：
```powershell
# 删除划分后的数据集（需要时可重新生成）
Remove-Item -Recurse -Force dataset/

# 删除旧的训练检查点（保留 best_model.pth）
Remove-Item -Recurse -Force checkpoints/
```

### Q10: 日常如何快速启动？

```powershell
# 最快方式：使用启动脚本（会自动激活虚拟环境）
.\start.ps1
```

或手动启动：
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

---

## 十、快速命令参考

### 首次完整流程（从零开始）

```powershell
# 1. 进入项目目录
cd d:\设计大赛_视觉识别

# 2. 允许脚本执行（仅首次）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. 一键配置环境
.\setup_env.ps1

# 4. 划分数据集
python prepare_data.py

# 5. 训练模型
python train.py

# 6. 启动 Web 界面
.\start.ps1
```

### 日常启动

```powershell
.\scripts\start.ps1
```

### 命令行预测

```powershell
# 单张预测
python predict.py --image data/kc/1.jpg

# 批量预测
python predict.py --dir data/kc
```

---

## 十一、技术支持

### 错误信息速查

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `ModuleNotFoundError` | 缺少依赖包 | `pip install -r requirements.txt` |
| `FileNotFoundError: 模型文件不存在` | 未训练模型 | `python train.py` |
| `CUDA out of memory` | 显存不足 | 减小 batch_size 或换小模型 |
| `Address already in use` | 端口被占用 | 换端口或关闭占用进程 |
| `Cannot connect to Docker daemon` | Docker 未启动 | 启动 Docker Desktop |
| `'pip' 不是内部或外部命令` | 虚拟环境未激活 | `.\.venv\Scripts\Activate.ps1` |
| `Access is denied` | 权限不足 | 以管理员身份运行终端 |

### 相关资源

- 模型库：[timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
- 深度学习框架：[PyTorch](https://pytorch.org/)
- Web 框架：[Streamlit](https://streamlit.io/)、[FastAPI](https://fastapi.tiangolo.com/)

---

*本项目仅供学习和研究使用。*
