"""
2026 年角膜地形图分类模型 - Streamlit Web 应用
提供用户友好的 Web 界面，支持图片上传、预测、结果可视化
"""

import streamlit as st
from model_service import get_model_service, initialize_service
from PIL import Image
import pandas as pd
import io
from pathlib import Path
from datetime import datetime

# ── 页面配置 ──
st.set_page_config(
    page_title="角膜地形图分类系统",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 设计令牌 ──
C = {
    "bg":          "#F5F3EE",       # 暖白背景
    "surface":     "#FFFFFF",       # 卡片白
    "surface_hi":  "#EDEAE4",       # 浅灰
    "border":      "#D6D2C9",       # 柔和边框
    "text":        "#2C2C2C",       # 深灰文字
    "text_dim":    "#7A7770",       # 辅助文字
    "accent":      "#588157",       # 苔藓绿主色
    "accent_hot":  "#3A5A40",       # 深绿
    "accent_bg":   "rgba(88,129,87,0.08)",
    "green":       "#588157",
    "green_bg":    "rgba(88,129,87,0.10)",
    "red":         "#BC4749",
    "red_bg":      "rgba(188,71,73,0.08)",
    "amber":       "#DDA15E",
}

# ── 全局 CSS ──
st.markdown(f"""
<style>
/* ── 基础 ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
    --bg: {C["bg"]};
    --surface: {C["surface"]};
    --surface-hi: {C["surface_hi"]};
    --border: {C["border"]};
    --text: {C["text"]};
    --text-dim: {C["text_dim"]};
    --accent: {C["accent"]};
    --green: {C["green"]};
    --red: {C["red"]};
}}

/* ── 噪点纹理背景 ── */
.stApp {{
    background-color: var(--bg) !important;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
    color: var(--text) !important;
}}

/* ── 隐藏 Streamlit 默认元素 ── */
#MainMenu, header[data-testid="stHeader"], footer {{
    visibility: hidden !important;
}}
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1200px !important;
}}

/* ── 侧边栏 ── */
section[data-testid="stSidebar"] {{
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}}
section[data-testid="stSidebar"] * {{
    color: var(--text) !important;
}}
section[data-testid="stSidebar"] .stMetricLabel {{
    color: var(--text-dim) !important;
}}
section[data-testid="stSidebar"] .stMetricValue {{
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* ── 排版 ── */
h1, h2, h3 {{
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
    font-weight: 700 !important;
}}
p, li, span, div, label {{
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}}

/* ── 卡片 ── */
.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}}
.card:hover {{
    border-color: var(--accent);
}}

/* ── 结果卡 ── */
.result-normal {{
    background: {C["green_bg"]};
    border: 1px solid {C["green"]};
    border-radius: 12px;
    padding: 2rem;
}}
.result-abnormal {{
    background: {C["red_bg"]};
    border: 1px solid {C["red"]};
    border-radius: 12px;
    padding: 2rem;
}}

/* ── 标签页 ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: var(--surface) !important;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px !important;
    color: var(--text-dim) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
}}
.stTabs [aria-selected="true"] {{
    background: var(--surface-hi) !important;
    color: var(--accent) !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: var(--accent) !important;
    height: 2px !important;
}}
.stTabs [data-baseweb="tab-content"] {{
    background: transparent !important;
    border: none !important;
}}

/* ── 按钮 ── */
.stButton > button {{
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(88, 129, 87, 0.3);
}}

/* ── 文件上传器 ── */
[data-testid="stFileUploader"] {{
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 2rem !important;
}}
[data-testid="stFileUploader"] section {{
    background: transparent !important;
    border: none !important;
}}
[data-testid="stFileUploader"] label {{
    color: var(--text-dim) !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* ── 数据表格 ── */
.dataframe {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}}
.dataframe th {{
    background: var(--surface-hi) !important;
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}}
.dataframe td {{
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* ── 进度条 ── */
.stProgress > div > div > div {{
    background: var(--accent) !important;
    border-radius: 4px !important;
}}
.stProgress > div > div {{
    background: var(--surface-hi) !important;
    border-radius: 4px !important;
}}

/* ── 指标卡 ── */
.stMetric {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}}

/* ── 信息 / 成功 / 错误 提示 ── */
.stAlert {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}}

/* ── 分割线 ── */
hr, .stDivider {{
    border-color: var(--border) !important;
    opacity: 0.5;
}}

/* ── 自定义滚动条 ── */
::-webkit-scrollbar {{
    width: 6px;
}}
::-webkit-scrollbar-track {{
    background: var(--bg);
}}
::-webkit-scrollbar-thumb {{
    background: var(--border);
    border-radius: 3px;
}}

/* ── 标题区装饰线 ── */
.hero-line {{
    width: 48px;
    height: 3px;
    background: var(--accent);
    border-radius: 2px;
    margin: 1rem 0 0.5rem 0;
}}

/* ── 动画 ── */
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.animate-in {{
    animation: fadeSlideUp 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
}}

@keyframes pulse-ring {{
    0%   {{ box-shadow: 0 0 0 0 rgba(163,177,138,0.4); }}
    70%  {{ box-shadow: 0 0 0 12px rgba(163,177,138,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(163,177,138,0); }}
}}
.pulse-ring {{
    animation: pulse-ring 2s cubic-bezier(0.34, 1.56, 0.64, 1) infinite;
}}
</style>
""", unsafe_allow_html=True)

# ── SVG 图标 ──
ICONS = {
    "eye": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "upload": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "search": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "check": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "warn": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "info": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    "download": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "bar": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    "layers": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "cpu": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "clock": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
}

def icon(name, inline=False):
    """渲染内联 SVG 图标"""
    return f'<span style="display:inline-flex;align-items:center;vertical-align:middle;gap:6px;">{ICONS[name]}</span>'


# ── 模型加载 ──
_model_load_error = None

@st.cache_resource
def load_model():
    global _model_load_error
    try:
        service = initialize_service()
        return service
    except Exception as e:
        _model_load_error = str(e)
        return None


model_service = load_model()


# ══════════════════════════════════════════
#  HEADER — 左对齐标题 + 右侧状态灯
# ══════════════════════════════════════════
col_head, col_status = st.columns([5, 1])
with col_head:
    st.markdown(f"""
    <div class="animate-in">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            {ICONS["eye"]}
            <span style="font-size:1.8rem;font-weight:700;color:{C["text"]};font-family:'DM Sans',sans-serif;letter-spacing:-0.5px;">
                角膜地形图智能分类
            </span>
        </div>
        <div class="hero-line"></div>
        <p style="color:{C["text_dim"]};font-size:0.95rem;margin-top:8px;">
            ConvNeXt V2 &middot; 近视术前风险评估
        </p>
    </div>
    """, unsafe_allow_html=True)
with col_status:
    if model_service:
        st.markdown(f"""
        <div style="text-align:right;padding-top:1.2rem;">
            <div style="display:inline-flex;align-items:center;gap:6px;
                        background:{C["green_bg"]};border:1px solid {C["green"]};
                        border-radius:20px;padding:6px 14px;font-size:0.82rem;color:{C["green"]};">
                <span style="width:6px;height:6px;border-radius:50%;background:{C["green"]};display:inline-block;" class="pulse-ring"></span>
                模型就绪
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:right;padding-top:1.2rem;">
            <div style="display:inline-flex;align-items:center;gap:6px;
                        background:{C["red_bg"]};border:1px solid {C["red"]};
                        border-radius:20px;padding:6px 14px;font-size:0.82rem;color:{C["red"]};">
                离线
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:0.8rem;">
            使用指南
        </p>
        <div style="background:{C["surface_hi"]};border-radius:10px;padding:1rem;font-size:0.88rem;line-height:1.7;">
            <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">
                <span style="color:{C["accent"]};font-family:'JetBrains Mono',monospace;font-weight:600;">01</span>
                <span>上传角膜地形图</span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">
                <span style="color:{C["accent"]};font-family:'JetBrains Mono',monospace;font-weight:600;">02</span>
                <span>点击分析</span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;">
                <span style="color:{C["accent"]};font-family:'JetBrains Mono',monospace;font-weight:600;">03</span>
                <span>查看结果与建议</span>
            </div>
        </div>
        <p style="font-size:0.78rem;color:{C["text_dim"]};margin-top:0.6rem;">
            支持 JPG / PNG，可批量上传
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="width:100%;height:1px;background:{C["border"]};margin:1rem 0;"></div>
    """, unsafe_allow_html=True)

    # 模型信息
    st.markdown(f"""
    <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:0.8rem;">
        {ICONS["layers"]} 模型信息
    </p>
    """, unsafe_allow_html=True)

    if model_service:
        info = model_service.get_model_info()
        if info['success']:
            m = info['model']
            st.markdown(f"""
            <div class="card" style="font-size:0.85rem;line-height:2.2;">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:{C["text_dim"]};">名称</span>
                    <span style="font-weight:600;color:{C["text"]};">{m["name"]}</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:{C["text_dim"]};">验证准确率</span>
                    <span style="font-weight:600;color:{C["accent"]};font-family:'JetBrains Mono',monospace;">{m["val_accuracy"]:.1f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:{C["text_dim"]};">训练轮次</span>
                    <span style="font-weight:600;color:{C["text"]};font-family:'JetBrains Mono',monospace;">{m["trained_epoch"]}</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:{C["text_dim"]};">输入尺寸</span>
                    <span style="font-weight:600;color:{C["text"]};font-family:'JetBrains Mono',monospace;">{m["input_size"]} x {m["input_size"]}</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:{C["text_dim"]};">类别数</span>
                    <span style="font-weight:600;color:{C["text"]};font-family:'JetBrains Mono',monospace;">{m["num_classes"]} (KC / Normal)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card"><p style="color:{C["red"]};">模型未加载</p></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="width:100%;height:1px;background:{C["border"]};margin:1rem 0;"></div>
    """, unsafe_allow_html=True)

    # 运行环境
    st.markdown(f"""
    <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:0.6rem;">
        {ICONS["cpu"]} 运行环境
    </p>
    <div class="card" style="font-size:0.85rem;line-height:2.2;">
        <div style="display:flex;justify-content:space-between;">
            <span style="color:{C["text_dim"]};">计算设备</span>
            <span style="font-weight:600;color:{C["text"]};">{"GPU" if model_service and "cuda" in model_service.health_check()["device"] else "CPU"}</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:{C["text_dim"]};">模型状态</span>
            <span style="font-weight:600;color:{{"#588157" if model_service else "#BC4749"}};">
                {"已加载" if model_service else "未加载"}
            </span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:{C["text_dim"]};">当前时间</span>
            <span style="font-family:'JetBrains Mono',monospace;color:{C["text_dim"]};">
                {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
#  主内容
# ══════════════════════════════════════════
if model_service is None:
    st.error("模型加载失败")
    if _model_load_error:
        st.warning(f"错误详情：{_model_load_error}")
    st.info(
        "可能的原因：\n"
        "1. Hugging Face 模型仓库访问失败（网络/权限问题）\n"
        "2. 如果仓库为 Private，请在 Streamlit Cloud → Settings → Secrets 中添加 `HF_TOKEN`\n"
        "3. 请确认模型文件 `best_model.pth` 已上传到 Hugging Face 仓库 `zzy4088/corneal-model`"
    )
    st.stop()


tab1, tab2, tab3 = st.tabs([
    f'{ICONS["search"]}  单张分析',
    f'{ICONS["layers"]}  批量分析',
    f'{ICONS["bar"]}  统计',
])


# ─── Tab 1: 单张预测 ───
with tab1:
    col_img, col_res = st.columns([2, 3], gap="large")  # 不对称比例

    with col_img:
        st.markdown(f"""
        <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;
                  letter-spacing:1.5px;font-weight:600;margin-bottom:0.6rem;">
            {ICONS["upload"]} 上传图片
        </p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "拖拽或点击上传",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

    with col_res:
        if uploaded_file is None:
            st.markdown(f"""
            <div class="card" style="display:flex;align-items:center;justify-content:center;
                        min-height:300px;text-align:center;">
                <div>
                    <p style="font-size:1.1rem;color:{C["text_dim"]};margin-bottom:0.3rem;">
                        等待上传
                    </p>
                    <p style="font-size:0.85rem;color:{C["text_dim"]};">
                        上传一张角膜地形图开始分析
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("开始分析", type="primary", use_container_width=True, key="single_btn"):
                with st.spinner("分析中..."):
                    result = model_service.predict(image)

                if result['success']:
                    is_abnormal = result['prediction'] == 'KC'
                    cls = "result-abnormal" if is_abnormal else "result-normal"
                    status_color = C["red"] if is_abnormal else C["green"]
                    status_text = "异常 — 建议进一步检查" if is_abnormal else "正常 — 符合手术条件"
                    status_icon = ICONS["warn"] if is_abnormal else ICONS["check"]

                    # 结果卡
                    st.markdown(f"""
                    <div class="{cls}" style="margin-bottom:1.2rem;">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.8rem;">
                            {status_icon}
                            <span style="font-size:1.4rem;font-weight:700;color:{status_color};font-family:'DM Sans',sans-serif;">
                                {result['class_name']}
                            </span>
                        </div>
                        <p style="font-size:0.92rem;color:{status_color};font-weight:500;">
                            {status_text}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 置信度（不对称小卡）
                    col_conf, col_prob = st.columns([1, 2], gap="medium")
                    with col_conf:
                        st.markdown(f"""
                        <div class="card" style="text-align:center;">
                            <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;
                                      letter-spacing:1px;margin-bottom:0.4rem;">置信度</p>
                            <p style="font-size:2rem;font-weight:700;color:{C["accent_hot"]};
                                      font-family:'JetBrains Mono',monospace;">
                                {result['confidence']*100:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_prob:
                        # 概率分布
                        st.markdown(f"""
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;
                                  letter-spacing:1px;margin-bottom:0.5rem;">概率分布</p>
                        """, unsafe_allow_html=True)
                        for cls_name, prob in result['probabilities'].items():
                            bar_w = prob * 100
                            bar_color = C["green"] if "正常" in cls_name else C["amber"]
                            st.markdown(f"""
                            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                                <span style="font-size:0.85rem;width:120px;color:{C["text"]};flex-shrink:0;">{cls_name}</span>
                                <div style="flex:1;height:8px;background:{C["surface_hi"]};border-radius:4px;overflow:hidden;">
                                    <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:4px;
                                                transition:width 0.6s cubic-bezier(0.34,1.56,0.64,1);"></div>
                                </div>
                                <span style="font-size:0.82rem;width:50px;text-align:right;
                                          font-family:'JetBrains Mono',monospace;color:{C["text_dim"]};">
                                    {prob*100:.1f}%
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(f"分析失败：{result.get('error', '未知错误')}")


# ─── Tab 2: 批量预测 ───
with tab2:
    uploaded_files = st.file_uploader(
        "选择多张角膜地形图",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_uploader",
    )

    if not uploaded_files:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:3rem;">
            <p style="color:{C["text_dim"]};">拖拽多张图片到此区域，或点击选择文件</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
            {ICONS["layers"]}
            <span style="font-size:0.92rem;color:{C["text_dim"]};">
                已选择 <strong style="color:{C["text"]};">{len(uploaded_files)}</strong> 张图片
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("开始批量分析", type="primary", use_container_width=True, key="batch_btn"):
            results = []
            progress_bar = st.progress(0, text="分析中...")

            for i, f in enumerate(uploaded_files):
                img = Image.open(f)
                r = model_service.predict(img)
                r['filename'] = f.name
                results.append(r)
                progress_bar.progress(
                    (i + 1) / len(uploaded_files),
                    text=f"{i+1}/{len(uploaded_files)}"
                )

            progress_bar.progress(1.0, text="完成")

            # 统计行 — 不对称
            kc_count = sum(1 for r in results if r.get('prediction') == 'KC')
            normal_count = len(results) - kc_count
            c1, c2, c3 = st.columns([1, 1, 2], gap="medium")
            with c1:
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;
                              letter-spacing:1px;">总计</p>
                    <p style="font-size:2rem;font-weight:700;color:{C["accent"]};
                              font-family:'JetBrains Mono',monospace;">{len(results)}</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;
                              letter-spacing:1px;">异常</p>
                    <p style="font-size:2rem;font-weight:700;color:{C["red"]};
                              font-family:'JetBrains Mono',monospace;">{kc_count}</p>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;
                              letter-spacing:1px;">正常</p>
                    <p style="font-size:2rem;font-weight:700;color:{C["green"]};
                              font-family:'JetBrains Mono',monospace;">{normal_count}</p>
                </div>
                """, unsafe_allow_html=True)

            # 结果表
            st.markdown(f"""
            <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;
                      letter-spacing:1.5px;font-weight:600;margin:1.5rem 0 0.6rem;">
                详细结果
            </p>
            """, unsafe_allow_html=True)

            result_df = pd.DataFrame(results)
            display_df = pd.DataFrame({
                '文件名': result_df['filename'],
                '结果':   result_df['class_name'],
                '置信度': (result_df['confidence'] * 100).round(1).astype(str) + '%',
                '建议':   result_df['suggestion'],
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载 CSV",
                data=csv,
                file_name=f"角膜分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ─── Tab 3: 关于 / 统计 ───
with tab3:
    col_about, col_tech = st.columns([3, 2], gap="large")  # 不对称

    with col_about:
        st.markdown(f"""
        <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;
                  letter-spacing:1.5px;font-weight:600;margin-bottom:1rem;">
            关于系统
        </p>
        <div class="card">
            <h3 style="font-size:1.15rem;margin-bottom:0.6rem;">角膜地形图智能分类系统</h3>
            <p style="color:{C["text_dim"]};font-size:0.9rem;line-height:1.7;">
                本系统采用 ConvNeXt V2 深度学习模型，自动识别角膜地形图中的圆锥角膜病变。
                为近视手术术前筛查提供客观、高效的辅助判断。
            </p>
            <div class="hero-line" style="margin-top:1rem;"></div>
            <p style="color:{C["text_dim"]};font-size:0.78rem;margin-top:0.8rem;">
                2026 设计大赛参赛作品 &middot; 仅供医疗辅助参考
            </p>
        </div>
    """, unsafe_allow_html=True)

    with col_tech:
        st.markdown(f"""
        <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;
                  letter-spacing:1.5px;font-weight:600;margin-bottom:1rem;">
            技术栈
        </p>
        <div class="card" style="font-size:0.88rem;line-height:2;">
            <div><span style="color:{C["accent"]};">模型</span> &nbsp; ConvNeXt V2 Base</div>
            <div><span style="color:{C["accent"]};">框架</span> &nbsp; PyTorch 2.x + timm</div>
            <div><span style="color:{C["accent"]};">前端</span> &nbsp; Streamlit</div>
            <div><span style="color:{C["accent"]};">API</span> &nbsp;&nbsp; FastAPI</div>
            <div><span style="color:{C["accent"]};">部署</span> &nbsp; Docker</div>
        </div>
    """, unsafe_allow_html=True)


# ── 页脚 ──
st.markdown(f"""
<div style="width:100%;height:1px;background:{C["border"]};margin:3rem 0 1.2rem;"></div>
<p style="text-align:center;font-size:0.78rem;color:{C["text_dim"]};">
    &copy; 2026 角膜地形图智能分类系统 &middot; 仅供医疗辅助参考，不作为诊断依据
</p>
""", unsafe_allow_html=True)
