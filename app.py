"""
角膜地形图智能诊断系统 - Streamlit Web 应用 (v2.0 增强版)
四分类诊断 + 可解释性分析 + 风险评估 + 病例管理
"""

import streamlit as st
from model_service import get_model_service, initialize_service
from risk_assessment import RiskAssessmentReport
from case_manager import CaseManager
from PIL import Image
import pandas as pd
import io
import base64
from pathlib import Path
from datetime import datetime

# ── 页面配置 ──
st.set_page_config(
    page_title="角膜地形图智能诊断系统",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 设计令牌 ──
C = {
    "bg": "#F5F3EE",
    "surface": "#FFFFFF",
    "surface_hi": "#EDEAE4",
    "border": "#D6D2C9",
    "text": "#2C2C2C",
    "text_dim": "#7A7770",
    "accent": "#588157",
    "accent_hot": "#3A5A40",
    "green": "#588157",
    "green_bg": "rgba(88,129,87,0.10)",
    "red": "#BC4749",
    "red_bg": "rgba(188,71,73,0.08)",
    "amber": "#DDA15E",
    "orange": "#E07A3A",
}

# ── 全局 CSS ──
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {{ --bg:{C["bg"]};--surface:{C["surface"]};--border:{C["border"]};--text:{C["text"]};--text-dim:{C["text_dim"]};--accent:{C["accent"]};--green:{C["green"]};--red:{C["red"]}; }}
.stApp {{ background-color:var(--bg)!important; color:var(--text)!important; }}
#MainMenu, header[data-testid="stHeader"], footer {{ visibility:hidden!important; }}
.block-container {{ padding-top:1.5rem!important; padding-bottom:3rem!important; max-width:1280px!important; }}
section[data-testid="stSidebar"] {{ background-color:var(--surface)!important; border-right:1px solid var(--border)!important; }}
section[data-testid="stSidebar"] * {{ color:var(--text)!important; }}
section[data-testid="stSidebar"] .stMetricValue {{ color:var(--accent)!important; font-family:'JetBrains Mono',monospace!important; }}
h1,h2,h3 {{ font-family:'DM Sans',sans-serif!important; color:var(--text)!important; font-weight:700!important; }}
p,li,span,div,label {{ font-family:'DM Sans',sans-serif!important; color:var(--text)!important; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.25rem; }}
.card:hover {{ border-color:var(--accent); }}
.result-normal {{ background:{C["green_bg"]}; border:1px solid {C["green"]}; border-radius:12px; padding:1.5rem; }}
.result-mild {{ background:rgba(221,161,94,0.10); border:1px solid {C["amber"]}; border-radius:12px; padding:1.5rem; }}
.result-moderate {{ background:rgba(224,122,58,0.10); border:1px solid {C["orange"]}; border-radius:12px; padding:1.5rem; }}
.result-severe {{ background:{C["red_bg"]}; border:1px solid {C["red"]}; border-radius:12px; padding:1.5rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap:0; background:var(--surface)!important; border-radius:8px; padding:4px; border:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ border-radius:6px!important; color:var(--text-dim)!important; font-family:'DM Sans',sans-serif!important; font-weight:500!important; padding:0.5rem 1.2rem!important; }}
.stTabs [aria-selected="true"] {{ background:var(--surface-hi)!important; color:var(--accent)!important; }}
.stTabs [data-baseweb="tab-content"] {{ background:transparent!important; border:none!important; }}
.stButton > button {{ background:var(--accent)!important; color:#FFFFFF!important; border:none!important; border-radius:8px!important; font-family:'DM Sans',sans-serif!important; font-weight:600!important; padding:0.75rem 1.5rem!important; }}
.stButton > button:hover {{ transform:translateY(-2px); box-shadow:0 6px 20px rgba(88,129,87,0.3); }}
[data-testid="stFileUploader"] {{ background:var(--surface)!important; border:1px dashed var(--border)!important; border-radius:12px!important; padding:2rem!important; }}
[data-testid="stFileUploader"] section {{ background:transparent!important; border:none!important; }}
.stProgress > div > div > div {{ background:var(--accent)!important; border-radius:4px!important; }}
.stMetric {{ background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:10px!important; }}
.dataframe {{ background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:8px!important; }}
.dataframe th {{ background:var(--surface-hi)!important; color:var(--accent)!important; font-size:0.8rem!important; }}
.stAlert {{ background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:8px!important; }}
::-webkit-scrollbar {{ width:6px; }}
::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}
.hero-line {{ width:48px; height:3px; background:var(--accent); border-radius:2px; margin:0.8rem 0 0.4rem; }}
@keyframes fadeSlideUp {{ from {{ opacity:0; transform:translateY(16px); }} to {{ opacity:1; transform:translateY(0); }} }}
.animate-in {{ animation:fadeSlideUp 0.5s ease forwards; }}
.indicator-row {{ display:flex; align-items:center; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid var(--border); }}
.indicator-row:last-child {{ border-bottom:none; }}
.step-card {{ display:flex; gap:0.75rem; align-items:flex-start; padding:0.6rem 0; }}
.step-num {{ width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.78rem; font-weight:700; flex-shrink:0; }}
</style>
""", unsafe_allow_html=True)

# ── SVG 图标 ──
ICONS = {
    "eye": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "upload": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "search": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "check": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>',
    "warn": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "layers": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "bar": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    "brain": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2a4 4 0 014 4c0 .73-.2 1.41-.54 2H18a4 4 0 010 8h-1.46A4 4 0 0112 20a4 4 0 01-4.54-4H6a4 4 0 010-8h2.54A4 4 0 0112 2z"/></svg>',
    "clipboard": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/></svg>',
}

# ── 模型加载 ──
@st.cache_resource
def load_model():
    try:
        return initialize_service()
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

@st.cache_resource
def load_case_manager():
    try:
        return CaseManager()
    except Exception:
        return None

model_service = load_model()
case_manager = load_case_manager()

# ── 辅助函数 ──

def get_severity_class(prediction: str) -> str:
    if prediction == 'Normal':
        return 'result-normal'
    elif prediction == 'Mild KC':
        return 'result-mild'
    elif prediction == 'Moderate KC':
        return 'result-moderate'
    else:
        return 'result-severe'

def get_severity_color(prediction: str) -> str:
    colors = {'Normal': C['green'], 'Mild KC': C['amber'], 'Moderate KC': C['orange'], 'Severe KC': C['red']}
    return colors.get(prediction, C['red'])

def get_severity_text(prediction: str) -> str:
    texts = {'Normal': '正常 - 符合手术条件', 'Mild KC': '轻度异常 - 需谨慎评估',
             'Moderate KC': '中度圆锥角膜 - 不建议激光手术', 'Severe KC': '重度圆锥角膜 - 需治疗干预'}
    return texts.get(prediction, '')

def render_indicator_table(indicators):
    """渲染临床指标对比表"""
    if not indicators:
        return
    html = '<div class="card" style="padding:0.8rem 1.2rem;">'
    for ind in indicators:
        mark = "✅" if not ind['abnormal'] else "❌"
        val_color = C['green'] if not ind['abnormal'] else C['red']
        html += f'''<div class="indicator-row">
            <span style="font-size:0.88rem;">{mark} {ind['name']}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;">
                <span style="color:{val_color};font-weight:600;">{ind['value']} {ind['unit']}</span>
                <span style="color:{C['text_dim']};font-size:0.78rem;margin-left:6px;">(正常 {ind['normal_range']})</span>
            </span>
        </div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_decision_path(decision_path):
    """渲染决策路径"""
    if not decision_path:
        return
    steps = decision_path.get('steps', [])
    html = '<div class="card">'
    for step in steps:
        s = step['step']
        is_normal = step['result'] == '正常'
        bg_color = C['green_bg'] if is_normal else C['red_bg']
        text_color = C['green'] if is_normal else C['red']
        arrow = f'<div style="text-align:center;color:{C["text_dim"]};font-size:0.85rem;margin:0.3rem 0;">↓</div>'
        html += f'''
        <div class="step-card">
            <div class="step-num" style="background:{bg_color};color:{text_color};">Step {s}</div>
            <div style="flex:1;">
                <div style="font-size:0.85rem;font-weight:600;">{step['feature']}</div>
                <div style="font-size:0.82rem;color:{C['text_dim']};margin-top:2px;">
                    标准: {step['threshold']} | 实际: <span style="color:{text_color};font-weight:600;">{step['actual']}</span> → <span style="color:{text_color};">{step['result']}</span>
                </div>
                <div style="font-size:0.75rem;color:{C['text_dim']};">贡献度: {step["contribution"]:.0%}</div>
            </div>
        </div>
        {arrow if s < len(steps) else ''}'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_probability_bars(probabilities):
    """渲染概率分布柱状图"""
    for cls_name, prob in probabilities.items():
        bar_w = prob * 100
        if '正常' in cls_name:
            bar_color = C['green']
        elif '轻度' in cls_name:
            bar_color = C['amber']
        elif '中度' in cls_name:
            bar_color = C['orange']
        else:
            bar_color = C['red']
        st.markdown(f'''
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <span style="font-size:0.85rem;width:140px;flex-shrink:0;">{cls_name}</span>
            <div style="flex:1;height:8px;background:{C['surface_hi']};border-radius:4px;overflow:hidden;">
                <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:4px;"></div>
            </div>
            <span style="font-size:0.82rem;width:50px;text-align:right;font-family:'JetBrains Mono',monospace;color:{C['text_dim']};">{prob*100:.1f}%</span>
        </div>''', unsafe_allow_html=True)


# ══════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════
col_head, col_status = st.columns([5, 1])
with col_head:
    st.markdown(f'''
    <div class="animate-in">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            {ICONS["eye"]}
            <span style="font-size:1.8rem;font-weight:700;color:{C["text"]};font-family:'DM Sans',sans-serif;letter-spacing:-0.5px;">
                角膜地形图智能诊断
            </span>
        </div>
        <div class="hero-line"></div>
        <p style="color:{C["text_dim"]};font-size:0.95rem;margin-top:8px;">
            ConvNeXt V2 &middot; 四分类精细诊断 &middot; 可解释性分析
        </p>
    </div>''', unsafe_allow_html=True)
with col_status:
    if model_service:
        num_cls = model_service.num_classes
        st.markdown(f'''
        <div style="text-align:right;padding-top:1.2rem;">
            <div style="display:inline-flex;align-items:center;gap:6px;
                        background:{C["green_bg"]};border:1px solid {C["green"]};
                        border-radius:20px;padding:6px 14px;font-size:0.82rem;color:{C["green"]};">
                <span style="width:6px;height:6px;border-radius:50%;background:{C["green"]};display:inline-block;"></span>
                模型就绪 ({num_cls}分类)
            </div>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''<div style="text-align:right;padding-top:1.2rem;">
            <div style="display:inline-flex;align-items:center;gap:6px;background:{C["red_bg"]};border:1px solid {C["red"]};border-radius:20px;padding:6px 14px;font-size:0.82rem;color:{C["red"]};">离线</div>
        </div>''', unsafe_allow_html=True)


# ══════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown(f'''
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
            <span>查看诊断结果与判断依据</span>
        </div>
    </div>
    <p style="font-size:0.78rem;color:{C["text_dim"]};margin-top:0.6rem;">支持 JPG / PNG，可批量上传</p>
    ''', unsafe_allow_html=True)

    st.markdown(f'<div style="width:100%;height:1px;background:{C["border"]};margin:1rem 0;"></div>', unsafe_allow_html=True)

    if model_service:
        info = model_service.get_model_info()
        if info['success']:
            m = info['model']
            cls_info = ", ".join(info.get('class_names', {}).values())
            st.markdown(f'''
            <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:0.8rem;">
                {ICONS["layers"]} 模型信息
            </p>
            <div class="card" style="font-size:0.85rem;line-height:2.2;">
                <div style="display:flex;justify-content:space-between;"><span style="color:{C["text_dim"]};">名称</span><span style="font-weight:600;">{m["name"]}</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:{C["text_dim"]};">模式</span><span style="font-weight:600;color:{C["accent"]};">{m["mode"]}</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:{C["text_dim"]};">验证准确率</span><span style="font-weight:600;color:{C["accent"]};font-family:'JetBrains Mono',monospace;">{m["val_accuracy"]:.1f}%</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:{C["text_dim"]};">类别</span><span style="font-weight:600;">{cls_info}</span></div>
            </div>''', unsafe_allow_html=True)

    st.markdown(f'<div style="width:100%;height:1px;background:{C["border"]};margin:1rem 0;"></div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="card" style="font-size:0.85rem;line-height:2.2;">
        <div style="display:flex;justify-content:space-between;">
            <span style="color:{C["text_dim"]};">计算设备</span>
            <span style="font-weight:600;">{"GPU" if model_service and "cuda" in model_service.health_check()["device"] else "CPU"}</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:{C["text_dim"]};">当前时间</span>
            <span style="font-family:'JetBrains Mono',monospace;color:{C["text_dim"]};">{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
    </div>''', unsafe_allow_html=True)


# ══════════════════════════════════════════
#  主内容
# ══════════════════════════════════════════
if model_service is None:
    st.error("模型加载失败，请检查 checkpoints 目录下是否有模型文件")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    f'{ICONS["search"]}  单张分析',
    f'{ICONS["layers"]}  批量分析',
    f'{ICONS["brain"]}  判断依据',
    f'{ICONS["clipboard"]}  病例管理',
])


# ─── Tab 1: 单张预测 ───
with tab1:
    col_img, col_res = st.columns([2, 3], gap="large")

    with col_img:
        st.markdown(f'''
        <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:0.6rem;">
            {ICONS["upload"]} 上传图片
        </p>''', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("拖拽或点击上传", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

    with col_res:
        if uploaded_file is None:
            st.markdown(f'''
            <div class="card" style="display:flex;align-items:center;justify-content:center;min-height:300px;text-align:center;">
                <div>
                    <p style="font-size:1.1rem;color:{C["text_dim"]};margin-bottom:0.3rem;">等待上传</p>
                    <p style="font-size:0.85rem;color:{C["text_dim"]};">上传一张角膜地形图开始分析</p>
                </div>
            </div>''', unsafe_allow_html=True)
        else:
            if st.button("开始分析", type="primary", use_container_width=True, key="single_btn"):
                with st.spinner("AI 分析中..."):
                    result = model_service.predict(image, enable_explainability=True)

                if result['success']:
                    pred = result['prediction']
                    cls_result = get_severity_class(pred)
                    color = get_severity_color(pred)
                    text = get_severity_text(pred)

                    # 结果卡片
                    st.markdown(f'''
                    <div class="{cls_result}" style="margin-bottom:1rem;">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.6rem;">
                            {ICONS["check"] if pred == 'Normal' else ICONS["warn"]}
                            <span style="font-size:1.4rem;font-weight:700;color:{color};">{result['class_name']}</span>
                        </div>
                        <p style="font-size:0.92rem;color:{color};font-weight:500;">{text}</p>
                    </div>''', unsafe_allow_html=True)

                    col_conf, col_prob = st.columns([1, 2], gap="medium")
                    with col_conf:
                        st.markdown(f'''
                        <div class="card" style="text-align:center;">
                            <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem;">置信度</p>
                            <p style="font-size:2rem;font-weight:700;color:{C["accent_hot"]};font-family:'JetBrains Mono',monospace;">{result["confidence"]*100:.1f}%</p>
                        </div>''', unsafe_allow_html=True)

                    with col_prob:
                        st.markdown(f'''
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">概率分布</p>''', unsafe_allow_html=True)
                        render_probability_bars(result.get('probabilities', {}))

                    # 热力图
                    overlay_bytes = result.get('heatmap_overlay_bytes')
                    if overlay_bytes:
                        st.markdown(f'''
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-top:1.2rem;margin-bottom:0.5rem;">
                            Grad-CAM 热力图（模型关注区域）
                        </p>''', unsafe_allow_html=True)
                        st.image(overlay_bytes, use_container_width=True, caption="红色=高关注度区域，蓝色=低关注度区域")

                    # 风险评估摘要
                    risk = result.get('risk_report')
                    if risk:
                        ra = risk.get('risk_assessment', {})
                        sr = ra.get('surgery_risk', {})
                        fu = ra.get('follow_up', {})

                        st.markdown(f'''
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-top:1rem;margin-bottom:0.5rem;">
                            风险评估
                        </p>''', unsafe_allow_html=True)

                        c1, c2 = st.columns(2, gap="medium")
                        with c1:
                            sr_color = C['green'] if '低' in sr.get('level', '') else C['red']
                            st.markdown(f'''
                            <div class="card" style="text-align:center;">
                                <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">手术风险</p>
                                <p style="font-size:1.3rem;font-weight:700;color:{sr_color};font-family:'JetBrains Mono',monospace;">{sr.get("level", "--")}</p>
                                <p style="font-size:0.78rem;color:{C["text_dim"]};margin-top:4px;">{sr.get("description", "")}</p>
                            </div>''', unsafe_allow_html=True)
                        with c2:
                            st.markdown(f'''
                            <div class="card" style="text-align:center;">
                                <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">随访建议</p>
                                <p style="font-size:1.3rem;font-weight:700;color:{C["accent"]};font-family:'JetBrains Mono',monospace;">{fu.get("interval", "--")}</p>
                                <p style="font-size:0.78rem;color:{C["text_dim"]};margin-top:4px;">{fu.get("description", "")}</p>
                            </div>''', unsafe_allow_html=True)
                else:
                    st.error(f"分析失败：{result.get('error', '未知错误')}")


# ─── Tab 2: 批量预测 ───
with tab2:
    uploaded_files = st.file_uploader("选择多张角膜地形图", type=['jpg', 'jpeg', 'png'],
                                      accept_multiple_files=True, label_visibility="collapsed", key="batch_uploader")

    if not uploaded_files:
        st.markdown(f'''
        <div class="card" style="text-align:center;padding:3rem;">
            <p style="color:{C["text_dim"]};">拖拽多张图片到此区域</p>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
            {ICONS["layers"]}
            <span style="font-size:0.92rem;color:{C["text_dim"]};">已选择 <strong>{len(uploaded_files)}</strong> 张图片</span>
        </div>''', unsafe_allow_html=True)

        if st.button("开始批量分析", type="primary", use_container_width=True, key="batch_btn"):
            results = []
            progress_bar = st.progress(0, text="分析中...")

            for i, f in enumerate(uploaded_files):
                img = Image.open(f)
                r = model_service.predict(img, enable_explainability=False)
                r['filename'] = f.name
                results.append(r)
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"{i+1}/{len(uploaded_files)}")

            progress_bar.progress(1.0, text="完成")

            # 统计
            dist = {}
            for r in results:
                cls = r.get('class_name', '未知')
                dist[cls] = dist.get(cls, 0) + 1

            cols = st.columns(len(dist), gap="small")
            for idx, (cls_name, count) in enumerate(dist.items()):
                with cols[idx]:
                    pred_key = r.get('prediction', 'Normal')
                    clr = get_severity_color(pred_key) if cls_name == r.get('class_name') else C['accent']
                    st.markdown(f'''
                    <div class="card" style="text-align:center;">
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">{cls_name[:4]}</p>
                        <p style="font-size:2rem;font-weight:700;color:{clr};font-family:'JetBrains Mono',monospace;">{count}</p>
                    </div>''', unsafe_allow_html=True)

            result_df = pd.DataFrame(results)
            display_df = pd.DataFrame({
                '文件名': result_df.get('filename', []),
                '结果': result_df.get('class_name', []),
                '置信度': (result_df.get('confidence', pd.Series([0])) * 100).round(1).astype(str) + '%',
                '建议': result_df.get('suggestion', []),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label="下载 CSV", data=csv,
                               file_name=f"角膜分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv", use_container_width=True)


# ─── Tab 3: 判断依据展示 ───
with tab3:
    st.markdown(f'''
    <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:0.6rem;">
        {ICONS["brain"]} AI 可解释性分析
    </p>
    <p style="color:{C["text_dim"]};font-size:0.9rem;margin-bottom:1rem;">
        上传图片后，系统将展示 AI 的判断依据：热力图、临床指标、决策路径、风险评估。
    </p>''', unsafe_allow_html=True)

    explain_file = st.file_uploader("上传角膜地形图进行可解释性分析", type=['jpg', 'jpeg', 'png'],
                                    label_visibility="collapsed", key="explain_uploader")

    if explain_file:
        col_ex_img, col_ex_res = st.columns([2, 3], gap="large")
        with col_ex_img:
            ex_image = Image.open(explain_file)
            st.image(ex_image, use_container_width=True)

        with col_ex_res:
            if st.button("生成判断依据", type="primary", use_container_width=True, key="explain_btn"):
                with st.spinner("深度分析中..."):
                    result = model_service.predict(ex_image, enable_explainability=True)

                if result['success'] and result.get('explainability'):
                    exp = result['explainability']
                    risk = result.get('risk_report', {})

                    # 诊断结果
                    pred = result['prediction']
                    cls_result = get_severity_class(pred)
                    color = get_severity_color(pred)
                    st.markdown(f'''
                    <div class="{cls_result}" style="margin-bottom:1rem;">
                        <span style="font-size:1.3rem;font-weight:700;color:{color};">{result['class_name']}</span>
                        <span style="margin-left:12px;font-family:'JetBrains Mono',monospace;">{result["confidence"]*100:.1f}%</span>
                    </div>''', unsafe_allow_html=True)

                    # 热力图
                    overlay_bytes = result.get('heatmap_overlay_bytes')
                    if overlay_bytes:
                        st.markdown(f'''
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:1rem 0 0.5rem;">
                            Grad-CAM 热力图
                        </p>''', unsafe_allow_html=True)
                        st.image(overlay_bytes, use_container_width=True)

                    # 关键区域
                    regions = exp.get('regions', [])
                    if regions:
                        st.markdown(f'''
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:1rem 0 0.5rem;">
                            关键区域分析
                        </p>''', unsafe_allow_html=True)
                        for r in regions[:4]:
                            sev_color = C['red'] if r['severity'] == '高' else (C['amber'] if r['severity'] == '中' else C['text_dim'])
                            st.markdown(f'''
                            <div class="card" style="margin-bottom:0.4rem;padding:0.6rem 1rem;">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <span style="font-size:0.88rem;font-weight:500;">{r['region_type']}</span>
                                    <span style="font-size:0.78rem;color:{sev_color};font-weight:600;">关注度 {r["avg_attention"]:.2f}</span>
                                </div>
                                <div style="font-size:0.78rem;color:{C["text_dim"]};margin-top:2px;">
                                    面积占比 {r["area_ratio"]:.1%} | 严重度: {r["severity"]}
                                </div>
                            </div>''', unsafe_allow_html=True)

                    # 临床指标
                    indicators = exp.get('indicators', [])
                    if indicators:
                        abnormal_cnt = exp.get('abnormal_indicator_count', 0)
                        total_cnt = exp.get('total_indicators', 0)
                        st.markdown(f'''
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:1rem 0 0.5rem;">
                            临床指标对比 <span style="color:{C["red"]};">({abnormal_cnt}/{total_cnt}项异常)</span>
                        </p>''', unsafe_allow_html=True)
                        render_indicator_table(indicators)

                    # 决策路径
                    dp = exp.get('decision_path', {})
                    if dp:
                        st.markdown(f'''
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:1rem 0 0.5rem;">
                            AI 决策路径
                        </p>''', unsafe_allow_html=True)
                        render_decision_path(dp)

                        explanation = dp.get('explanation', '')
                        if explanation:
                            st.markdown(f'''
                            <div class="card" style="margin-top:0.5rem;">
                                <p style="font-size:0.88rem;color:{C["accent"]};font-weight:600;margin-bottom:4px;">综合判断</p>
                                <p style="font-size:0.85rem;color:{C["text_dim"]};">{explanation}</p>
                            </div>''', unsafe_allow_html=True)

                    # 风险评估报告
                    if risk:
                        ra = risk.get('risk_assessment', {})
                        recs = risk.get('clinical_recommendations', [])

                        if ra:
                            st.markdown(f'''
                            <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:1rem 0 0.5rem;">
                                风险评估
                            </p>''', unsafe_allow_html=True)

                            rc1, rc2, rc3 = st.columns(3, gap="small")
                            sr = ra.get('surgery_risk', {})
                            pr = ra.get('progression_risk', {})
                            fu = ra.get('follow_up', {})

                            with rc1:
                                sr_c = C['green'] if '低' in sr.get('level', '') else C['red']
                                st.markdown(f'''
                                <div class="card" style="text-align:center;padding:0.8rem;">
                                    <p style="font-size:0.68rem;color:{C["text_dim"]};text-transform:uppercase;">手术风险</p>
                                    <p style="font-size:1.1rem;font-weight:700;color:{sr_c};">{sr.get("level","--")}</p>
                                </div>''', unsafe_allow_html=True)
                            with rc2:
                                pr_c = C['green'] if pr.get('level') in ('极低',) else (C['amber'] if pr.get('level') == '中等' else C['red'])
                                st.markdown(f'''
                                <div class="card" style="text-align:center;padding:0.8rem;">
                                    <p style="font-size:0.68rem;color:{C["text_dim"]};text-transform:uppercase;">进展风险</p>
                                    <p style="font-size:1.1rem;font-weight:700;color:{pr_c};">{pr.get("level","--")}</p>
                                </div>''', unsafe_allow_html=True)
                            with rc3:
                                st.markdown(f'''
                                <div class="card" style="text-align:center;padding:0.8rem;">
                                    <p style="font-size:0.68rem;color:{C["text_dim"]};text-transform:uppercase;">随访间隔</p>
                                    <p style="font-size:1.1rem;font-weight:700;color:{C["accent"]};">{fu.get("interval","--")}</p>
                                </div>''', unsafe_allow_html=True)

                        if recs:
                            st.markdown(f'''
                            <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:1rem 0 0.5rem;">
                                临床建议
                            </p>''', unsafe_allow_html=True)
                            for i, rec in enumerate(recs, 1):
                                st.markdown(f'''
                                <div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:4px;">
                                    <span style="width:20px;height:20px;border-radius:50%;background:{C["green_bg"]};color:{C["green"]};display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;flex-shrink:0;">{i}</span>
                                    <span style="font-size:0.85rem;line-height:1.5;">{rec}</span>
                                </div>''', unsafe_allow_html=True)

                elif not result['success']:
                    st.error(f"分析失败：{result.get('error', '')}")
                else:
                    st.warning("可解释性分析结果为空，请确认已启用该功能")
    else:
        st.markdown(f'''
        <div class="card" style="text-align:center;padding:4rem;">
            <p style="font-size:1.1rem;color:{C["text_dim"]};margin-bottom:0.3rem;">上传图片查看 AI 判断依据</p>
            <p style="font-size:0.85rem;color:{C["text_dim"]};">包括热力图、临床指标、决策路径、风险评估</p>
        </div>''', unsafe_allow_html=True)


# ─── Tab 4: 病例管理 ───
with tab4:
    if case_manager is None:
        st.warning("病例管理系统未初始化")
    else:
        st.markdown(f'''
        <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:0.8rem;">
            {ICONS["clipboard"]} 病例管理
        </p>''', unsafe_allow_html=True)

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["患者列表", "新建患者", "统计分析"])

        with sub_tab1:
            keyword = st.text_input("搜索患者（姓名/ID/电话）", placeholder="输入关键词搜索...")
            if keyword:
                patients = case_manager.search_patients(keyword)
            else:
                patients = case_manager.search_patients()

            if patients:
                patient_df = pd.DataFrame(patients)
                display_cols = ['patient_id', 'name', 'age', 'gender', 'created_at']
                existing_cols = [c for c in display_cols if c in patient_df.columns]
                rename_map = {'patient_id': 'ID', 'name': '姓名', 'age': '年龄', 'gender': '性别', 'created_at': '创建时间'}
                display_df = patient_df[existing_cols].rename(columns=rename_map)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                selected_id = st.selectbox("选择患者查看详情", [p['patient_id'] for p in patients])
                if selected_id:
                    exams = case_manager.get_patient_examinations(selected_id)
                    if exams:
                        st.markdown(f'''
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:1rem 0 0.5rem;">
                            检查记录
                        </p>''', unsafe_allow_html=True)
                        exam_df = pd.DataFrame(exams)
                        exam_display = exam_df[['exam_date', 'prediction', 'class_name', 'confidence', 'risk_level']].rename(
                            columns={'exam_date': '日期', 'prediction': '分类', 'class_name': '名称', 'confidence': '置信度', 'risk_level': '风险等级'})
                        st.dataframe(exam_display, use_container_width=True, hide_index=True)
            else:
                st.info("暂无患者记录")

        with sub_tab2:
            with st.form("new_patient_form"):
                name = st.text_input("姓名 *", required=True)
                age = st.number_input("年龄", min_value=0, max_value=150, value=30)
                gender = st.selectbox("性别", ["男", "女", "其他"])
                phone = st.text_input("电话")
                notes = st.text_area("备注")
                submitted = st.form_submit_button("创建患者", type="primary")

                if submitted and name:
                    patient = case_manager.add_patient(name=name, age=int(age), gender=gender, phone=phone, notes=notes)
                    st.success(f"患者创建成功: {patient['patient_id']}")
                    st.json(patient)

        with sub_tab3:
            stats = case_manager.get_statistics()
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.markdown(f'''
                <div class="card" style="text-align:center;">
                    <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">总患者数</p>
                    <p style="font-size:2rem;font-weight:700;color:{C["accent"]};font-family:'JetBrains Mono',monospace;">{stats["total_patients"]}</p>
                </div>''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''
                <div class="card" style="text-align:center;">
                    <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">总检查数</p>
                    <p style="font-size:2rem;font-weight:700;color:{C["accent"]};font-family:'JetBrains Mono',monospace;">{stats["total_examinations"]}</p>
                </div>''', unsafe_allow_html=True)
            with c3:
                st.markdown(f'''
                <div class="card" style="text-align:center;">
                    <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">待随访</p>
                    <p style="font-size:2rem;font-weight:700;color:{C["amber"]};font-family:'JetBrains Mono',monospace;">{stats["pending_follow_ups"]}</p>
                </div>''', unsafe_allow_html=True)

            dist = stats.get('prediction_distribution', {})
            if dist:
                st.markdown(f'''
                <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:1rem 0 0.5rem;">
                    诊断分布
                </p>''', unsafe_allow_html=True)
                for pred, count in dist.items():
                    st.markdown(f'''
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                        <span style="font-size:0.85rem;width:140px;">{pred}</span>
                        <div style="flex:1;height:8px;background:{C["surface_hi"]};border-radius:4px;overflow:hidden;">
                            <div style="width:{count/max(dist.values())*100:.0f}%;height:100%;background:{C["accent"]};border-radius:4px;"></div>
                        </div>
                        <span style="font-size:0.82rem;width:50px;text-align:right;font-family:'JetBrains Mono',monospace;">{count}</span>
                    </div>''', unsafe_allow_html=True)


# ── 页脚 ──
st.markdown(f'''
<div style="width:100%;height:1px;background:{C["border"]};margin:3rem 0 1rem;"></div>
<p style="text-align:center;font-size:0.78rem;color:{C["text_dim"]};">
    &copy; 2026 角膜地形图智能诊断系统 v2.0 &middot; 仅供医疗辅助参考，不作为诊断依据
</p>''', unsafe_allow_html=True)
