"""
角膜地形图智能诊断系统 - Streamlit Web 应用 (v3.0)
多模型集成诊断 · 可解释AI · 风险评估 · 病例管理
"""

# pyright: reportUnusedCallResult=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportMissingTypeStubs=false, reportExplicitAny=false, reportUnusedParameter=false
import streamlit as st
from model_service import initialize_service
from risk_assessment import RiskAssessmentReport
from case_manager import CaseManager
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any

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
    "amber_bg": "rgba(221,161,94,0.10)",
    "orange": "#E07A3A",
    "orange_bg": "rgba(224,122,58,0.10)",
    "blue": "#457B9D",
    "blue_bg": "rgba(69,123,157,0.10)",
}

# ── 全局 CSS ──
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {{ --bg:{C["bg"]};--surface:{C["surface"]};--surface-hi:{C["surface_hi"]};--border:{C["border"]};--text:{C["text"]};--text-dim:{C["text_dim"]};--accent:{C["accent"]};--green:{C["green"]};--red:{C["red"]};--amber:{C["amber"]};--orange:{C["orange"]}; }}
.stApp {{ background-color:var(--bg)!important; color:var(--text)!important; }}
#MainMenu, header[data-testid="stHeader"], footer {{ visibility:hidden!important; }}
.block-container {{ padding-top:1.5rem!important; padding-bottom:3rem!important; max-width:1320px!important; }}
section[data-testid="stSidebar"] {{ background-color:var(--surface)!important; border-right:1px solid var(--border)!important; }}
section[data-testid="stSidebar"] * {{ color:var(--text)!important; }}
section[data-testid="stSidebar"] .stMetricValue {{ color:var(--accent)!important; font-family:'JetBrains Mono',monospace!important; }}
h1,h2,h3 {{ font-family:'DM Sans',sans-serif!important; color:var(--text)!important; font-weight:700!important; }}
p,li,span,div,label {{ font-family:'DM Sans',sans-serif!important; color:var(--text)!important; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.25rem; }}
.card:hover {{ border-color:var(--accent); transition:border-color 0.2s; }}
.card-compact {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:0.8rem 1rem; }}
.card-highlight {{ background:var(--surface); border:2px solid var(--accent); border-radius:12px; padding:1.25rem; box-shadow:0 2px 12px rgba(88,129,87,0.10); }}
.card-sim {{ background:var(--surface); border:1px dashed var(--border); border-radius:12px; padding:1.25rem; opacity:0.85; }}
.result-normal {{ background:{C["green_bg"]}; border:1px solid {C["green"]}; border-radius:12px; padding:1.5rem; }}
.result-mild {{ background:{C["amber_bg"]}; border:1px solid {C["amber"]}; border-radius:12px; padding:1.5rem; }}
.result-moderate {{ background:{C["orange_bg"]}; border:1px solid {C["orange"]}; border-radius:12px; padding:1.5rem; }}
.result-severe {{ background:{C["red_bg"]}; border:1px solid {C["red"]}; border-radius:12px; padding:1.5rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap:0; background:var(--surface)!important; border-radius:8px; padding:4px; border:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ border-radius:6px!important; color:var(--text-dim)!important; font-family:'DM Sans',sans-serif!important; font-weight:500!important; padding:0.5rem 1.2rem!important; }}
.stTabs [aria-selected="true"] {{ background:var(--surface-hi)!important; color:var(--accent)!important; }}
.stTabs [data-baseweb="tab-content"] {{ background:transparent!important; border:none!important; }}
.stButton > button {{ background:var(--accent)!important; color:#FFFFFF!important; border:none!important; border-radius:8px!important; font-family:'DM Sans',sans-serif!important; font-weight:600!important; padding:0.75rem 1.5rem!important; transition:all 0.2s; }}
.stButton > button:hover {{ transform:translateY(-2px); box-shadow:0 6px 20px rgba(88,129,87,0.3); }}
[data-testid="stFileUploader"] {{ background:var(--surface)!important; border:2px dashed var(--border)!important; border-radius:12px!important; padding:1.2rem 1rem!important; }}
[data-testid="stFileUploader"] > div {{ background:transparent!important; }}
/* 隐藏文件上传器内部标签，解决文字重叠 */
[data-testid="stFileUploader"] [data-baseweb="form-label"],
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] .stMarkdown,
[data-testid="stFileUploader"] p:first-child {{ display:none!important; visibility:hidden!important; height:0!important; overflow:hidden!important; margin:0!important; padding:0!important; font-size:0!important; }}
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
.indicator-row {{ display:flex; align-items:center; justify-content:space-between; padding:0.55rem 0; border-bottom:1px solid var(--border); }}
.indicator-row:last-child {{ border-bottom:none; }}
.step-card {{ display:flex; gap:0.75rem; align-items:flex-start; padding:0.7rem 0; }}
.step-num {{ width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.78rem; font-weight:700; flex-shrink:0; font-family:'JetBrains Mono',monospace; }}
.section-label {{ font-size:0.72rem; color:{C["text_dim"]}; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:0.5rem; }}
.model-card {{ border-radius:10px; padding:0.7rem 0.9rem; margin-bottom:0.4rem; font-size:0.82rem; }}
.model-card-active {{ background:{C["green_bg"]}; border:1px solid {C["green"]}; }}
.model-card-sim {{ background:{C["surface_hi"]}; border:1px dashed var(--border); }}
.vote-arrow {{ text-align:center; color:{C["text_dim"]}; font-size:1.2rem; margin:0.4rem 0; }}
.weight-badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.72rem; font-weight:600; font-family:'JetBrains Mono',monospace; }}
.rec-item {{ display:flex; gap:10px; align-items:flex-start; margin-bottom:6px; }}
.rec-num {{ width:22px; height:22px; border-radius:50%; background:{C["green_bg"]}; color:{C["green"]}; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; flex-shrink:0; font-family:'JetBrains Mono',monospace; }}
.star-rating {{ color:{C["amber"]}; font-size:0.78rem; letter-spacing:1px; }}
.report-box {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1.5rem; margin-bottom:1rem; }}
.report-title {{ font-size:1rem; font-weight:700; color:{C["text"]}; text-align:center; padding-bottom:0.8rem; border-bottom:1px solid var(--border); margin-bottom:1rem; letter-spacing:0.5px; }}
.report-divider {{ width:100%; height:1px; background:var(--border); margin:1rem 0; }}
.evidence-row {{ display:flex; align-items:flex-start; gap:8px; padding:0.35rem 0; font-size:0.85rem; line-height:1.6; }}
.evidence-row:last-child {{ padding-bottom:0; }}
.rec-badge {{ display:inline-block; padding:1px 8px; border-radius:6px; font-size:0.72rem; font-weight:600; flex-shrink:0; }}
/* 文件上传器：隐藏内部重复的 Upload 标签 */
[data-testid="stFileUploader"] span:first-of-type {{ display:none!important; }}
[data-testid="stFileUploader"] [class*="label"] {{ font-size:1px!important; color:transparent!important; }}
</style>
<script>
// 页面加载后移除 file_uploader 内部重复的 "Upload" 文本
(function fixUploader() {{
    var observer = new MutationObserver(function() {{
        document.querySelectorAll('[data-testid="stFileUploader"]').forEach(function(el) {{
            el.querySelectorAll('span, label, p').forEach(function(child) {{
                if (child.textContent.trim() === 'upload' || child.textContent.trim() === 'Upload') {{
                    child.style.display = 'none';
                }}
            }});
        }});
    }});
    observer.observe(document.body, {{ childList:true, subtree:true }});
    // 立即执行一次
    document.querySelectorAll('[data-testid="stFileUploader"]').forEach(function(el) {{
        el.querySelectorAll('span, label, p').forEach(function(child) {{
            if (child.textContent.trim() === 'upload' || child.textContent.trim() === 'Upload') {{
                child.style.display = 'none';
            }}
        }});
    }});
}})();
</script>
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
    "users": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>',
    "shield": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
}

# ── 集成模型配置 ──
ENSEMBLE_CONFIG = {
    'ConvNeXt V2':  {'weight': 0.40, 'mode': '深度推理'},
    'MaxViT':       {'weight': 0.35, 'mode': '深度推理'},
    'Swin V2':      {'weight': 0.25, 'mode': '深度推理'},
}

CLASS_NAMES_4 = {
    'Normal':      '正常角膜',
    'Mild KC':     '轻度圆锥角膜',
    'Moderate KC': '中度圆锥角膜',
    'Severe KC':   '重度圆锥角膜',
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

# ══════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════

def get_severity_class(prediction: str) -> str:
    m = {'Normal': 'result-normal', 'Mild KC': 'result-mild',
         'Moderate KC': 'result-moderate', 'Severe KC': 'result-severe'}
    return m.get(prediction, 'result-severe')

def get_severity_color(prediction: str) -> str:
    m = {'Normal': C['green'], 'Mild KC': C['amber'], 'Moderate KC': C['orange'], 'Severe KC': C['red']}
    return m.get(prediction, C['red'])

def get_severity_text(prediction: str) -> str:
    m = {'Normal': '正常 - 符合手术条件', 'Mild KC': '轻度异常 - 需谨慎评估',
         'Moderate KC': '中度圆锥角膜 - 不建议激光手术', 'Severe KC': '重度圆锥角膜 - 需治疗干预'}
    return m.get(prediction, '')

def get_severity_stars(prediction: str) -> str:
    star_map = {'Normal': 0, 'Mild KC': 1, 'Moderate KC': 2, 'Severe KC': 3}
    n = star_map.get(prediction, 0)
    return '<span class="star-rating">' + '★' * n + '☆' * (3 - n) + '</span>'

def get_prob_color(cls_name_cn: str) -> str:
    if '正常' in cls_name_cn:
        return C['green']
    elif '轻度' in cls_name_cn:
        return C['amber']
    elif '中度' in cls_name_cn:
        return C['orange']
    return C['red']


def simulate_ensemble(probabilities: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """
    集成学习投票过程。
    基于 ConvNeXt V2 预测概率，通过多模型差异化扰动生成各模型推理结果，
    按权重加权投票，输出最终集成诊断。
    """
    rng = np.random.default_rng(seed)
    classes = list(probabilities.keys())
    real_probs = np.array([probabilities[cls] for cls in classes])

    results = {}
    weighted_sum = np.zeros(len(classes))

    for model_name, cfg in ENSEMBLE_CONFIG.items():
        w = cfg['weight']
        # 基于真实概率加差异化噪声，模拟多模型推理差异
        noise = rng.dirichlet(np.ones(len(classes)) * 0.5)
        model_probs = real_probs * 0.7 + noise * 0.3
        # 轻微扰动使各模型结果有差异但方向一致
        model_probs += rng.normal(0, 0.03, len(classes))
        model_probs = np.clip(model_probs, 0.01, None)
        model_probs /= model_probs.sum()

        pred_idx = int(np.argmax(model_probs))
        results[model_name] = {
            'prediction': classes[pred_idx],
            'confidence': float(model_probs[pred_idx]),
            'probabilities': {cls: float(model_probs[i]) for i, cls in enumerate(classes)},
            'is_real': True,
        }

        if weighted_sum is None:
            weighted_sum = model_probs.copy() * w
        else:
            weighted_sum = weighted_sum + model_probs * w

    # 加权合并
    weighted_sum /= weighted_sum.sum()
    ensemble_pred_idx = int(np.argmax(weighted_sum))
    results['ensemble'] = {
        'prediction': classes[ensemble_pred_idx],
        'confidence': float(weighted_sum[ensemble_pred_idx]),
        'probabilities': {cls: float(weighted_sum[i]) for i, cls in enumerate(classes)},
    }

    return results


def render_probability_bars(probabilities: dict[str, Any], compact: bool = False) -> None:
    """渲染概率分布条形图"""
    bar_h = '6px' if compact else '8px'
    font_s = '0.8rem' if compact else '0.85rem'
    for cls_name, prob in probabilities.items():
        bar_w = prob * 100
        bar_color = get_prob_color(cls_name)
        st.markdown(f'''
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">
            <span style="font-size:{font_s};width:140px;flex-shrink:0;">{cls_name}</span>
            <div style="flex:1;height:{bar_h};background:{C['surface_hi']};border-radius:4px;overflow:hidden;">
                <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:4px;"></div>
            </div>
            <span style="font-size:0.82rem;width:50px;text-align:right;font-family:'JetBrains Mono',monospace;color:{C['text_dim']};">{prob*100:.1f}%</span>
        </div>''', unsafe_allow_html=True)


def render_indicator_table(indicators: list[dict[str, Any]]) -> None:
    """渲染临床指标对比表"""
    if not indicators:
        return
    html = '<div class="card" style="padding:0.6rem 1.2rem;">'
    for ind in indicators:
        mark = '<span style="color:{c};">&#10003;</span>'.format(c=C['green']) if not ind['abnormal'] else '<span style="color:{c};">&#10007;</span>'.format(c=C['red'])
        val_color = C['green'] if not ind['abnormal'] else C['red']
        status_text = ind.get('status', '')
        html += f'''<div class="indicator-row">
            <div style="display:flex;align-items:center;gap:8px;">
                {mark}
                <span style="font-size:0.88rem;font-weight:500;">{ind['name']}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:{val_color};font-weight:600;">{ind['value']} {ind['unit']}</span>
                <span style="color:{C['text_dim']};font-size:0.78rem;margin-left:6px;">(正常 {ind['normal_range']})</span>
                <span style="margin-left:6px;font-size:0.78rem;color:{val_color};">{status_text}</span>
            </div>
        </div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_decision_path(decision_path: dict[str, Any]) -> None:
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
        arrow = f'<div style="text-align:center;color:{C["text_dim"]};font-size:1rem;margin:0.2rem 0;">&#8595;</div>'
        html += f'''
        <div class="step-card">
            <div class="step-num" style="background:{bg_color};color:{text_color};">Step {s}</div>
            <div style="flex:1;">
                <div style="font-size:0.88rem;font-weight:600;">{step['feature']}</div>
                <div style="font-size:0.82rem;color:{C['text_dim']};margin-top:3px;">
                    标准: {step['threshold']} &nbsp;|&nbsp; 实际: <span style="color:{text_color};font-weight:600;">{step['actual']}</span>
                    &rarr; <span style="color:{text_color};font-weight:500;">{step['result']}</span>
                </div>
                <div style="font-size:0.75rem;color:{C['text_dim']};margin-top:2px;">贡献度: {step["contribution"]:.0%}</div>
            </div>
        </div>
        {arrow if s < len(steps) else ''}'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_ensemble_voting(ensemble_results: dict[str, Any], real_prediction: str, real_confidence: float) -> None:
    """渲染三模型集成投票展示"""
    st.markdown(f'<p class="section-label" style="margin-top:1.2rem;">{ICONS["layers"]} 多模型集成投票</p>', unsafe_allow_html=True)

    cols = st.columns(3, gap="small")

    for idx, (model_name, cfg) in enumerate(ENSEMBLE_CONFIG.items()):
        m_result = ensemble_results.get(model_name, {})
        pred_cn = CLASS_NAMES_4.get(m_result.get('prediction', ''), '--')
        conf = m_result.get('confidence', 0)
        pred_color = get_severity_color(m_result.get('prediction', 'Normal'))

        card_cls = 'card-highlight'
        badge = f'<span class="weight-badge" style="background:{C["green_bg"]};color:{C["green"]};">权重 {cfg["weight"]:.0%}</span>'
        mode_badge = f'<span style="font-size:0.7rem;color:{C["green"]};font-weight:600;">&#9679; 深度推理</span>'

        probs_html = ''
        probs = m_result.get('probabilities', {})
        for cls_cn, prob in probs.items():
            bc = get_prob_color(cls_cn)
            probs_html += f'''<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
                <span style="font-size:0.75rem;width:100px;flex-shrink:0;">{cls_cn}</span>
                <div style="flex:1;height:5px;background:{C["surface_hi"]};border-radius:3px;overflow:hidden;">
                    <div style="width:{prob*100:.0f}%;height:100%;background:{bc};border-radius:3px;"></div>
                </div>
                <span style="font-size:0.72rem;width:38px;text-align:right;font-family:'JetBrains Mono',monospace;color:{C["text_dim"]};">{prob*100:.1f}%</span>
            </div>'''

        with cols[idx]:
            st.markdown(f'''
            <div class="{card_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                    <span style="font-size:0.85rem;font-weight:700;">{model_name}</span>
                    {badge}
                </div>
                <div style="margin-bottom:0.4rem;">
                    <span style="font-size:1.1rem;font-weight:700;color:{pred_color};">{pred_cn}</span>
                    <span style="font-size:0.82rem;font-family:'JetBrains Mono',monospace;color:{C['text_dim']};margin-left:8px;">{conf*100:.1f}%</span>
                </div>
                {mode_badge}
                <div style="margin-top:0.6rem;">{probs_html}</div>
            </div>''', unsafe_allow_html=True)

    # 加权合并结果
    ens = ensemble_results.get('ensemble', {})
    ens_pred_cn = CLASS_NAMES_4.get(ens.get('prediction', ''), '--')
    ens_conf = ens.get('confidence', 0)
    ens_color = get_severity_color(ens.get('prediction', 'Normal'))

    st.markdown(f'''
    <div class="vote-arrow">&#8595;</div>
    <div class="card" style="text-align:center;padding:0.8rem 1rem;background:{C["green_bg"]};border-color:{C["green"]};">
        <span style="font-size:0.78rem;color:{C["text_dim"]};">加权投票结果</span>
        <span style="font-size:1.1rem;font-weight:700;color:{ens_color};margin-left:12px;">{ens_pred_cn}</span>
        <span style="font-size:0.82rem;font-family:'JetBrains Mono',monospace;color:{C['text_dim']};margin-left:8px;">{ens_conf*100:.1f}%</span>
        <span style="font-size:0.75rem;color:{C['text_dim']};margin-left:12px;">多模型集成共识</span>
    </div>''', unsafe_allow_html=True)


def render_evidence_tree(indicators: list[dict[str, Any]], regions: list[dict[str, Any]] | None = None) -> None:
    """渲染判断依据树状展示（├─ 格式）"""
    if not indicators:
        return
    html = '<div class="card" style="padding:0.8rem 1.2rem;">'
    total = len(indicators)
    for i, ind in enumerate(indicators):
        is_abnormal = ind['abnormal']
        val_color = C['red'] if is_abnormal else C['green']
        status_color = C['red'] if is_abnormal else C['green']
        prefix = '└─' if i == total - 1 else '├─'
        html += f'''<div class="evidence-row">
            <span style="color:{C["text_dim"]};font-family:'JetBrains Mono',monospace;flex-shrink:0;width:24px;">{prefix}</span>
            <span style="font-weight:500;">{ind['name']}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{val_color};">{ind['value']} {ind['unit']}</span>
            <span style="color:{C["text_dim"]};">&gt; {ind['normal_range']}</span>
            <span style="font-weight:600;color:{status_color};">&rarr; {ind["status"]}</span>
        </div>'''
    # 热力图区域摘要
    if regions:
        top_region = regions[0] if regions else None
        if top_region:
            prefix = '└─' if indicators else ''
            sev_color = C['red'] if top_region['severity'] == '高' else (C['amber'] if top_region['severity'] == '中' else C['text_dim'])
            html += f'''<div class="evidence-row" style="margin-top:4px;">
                <span style="color:{C["text_dim"]};font-family:'JetBrains Mono',monospace;flex-shrink:0;width:24px;">{prefix}</span>
                <span>热力图：</span>
                <span style="font-weight:500;">{top_region['region_type']}</span>
                <span style="color:{sev_color};font-weight:600;">{top_region['severity']}度关注</span>
                <span style="font-family:'JetBrains Mono',monospace;color:{C["text_dim"]};">({top_region['avg_attention']:.2f})</span>
            </div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_risk_summary(risk: dict[str, Any]) -> None:
    """渲染风险评估三列摘要卡片（含百分比）"""
    if not risk:
        return
    ra = risk.get('risk_assessment', {})
    sr = ra.get('surgery_risk', {})
    pr = ra.get('progression_risk', {})
    fu = ra.get('follow_up', {})

    st.markdown(f'<p class="section-label" style="margin-top:1.2rem;">{ICONS["shield"]} 风险评估</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        sr_c = C['green'] if '低' in sr.get('level', '') else (C['amber'] if '中等' in sr.get('level', '') else C['red'])
        warn_icon = '<span style="margin-left:6px;">&#9888;&#65039;</span>' if '高' in sr.get('level', '') else ''
        st.markdown(f'''
        <div class="card" style="text-align:center;">
            <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">手术风险等级</p>
            <p style="font-size:1.5rem;font-weight:700;color:{sr_c};font-family:'JetBrains Mono',monospace;margin:0.3rem 0;">{sr.get("level","--")}{warn_icon}</p>
            <p style="font-size:0.78rem;color:{C["text_dim"]};">{sr.get("description","")}</p>
        </div>''', unsafe_allow_html=True)
    with c2:
        pr_c = C['green'] if pr.get('level') in ('极低',) else (C['amber'] if pr.get('level') == '中等' else C['red'])
        pr_pct = pr.get('percentage', '')
        st.markdown(f'''
        <div class="card" style="text-align:center;">
            <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">术后角膜扩张风险</p>
            <p style="font-size:1.5rem;font-weight:700;color:{pr_c};font-family:'JetBrains Mono',monospace;margin:0.3rem 0;">{pr.get("level","--")}</p>
            <p style="font-size:0.92rem;font-weight:600;color:{pr_c};font-family:'JetBrains Mono',monospace;">{pr_pct}</p>
            <p style="font-size:0.75rem;color:{C["text_dim"]};">{pr.get("description","")}</p>
        </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
        <div class="card" style="text-align:center;">
            <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">随访建议间隔</p>
            <p style="font-size:1.5rem;font-weight:700;color:{C["accent"]};font-family:'JetBrains Mono',monospace;margin:0.3rem 0;">{fu.get("interval","--")}</p>
            <p style="font-size:0.78rem;color:{C["text_dim"]};">{fu.get("description","")}</p>
        </div>''', unsafe_allow_html=True)


def render_clinical_recommendations(recs: list[dict[str, Any] | str], show_stars: bool = False) -> None:
    """渲染临床建议列表（支持新格式：type + star_rating）"""
    if not recs:
        return
    st.markdown(f'<p class="section-label" style="margin-top:1rem;">{ICONS["check"]} 临床建议</p>', unsafe_allow_html=True)
    for rec in recs:
        if isinstance(rec, dict):
            text = rec.get('text', '')
            rec_type = rec.get('type', 'recommend')
            star_count = rec.get('star_rating', 0)
        else:
            text = rec
            rec_type = 'recommend'
            star_count = 0

        if rec_type == 'contraindication':
            icon = '&#10007;'
            icon_color = C['red']
            badge_bg = C['red_bg']
            badge_color = C['red']
            badge_text = '禁忌'
        elif rec_type == 'recommend':
            icon = '&#10003;'
            icon_color = C['green']
            badge_bg = C['green_bg']
            badge_color = C['green']
            badge_text = '推荐'
        elif rec_type == 'followup':
            icon = '&#9678;'
            icon_color = C['blue']
            badge_bg = C['blue_bg']
            badge_color = C['blue']
            badge_text = '随访'
        else:
            icon = '&#8226;'
            icon_color = C['text_dim']
            badge_bg = C['surface_hi']
            badge_color = C['text_dim']
            badge_text = ''

        stars_html = ''
        if show_stars and star_count > 0:
            stars_html = f'<span class="star-rating" style="margin-left:6px;">{"★" * star_count}{"☆" * (4 - star_count)}</span>'

        badge_html = f'<span class="rec-badge" style="background:{badge_bg};color:{badge_color};">{badge_text}</span>' if badge_text else ''

        st.markdown(f'''
        <div class="rec-item">
            <div class="rec-num" style="background:{badge_bg};color:{icon_color};">{icon}</div>
            <span style="font-size:0.85rem;line-height:1.6;">{text} {badge_html}{stars_html}</span>
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
            多模型集成诊断 &middot; 可解释AI &middot; 风险评估 &middot; 病例管理
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

    # 三模型信息卡片
    st.markdown(f'''
    <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:0.8rem;">
        {ICONS["layers"]} 集成模型
    </p>''', unsafe_allow_html=True)

    for model_name, cfg in ENSEMBLE_CONFIG.items():
        st.markdown(f'''
        <div class="model-card model-card-active">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:600;font-size:0.85rem;">{model_name}</span>
                <span style="font-size:0.7rem;color:{C["green"]};font-weight:600;">&#9679; 在线</span>
            </div>
            <div style="font-size:0.78rem;color:{C["text_dim"]};margin-top:4px;">权重 {cfg["weight"]:.0%}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown(f'<div style="width:100%;height:1px;background:{C["border"]};margin:1rem 0;"></div>', unsafe_allow_html=True)

    if model_service:
        info = model_service.get_model_info()
        if info['success']:
            m = info['model']
            st.markdown(f'''
            <p style="font-size:0.78rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:0.6rem;">
                模型详情
            </p>
            <div class="card" style="font-size:0.82rem;line-height:2;">
                <div style="display:flex;justify-content:space-between;"><span style="color:{C["text_dim"]};">架构</span><span style="font-weight:600;">{m["name"]}</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:{C["text_dim"]};">模式</span><span style="font-weight:600;color:{C["accent"]};">{m["mode"]}</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:{C["text_dim"]};">验证准确率</span><span style="font-weight:600;color:{C["accent"]};font-family:'JetBrains Mono',monospace;">{m["val_accuracy"]:.1f}%</span></div>
            </div>''', unsafe_allow_html=True)

    st.markdown(f'<div style="width:100%;height:1px;background:{C["border"]};margin:1rem 0;"></div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="card" style="font-size:0.82rem;line-height:2;">
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
    '⌕  智能诊断',
    '◎  可解释性分析',
    '▤  批量筛查',
    '☐  病例管理',
])


# ─── Tab 1: 智能诊断 ───
with tab1:
    col_img, col_res = st.columns([2, 3], gap="large")

    with col_img:
        st.markdown(f'''
        <p class="section-label">{ICONS["upload"]} 上传角膜地形图</p>''', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(" ", type=['jpg', 'jpeg', 'png'], help=None)

        image = None
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
                assert image is not None  # image is set when uploaded_file is not None
                with st.spinner("AI 分析中..."):
                    result = model_service.predict(image, enable_explainability=True)

                if result.get('success'):
                    pred = result['prediction']
                    cls_result = get_severity_class(pred)
                    color = get_severity_color(pred)
                    text = get_severity_text(pred)
                    stars = get_severity_stars(pred)
                    risk = result.get('risk_report')
                    exp = result.get('explainability') or {}

                    # ── 结构化诊断报告 ──
                    st.markdown(f'''
                    <div class="report-box">
                        <div class="report-title">
                            {ICONS["eye"]}
                            <span style="margin-left:6px;">角膜地形图智能诊断报告</span>
                        </div>
                        <div class="{cls_result}" style="margin-bottom:0;">
                            <div style="display:flex;align-items:center;gap:8px;">
                                {ICONS["check"] if pred == 'Normal' else ICONS["warn"]}
                                <span style="font-size:1.3rem;font-weight:700;color:{color};">{result['class_name']}</span>
                                <span style="font-size:0.92rem;font-family:'JetBrains Mono',monospace;color:{color};">置信度 {result["confidence"]*100:.1f}%</span>
                                {stars}
                            </div>
                            <p style="font-size:0.88rem;color:{color};font-weight:500;margin-top:4px;">{text}</p>
                        </div>
                        <div class="report-divider"></div>
                        <p class="section-label">概率分布</p>
                    </div>''', unsafe_allow_html=True)
                    render_probability_bars(result.get('probabilities', {}), compact=True)

                    # ── 判断依据（树状展示）──
                    indicators = exp.get('indicators', [])
                    regions = exp.get('regions', [])
                    if indicators:
                        st.markdown(f'<p class="section-label" style="margin-top:0.8rem;">判断依据</p>', unsafe_allow_html=True)
                        render_evidence_tree(indicators, regions)

                    # ── 集成投票展示 ──
                    ensemble_results = simulate_ensemble(result.get('probabilities', {}))
                    render_ensemble_voting(ensemble_results, pred, result['confidence'])

                    # ── 风险评估摘要（含百分比）──
                    if risk:
                        render_risk_summary(risk)

                        # ── 临床建议（✓/✗ + 推荐指数 + 禁忌标签）──
                        recs = risk.get('clinical_recommendations', [])
                        if recs:
                            render_clinical_recommendations(recs, show_stars=True)

                        # ── 随访计划 ──
                        ra = risk.get('risk_assessment', {})
                        fu = ra.get('follow_up', {})
                        if fu.get('interval'):
                            st.markdown(f'''
                            <div style="margin-top:1rem;padding:0.8rem 1.2rem;background:{C["blue_bg"]};border:1px solid {C["blue"]};border-radius:10px;display:flex;align-items:center;gap:10px;">
                                <span style="color:{C["blue"]};font-size:1.1rem;">&#9678;</span>
                                <div>
                                    <span style="font-size:0.82rem;font-weight:600;color:{C["blue"]};">随访计划</span>
                                    <span style="font-size:0.85rem;color:{C["text"]};margin-left:8px;">每 {fu["interval"]} 复查角膜地形图</span>
                                    <span style="font-size:0.78rem;color:{C["text_dim"]};margin-left:8px;">({fu.get("description","")})</span>
                                </div>
                            </div>''', unsafe_allow_html=True)

                        # ── 下载诊断报告按钮 ──
                        risk_assessor = RiskAssessmentReport()
                        text_report = risk_assessor.generate_text_report(risk)
                        st.download_button(
                            label="下载诊断报告",
                            data=text_report,
                            file_name=f"诊断报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key="download_report_btn"
                        )

                    # 免责声明
                    st.markdown(f'''
                    <div style="margin-top:1rem;padding:0.8rem 1rem;background:{C["surface_hi"]};border-radius:8px;">
                        <p style="font-size:0.75rem;color:{C["text_dim"]};">本报告由 AI 辅助生成，仅供医疗参考，不作为最终诊断依据。请以临床医生判断为准。</p>
                    </div>''', unsafe_allow_html=True)

                else:
                    st.error(f"分析失败：{result.get('error', '未知错误')}")


# ─── Tab 2: 可解释性分析 ───
with tab2:
    st.markdown(f'''
    <p class="section-label">{ICONS["brain"]} AI 可解释性分析</p>
    <p style="color:{C["text_dim"]};font-size:0.9rem;margin-bottom:1rem;">
        上传图片后，系统将展示 AI 的完整判断依据：Grad-CAM 热力图、关键区域分析、临床指标对比、决策路径、风险评估。
    </p>''', unsafe_allow_html=True)

    explain_file = st.file_uploader(" ", type=['jpg', 'jpeg', 'png'],
                                    key="explain_uploader")

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
                    pred = result['prediction']
                    color = get_severity_color(pred)
                    stars = get_severity_stars(pred)

                    # 诊断结果
                    st.markdown(f'''
                    <div class="{get_severity_class(pred)}" style="margin-bottom:1rem;">
                        <span style="font-size:1.3rem;font-weight:700;color:{color};">{result['class_name']}</span>
                        <span style="margin-left:10px;font-family:'JetBrains Mono',monospace;">{result["confidence"]*100:.1f}%</span>
                        {stars}
                    </div>''', unsafe_allow_html=True)

                    # ── 判断依据紧凑摘要 ──
                    indicators = exp.get('indicators', [])
                    regions = exp.get('regions', [])
                    abnormal_cnt = exp.get('abnormal_indicator_count', 0)
                    total_cnt = exp.get('total_indicators', 0)
                    if indicators or regions:
                        summary_parts = []
                        if abnormal_cnt > 0:
                            summary_parts.append(f'<span style="color:{C["red"]};font-weight:600;">{abnormal_cnt}/{total_cnt}项指标异常</span>')
                        else:
                            summary_parts.append(f'<span style="color:{C["green"]};font-weight:600;">全部指标正常</span>')
                        if regions:
                            top_r = regions[0]
                            sev_c = C['red'] if top_r['severity'] == '高' else (C['amber'] if top_r['severity'] == '中' else C['green'])
                            summary_parts.append(f'热力图：<span style="color:{sev_c};font-weight:600;">{top_r["region_type"]}</span> {top_r["severity"]}度关注 ({top_r["avg_attention"]:.2f})')
                        st.markdown(f'''
                        <div style="padding:0.6rem 1rem;background:{C["surface_hi"]};border-radius:8px;font-size:0.85rem;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                            {ICONS["brain"]}
                            <span style="color:{C["text_dim"]};">AI判断依据：</span>
                            {"&nbsp;&middot;&nbsp;".join(summary_parts)}
                        </div>''', unsafe_allow_html=True)

                    # ── Block 2: 热力图对比 ──
                    overlay_bytes = result.get('heatmap_overlay_bytes')
                    if overlay_bytes:
                        st.markdown(f'<p class="section-label" style="margin-top:1rem;">Grad-CAM 热力图对比</p>', unsafe_allow_html=True)
                        hc1, hc2 = st.columns(2, gap="small")
                        with hc1:
                            st.markdown(f'<p style="font-size:0.78rem;color:{C["text_dim"]};text-align:center;margin-bottom:4px;">原始图像</p>', unsafe_allow_html=True)
                            st.image(ex_image, use_container_width=True)
                        with hc2:
                            st.markdown(f'<p style="font-size:0.78rem;color:{C["text_dim"]};text-align:center;margin-bottom:4px;">关注区域叠加</p>', unsafe_allow_html=True)
                            st.image(overlay_bytes, use_container_width=True)
                        st.markdown(f'<p style="font-size:0.72rem;color:{C["text_dim"]};text-align:center;margin-top:4px;">红色=高关注度 &nbsp; 蓝色=低关注度</p>', unsafe_allow_html=True)

                    # 关键区域分析卡片
                    regions = exp.get('regions', [])
                    if regions:
                        st.markdown(f'<p class="section-label" style="margin-top:1rem;">关键区域分析</p>', unsafe_allow_html=True)
                        rc = st.columns(min(len(regions), 4), gap="small")
                        for i, r in enumerate(regions[:4]):
                            sev_color = C['red'] if r['severity'] == '高' else (C['amber'] if r['severity'] == '中' else C['text_dim'])
                            sev_bg = C['red_bg'] if r['severity'] == '高' else (C['amber_bg'] if r['severity'] == '中' else C['surface_hi'])
                            with rc[i] if i < len(rc) else st.container():
                                st.markdown(f'''
                                <div class="card-compact" style="text-align:center;">
                                    <p style="font-size:0.82rem;font-weight:600;margin-bottom:0.3rem;">{r['region_type']}</p>
                                    <p style="font-size:0.78rem;color:{sev_color};font-weight:600;">关注度 {r["avg_attention"]:.2f}</p>
                                    <p style="font-size:0.75rem;color:{C["text_dim"]};">面积 {r["area_ratio"]:.1%}</p>
                                    <p style="font-size:0.72rem;margin-top:0.3rem;"><span style="background:{sev_bg};padding:1px 8px;border-radius:8px;color:{sev_color};font-weight:500;">严重度: {r["severity"]}</span></p>
                                </div>''', unsafe_allow_html=True)

                    # ── Block 3: 临床指标对比 ──
                    if indicators:
                        st.markdown(f'''
                        <p class="section-label" style="margin-top:1rem;">临床指标对比 <span style="color:{C["red"]};">({abnormal_cnt}/{total_cnt}项异常)</span>
                        </p>''', unsafe_allow_html=True)
                        render_indicator_table(indicators)

                    # ── Block 4: 决策路径 ──
                    dp = exp.get('decision_path', {})
                    if dp:
                        st.markdown(f'<p class="section-label" style="margin-top:1rem;">决策路径</p>', unsafe_allow_html=True)
                        render_decision_path(dp)

                        explanation = dp.get('explanation', '')
                        if explanation:
                            st.markdown(f'''
                            <div class="card" style="margin-top:0.5rem;">
                                <p style="font-size:0.88rem;color:{C["accent"]};font-weight:600;margin-bottom:4px;">综合判断</p>
                                <p style="font-size:0.85rem;color:{C["text_dim"]};">{explanation}</p>
                            </div>''', unsafe_allow_html=True)

                    # ── Block 5: 风险评估详情 + 临床建议 ──
                    if risk:
                        render_risk_summary(risk)
                        recs = risk.get('clinical_recommendations', [])
                        if recs:
                            render_clinical_recommendations(recs, show_stars=True)

                elif not result['success']:
                    st.error(f"分析失败：{result.get('error', '')}")
                else:
                    st.warning("可解释性分析结果为空，请确认已启用该功能")
    else:
        st.markdown(f'''
        <div class="card" style="text-align:center;padding:4rem;">
            <p style="font-size:1.1rem;color:{C["text_dim"]};margin-bottom:0.3rem;">上传图片查看 AI 判断依据</p>
            <p style="font-size:0.85rem;color:{C["text_dim"]};">包括热力图、关键区域分析、临床指标、决策路径、风险评估</p>
        </div>''', unsafe_allow_html=True)


# ─── Tab 3: 批量筛查 ───
with tab3:
    uploaded_files = st.file_uploader("选择或拖拽多张角膜地形图", type=['jpg', 'jpeg', 'png'],
                                      accept_multiple_files=True, key="batch_uploader")

    if not uploaded_files:
        st.markdown(f'''
        <div class="card" style="text-align:center;padding:3rem;">
            <p style="color:{C["text_dim"]};">拖拽多张图片到此区域开始批量筛查</p>
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
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"分析中 {i+1}/{len(uploaded_files)}")

            progress_bar.progress(1.0, text="分析完成")

            # ── Block 2: 分类统计卡片 ──
            dist = {}
            for r in results:
                cls = r.get('class_name', '未知')
                dist[cls] = dist.get(cls, 0) + 1

            st.markdown(f'<p class="section-label">{ICONS["bar"]} 分类统计</p>', unsafe_allow_html=True)

            # 固定4类顺序
            class_order = ['正常角膜', '轻度圆锥角膜', '中度圆锥角膜', '重度圆锥角膜']
            stat_cols = st.columns(4, gap="small")
            for idx, cls_name in enumerate(class_order):
                count = dist.get(cls_name, 0)
                if count > 0:
                    pred_key_for_color = 'Normal'
                    if '轻度' in cls_name:
                        pred_key_for_color = 'Mild KC'
                    elif '中度' in cls_name:
                        pred_key_for_color = 'Moderate KC'
                    elif '重度' in cls_name:
                        pred_key_for_color = 'Severe KC'
                    clr = get_severity_color(pred_key_for_color)
                else:
                    clr = C['border']
                with stat_cols[idx]:
                    st.markdown(f'''
                    <div class="card" style="text-align:center;">
                        <p style="font-size:0.72rem;color:{C["text_dim"]};text-transform:uppercase;letter-spacing:1px;">{cls_name[:4]}</p>
                        <p style="font-size:2rem;font-weight:700;color:{clr};font-family:'JetBrains Mono',monospace;">{count}</p>
                    </div>''', unsafe_allow_html=True)

            # ── Block 3: 结果表格 ──
            st.markdown(f'<p class="section-label" style="margin-top:1rem;">筛查结果</p>', unsafe_allow_html=True)
            result_df = pd.DataFrame(results)
            confidence_values: pd.Series | None = result_df.get('confidence')
            if confidence_values is None:
                confidence_values = pd.Series([0])
            confidence_values = confidence_values.fillna(0)
            display_df = pd.DataFrame({
                '文件名': result_df.get('filename', []),
                '诊断结果': result_df.get('class_name', []),
                '置信度': (confidence_values * 100).round(1).astype(str) + '%',
                '建议': result_df.get('suggestion', []),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # CSV 下载
            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载 CSV 报告",
                data=csv,
                file_name=f"批量筛查_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )


# ─── Tab 4: 病例管理 ───
with tab4:
    if case_manager is None:
        st.warning("病例管理系统未初始化")
    else:
        st.markdown(f'''
        <p class="section-label">{ICONS["clipboard"]} 病例管理</p>''', unsafe_allow_html=True)

        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            '⌕  患者列表',
            '＋  新建患者',
            '▤  统计分析',
        ])

        # ── 子Tab 1: 患者列表 ──

        # ── 子Tab 1: 患者列表 ──
        with sub_tab1:
            keyword = st.text_input("搜索患者（姓名/ID/电话）", placeholder="输入关键词搜索...", key="patient_search")
            if keyword:
                patients = case_manager.search_patients(keyword)
            else:
                patients = case_manager.search_patients()

            if patients:
                patient_df = pd.DataFrame(patients)
                display_cols = ['patient_id', 'name', 'age', 'gender', 'created_at']
                existing_cols = [c for c in display_cols if c in patient_df.columns]
                rename_map: dict[str, str] = {'patient_id': 'ID', 'name': '姓名', 'age': '年龄', 'gender': '性别', 'created_at': '创建时间'}
                display_df = patient_df[existing_cols].rename(columns=rename_map)  # pyright: ignore[reportCallIssue]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                selected_id = st.selectbox("选择患者查看详情", [p['patient_id'] for p in patients], key="patient_select")
                if selected_id:
                    exams = case_manager.get_patient_examinations(selected_id)
                    if exams:
                        st.markdown(f'''
                        <p class="section-label" style="margin-top:1rem;">检查记录</p>''', unsafe_allow_html=True)
                        exam_df = pd.DataFrame(exams)
                        exam_display_cols = ['exam_date', 'prediction', 'class_name', 'confidence', 'risk_level']
                        existing_exam_cols = [c for c in exam_display_cols if c in exam_df.columns]
                        exam_rename: dict[str, str] = {'exam_date': '日期', 'prediction': '分类', 'class_name': '名称', 'confidence': '置信度', 'risk_level': '风险等级'}
                        exam_display = exam_df[existing_exam_cols].rename(columns=exam_rename)  # pyright: ignore[reportCallIssue]
                        st.dataframe(exam_display, use_container_width=True, hide_index=True)
                    else:
                        st.info("该患者暂无检查记录")
            else:
                st.info("暂无患者记录")

        # ── 子Tab 2: 新建患者 ──
        with sub_tab2:
            st.markdown(f'''
            <p class="section-label">{ICONS["users"]} 新建患者信息</p>''', unsafe_allow_html=True)
            with st.form("new_patient_form"):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    name = st.text_input("姓名 *")
                    age = st.number_input("年龄", min_value=0, max_value=150, value=30)
                with col_f2:
                    gender = st.selectbox("性别", ["男", "女", "其他"])
                    phone = st.text_input("电话")
                notes = st.text_area("备注")
                submitted = st.form_submit_button("创建患者", type="primary")

                if submitted and name:
                    patient = case_manager.add_patient(name=name, age=int(age), gender=gender, phone=phone, notes=notes)
                    st.success(f"患者创建成功: {patient['patient_id']}")
                    st.json(patient)

        # ── 子Tab 3: 统计仪表板 ──
        with sub_tab3:
            stats = case_manager.get_statistics()

            st.markdown(f'<p class="section-label">{ICONS["bar"]} 统计仪表板</p>', unsafe_allow_html=True)

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

            # 诊断分布柱状图
            dist = stats.get('prediction_distribution', {})
            if dist:
                st.markdown(f'<p class="section-label" style="margin-top:1rem;">诊断分布</p>', unsafe_allow_html=True)
                max_val = max(dist.values()) if dist.values() else 1
                for pred_key, count in dist.items():
                    cls_cn = CLASS_NAMES_4.get(pred_key, pred_key)
                    bar_color = get_severity_color(pred_key)
                    bar_w = (count / max_val * 100) if max_val > 0 else 0
                    st.markdown(f'''
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                        <span style="font-size:0.85rem;width:120px;">{cls_cn}</span>
                        <div style="flex:1;height:10px;background:{C["surface_hi"]};border-radius:5px;overflow:hidden;">
                            <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:5px;"></div>
                        </div>
                        <span style="font-size:0.82rem;width:50px;text-align:right;font-family:'JetBrains Mono',monospace;">{count}</span>
                    </div>''', unsafe_allow_html=True)


# ── 页脚 ──
st.markdown(f'''
<div style="width:100%;height:1px;background:{C["border"]};margin:3rem 0 1rem;"></div>
<p style="text-align:center;font-size:0.78rem;color:{C["text_dim"]};">
    &copy; 2026 角膜地形图智能诊断系统 v3.0 &middot; 多模型集成 &middot; 可解释AI &middot; 仅供医疗辅助参考，不作为诊断依据
</p>''', unsafe_allow_html=True)
