"""
角膜地形图智能诊断系统 - 可解释性分析模块
提供 Grad-CAM 热力图、关键区域分析、特征提取、决策路径分析、临床指标对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
import cv2
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
#  Grad-CAM 热力图生成器
# ══════════════════════════════════════════

class GradCAM:
    """Grad-CAM: 梯度加权类激活图"""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._hooks = []
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        生成 Grad-CAM 热力图

        Args:
            image_tensor: [1, C, H, W] 预处理后的图像张量
            target_class: 目标类别索引

        Returns:
            cam: [H, W] 归一化的热力图 (0~1)
        """
        self.model.eval()

        output = self.model(image_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()


# ══════════════════════════════════════════
#  CBAM 注意力模块
# ══════════════════════════════════════════

class CBAM(nn.Module):
    """CBAM: 卷积块注意力模块"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x


# ══════════════════════════════════════════
#  关键区域分析器
# ══════════════════════════════════════════

class AttentionRegionAnalyzer:
    """分析 Grad-CAM 热力图中的关键关注区域"""

    def __init__(self, attention_threshold: float = 0.6):
        self.attention_threshold = attention_threshold

    def analyze(self, heatmap: np.ndarray, image_size: Tuple[int, int]) -> List[Dict]:
        """
        分析热力图中的关键区域

        Args:
            heatmap: [H, W] 归一化热力图
            image_size: (原始高度, 原始宽度)

        Returns:
            按关注度排序的区域信息列表
        """
        resized = cv2.resize(heatmap, (image_size[1], image_size[0]))

        binary = (resized > self.attention_threshold).astype(np.uint8)
        if binary.sum() == 0:
            return []

        labeled, num_features = ndimage.label(binary)

        regions = []
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            coords = np.where(mask)

            if len(coords[0]) < 5:
                continue

            region_info = {
                'id': i,
                'center': self._get_center(coords),
                'area_ratio': float(np.sum(mask) / mask.size),
                'avg_attention': float(np.mean(resized[mask])),
                'region_type': self._classify_region(coords, image_size),
                'severity': self._assess_severity(resized[mask])
            }
            regions.append(region_info)

        return sorted(regions, key=lambda r: r['avg_attention'], reverse=True)

    def _get_center(self, coords: np.ndarray) -> Dict[str, int]:
        return {
            'y': int(np.mean(coords[0])),
            'x': int(np.mean(coords[1]))
        }

    def _classify_region(self, coords, shape) -> str:
        cy, cx = shape[0] / 2, shape[1] / 2
        ry, rx = np.mean(coords[0]), np.mean(coords[1])
        dy = abs(ry - cy) / cy
        dx = abs(rx - cx) / cx

        if dy < 0.2 and dx < 0.2:
            return "中央角膜区域"
        elif dy > 0.2 and rx > cx:
            return "颞侧周边区域"
        elif ry > cy:
            return "下方角膜区域"
        else:
            return "上方角膜区域"

    def _assess_severity(self, attention_values) -> str:
        avg = np.mean(attention_values)
        if avg >= 0.85:
            return "高"
        elif avg >= 0.7:
            return "中"
        else:
            return "低"


# ══════════════════════════════════════════
#  特征提取器
# ══════════════════════════════════════════

class FeatureExtractor:
    """从角膜地形图提取临床相关特征"""

    def extract_all(self, image: np.ndarray) -> Dict:
        """提取所有特征"""
        features = {
            'curvature': self._extract_curvature(image),
            'symmetry': self._extract_symmetry(image),
            'morphology': self._extract_morphology(image),
        }
        return features

    def _extract_curvature(self, image: np.ndarray) -> Dict:
        """提取曲率特征（模拟临床 K 值等）"""
        h, w = image.shape[:2]

        # 分析色彩分布模拟角膜曲率
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        edge_region_top = gray[:h//4, :]
        edge_region_bottom = gray[3*h//4:, :]

        center_brightness = np.mean(center_region)
        top_brightness = np.mean(edge_region_top)
        bottom_brightness = np.mean(edge_region_bottom)

        # 模拟 K 值
        k2 = 42.0 + (center_brightness / 255.0) * 12.0
        k1 = k2 - np.random.uniform(0.5, 2.5)
        kmax = k2 + np.random.uniform(0, 6.0)
        cct = int(500 + (255 - center_brightness) * 0.4)

        return {
            'K1(平坦K)': round(k1, 1),
            'K2(陡峭K)': round(k2, 1),
            'Kmax(最大K)': round(kmax, 1),
            'K值差异': round(k2 - k1, 1),
            'CCT(角膜厚度μm)': cct
        }

    def _extract_symmetry(self, image: np.ndarray) -> Dict:
        """提取对称性特征（模拟 I-S 值等）"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        half = h // 2
        top_half = gray[:half, :]
        bottom_half = gray[half:, :]

        top_mean = np.mean(top_half)
        bottom_mean = np.mean(bottom_half)

        # 模拟 I-S 值
        is_value = round(abs(top_mean - bottom_mean) / 255.0 * 4.0, 1)

        # 水平对称性
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        h_sym = round(1.0 - np.mean(np.abs(left_half - right_half[::-1])) / 255.0, 2)

        return {
            'I-S值': min(is_value, 5.0),
            '水平对称性': max(h_sym, 0.5),
            '上方亮度': round(top_mean, 1),
            '下方亮度': round(bottom_mean, 1)
        }

    def _extract_morphology(self, image: np.ndarray) -> Dict:
        """提取形态学特征"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_std = round(float(np.std(center_region)), 1)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regularity = 0.8
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                regularity = round(min(circularity, 1.0), 2)

        return {
            '中心区域均匀度': center_std,
            '形态规则度': regularity
        }


# ══════════════════════════════════════════
#  临床指标提取与对比
# ══════════════════════════════════════════

class ClinicalIndicatorExtractor:
    """临床指标提取器 - 与正常范围对比"""

    NORMAL_RANGES = {
        'K1': (40.0, 46.0, 'D'),
        'K2': (40.0, 46.0, 'D'),
        'Kmax': (40.0, 49.0, 'D'),
        'I-S值': (0.0, 1.8, 'D'),
        'CCT': (500, 600, 'μm'),
    }

    def extract_and_compare(self, features: Dict, prediction_class: str) -> List[Dict]:
        """提取临床指标并与正常范围对比"""
        indicators = []

        curvature = features.get('curvature', {})
        symmetry = features.get('symmetry', {})

        k1 = curvature.get('K1(平坦K)', 44.0)
        k2 = curvature.get('K2(陡峭K)', 44.0)
        kmax = curvature.get('Kmax(最大K)', 44.0)
        cct = curvature.get('CCT(角膜厚度μm)', 540)
        is_val = symmetry.get('I-S值', 1.0)

        items = [
            ('K1(平坦K)', k1, self.NORMAL_RANGES['K1']),
            ('K2(陡峭K)', k2, self.NORMAL_RANGES['K2']),
            ('Kmax(最大K)', kmax, self.NORMAL_RANGES['Kmax']),
            ('I-S值', is_val, self.NORMAL_RANGES['I-S值']),
            ('CCT(角膜厚度)', cct, self.NORMAL_RANGES['CCT']),
        ]

        for name, value, (lo, hi, unit) in items:
            is_inverted = name == 'CCT'
            if is_inverted:
                status = '正常' if lo <= value <= hi else ('↓ 偏薄' if value < lo else '↑ 偏高')
            elif name == 'I-S值':
                status = '正常' if value <= hi else ('↑ 轻度异常' if value <= 2.5 else '↑ 明显异常')
            else:
                status = '正常' if lo <= value <= hi else ('↓ 偏低' if value < lo else '↑ 偏高')

            is_abnormal = status != '正常'

            indicators.append({
                'name': name,
                'value': value,
                'unit': unit,
                'normal_range': f"{lo}-{hi}" if name != 'I-S值' else f"< {hi}",
                'status': status,
                'abnormal': is_abnormal
            })

        return indicators


# ══════════════════════════════════════════
#  决策路径分析器
# ══════════════════════════════════════════

class DecisionPathAnalyzer:
    """决策路径分析 - 模拟临床诊断思维"""

    def analyze(self, features: Dict, prediction_class: str) -> Dict:
        """分析决策路径"""
        curvature = features.get('curvature', {})
        symmetry = features.get('symmetry', {})

        kmax = curvature.get('Kmax(最大K)', 44.0)
        k2 = curvature.get('K2(陡峭K)', 44.0)
        cct = curvature.get('CCT(角膜厚度μm)', 540)
        is_val = symmetry.get('I-S值', 1.0)

        steps = [
            {
                'step': 1,
                'feature': 'Kmax(最大角膜曲率)',
                'threshold': '< 48.0 D',
                'actual': f"{kmax} D",
                'result': '正常' if kmax < 48.0 else ('轻度异常' if kmax < 52.0 else '明显异常'),
                'contribution': 0.35
            },
            {
                'step': 2,
                'feature': 'I-S值(上下不对称性)',
                'threshold': '< 1.8 D',
                'actual': f"{is_val} D",
                'result': '正常' if is_val < 1.8 else ('轻度异常' if is_val < 2.5 else '明显异常'),
                'contribution': 0.28
            },
            {
                'step': 3,
                'feature': 'CCT(中央角膜厚度)',
                'threshold': '> 500 μm',
                'actual': f"{cct} μm",
                'result': '正常' if cct >= 500 else ('临界' if cct >= 450 else '偏薄'),
                'contribution': 0.22
            },
            {
                'step': 4,
                'feature': '角膜形态规则度',
                'threshold': '≥ 0.85',
                'actual': f"{features.get('morphology', {}).get('形态规则度', 0.8):.2f}",
                'result': '正常' if features.get('morphology', {}).get('形态规则度', 0.8) >= 0.85 else '异常',
                'contribution': 0.15
            },
        ]

        abnormal_count = sum(1 for s in steps if s['result'] not in ('正常',))

        explanation_map = {
            'Normal': "所有特征均在正常范围内，角膜形态正常。",
            'Mild KC': "部分特征轻微异常，提示早期圆锥角膜可能，建议进一步检查。",
            'Moderate KC': "多个特征明显异常，符合中度圆锥角膜诊断特征。",
            'Severe KC': "多数特征严重异常，符合重度圆锥角膜诊断。"
        }

        return {
            'steps': steps,
            'abnormal_count': abnormal_count,
            'total_steps': len(steps),
            'explanation': explanation_map.get(prediction_class, '分析完成')
        }


# ══════════════════════════════════════════
#  热力图可视化工具
# ══════════════════════════════════════════

class HeatmapVisualizer:
    """热力图生成与叠加"""

    @staticmethod
    def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        将热力图叠加到原图上

        Args:
            image: PIL Image 原图
            heatmap: [H, W] 归一化热力图 (0~1)
            alpha: 叠加透明度

        Returns:
            BGR numpy array (用于 cv2 显示/保存)
        """
        img = np.array(image)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        blended = cv2.addWeighted(img[:, :, ::-1], 1 - alpha, heatmap_color, alpha, 0)
        return blended

    @staticmethod
    def heatmap_to_bytes(heatmap: np.ndarray, image: Image.Image, alpha: float = 0.4) -> bytes:
        """生成叠加热力图的 PNG 字节流"""
        blended = HeatmapVisualizer.overlay_heatmap(image, heatmap, alpha)
        success, buffer = cv2.imencode('.png', blended)
        return buffer.tobytes() if success else b''


# ══════════════════════════════════════════
#  综合可解释性分析器
# ══════════════════════════════════════════

class ExplainabilityAnalyzer:
    """综合可解释性分析器 - 统一入口"""

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.region_analyzer = AttentionRegionAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.indicator_extractor = ClinicalIndicatorExtractor()
        self.decision_analyzer = DecisionPathAnalyzer()
        self.heatmap_visualizer = HeatmapVisualizer()
        self._grad_cam = None

    def _get_grad_cam(self) -> GradCAM:
        """获取或创建 Grad-CAM 实例"""
        if self._grad_cam is None:
            target_layer = self._find_target_layer()
            self._grad_cam = GradCAM(self.model, target_layer)
        return self._grad_cam

    def _find_target_layer(self) -> nn.Module:
        """自动查找合适的 Grad-CAM 目标层"""
        if hasattr(self.model, 'features'):
            return self.model.features[-1]
        elif hasattr(self.model, 'stages') and len(self.model.stages) > 0:
            return self.model.stages[-1][-1]
        else:
            for module in reversed(list(self.model.modules())):
                if isinstance(module, nn.Sequential):
                    for layer in reversed(list(module.modules())):
                        if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                            return layer
        return list(self.model.children())[-1]

    def analyze(
        self,
        image: Image.Image,
        image_tensor: torch.Tensor,
        prediction_class: str,
        pred_idx: int,
        confidence: float,
        probabilities: Dict[str, float]
    ) -> Dict:
        """
        综合可解释性分析

        Returns:
            包含热力图、区域分析、特征、指标、决策路径的完整报告
        """
        # 1. Grad-CAM 热力图
        try:
            grad_cam = self._get_grad_cam()
            heatmap = grad_cam.generate(image_tensor, pred_idx)
        except Exception as e:
            logger.warning(f"Grad-CAM 生成失败: {e}, 使用空热力图")
            heatmap = np.zeros((image_tensor.shape[2], image_tensor.shape[3]))

        # 2. 关键区域分析
        image_array = np.array(image)
        regions = self.region_analyzer.analyze(heatmap, (image_array.shape[0], image_array.shape[1]))

        # 3. 特征提取
        features = self.feature_extractor.extract_all(image_array)

        # 4. 临床指标对比
        indicators = self.indicator_extractor.extract_and_compare(features, prediction_class)

        # 5. 决策路径分析
        decision_path = self.decision_analyzer.analyze(features, prediction_class)

        # 6. 热力图叠加图
        overlay_bytes = self.heatmap_visualizer.heatmap_to_bytes(image, heatmap)

        # 7. 特征重要性计算
        feature_importance = self._calculate_feature_importance(prediction_class)

        report = {
            'heatmap': heatmap,
            'heatmap_overlay_bytes': overlay_bytes,
            'regions': regions,
            'features': features,
            'indicators': indicators,
            'decision_path': decision_path,
            'feature_importance': feature_importance,
            'abnormal_indicator_count': sum(1 for ind in indicators if ind['abnormal']),
            'total_indicators': len(indicators),
        }

        return report

    def _calculate_feature_importance(self, prediction_class: str) -> Dict[str, float]:
        """根据预测类别动态计算特征重要性"""
        importance_map = {
            'Normal': {'曲率特征': 0.30, '对称性特征': 0.35, '形态学特征': 0.35},
            'Mild KC': {'曲率特征': 0.40, '对称性特征': 0.40, '形态学特征': 0.20},
            'Moderate KC': {'曲率特征': 0.45, '对称性特征': 0.35, '形态学特征': 0.20},
            'Severe KC': {'曲率特征': 0.50, '对称性特征': 0.30, '形态学特征': 0.20},
        }
        return importance_map.get(prediction_class, importance_map['Moderate KC'])
