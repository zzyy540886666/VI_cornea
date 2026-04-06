"""
角膜地形图智能诊断系统 - 模型服务封装 (v2.0 增强版)
支持: 4分类 / 集成模型 / 可解释性分析 / 风险评估
"""

import os
import time
import traceback

_cloud_env = os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('IS_STREAMLIT_CLOUD')
if not _cloud_env and os.environ.get('HF_ENDPOINT') is None:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from timm import create_model
from PIL import Image
from pathlib import Path
from typing import List, Dict, Union, Optional
import logging
from datetime import datetime
import numpy as np

from explainability import ExplainabilityAnalyzer
from risk_assessment import RiskAssessmentReport

HF_REPO_ID = "zzy4088/corneal-model"
HF_MODEL_FILE = "best_model.pth"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
#  类别配置
# ══════════════════════════════════════════

CLASS_CONFIG_4 = {
    'Normal':      {'idx': 0, 'name_cn': '正常角膜',      'color': '#588157', 'severity': 0},
    'Mild KC':     {'idx': 1, 'name_cn': '轻度圆锥角膜',  'color': '#DDA15E', 'severity': 1},
    'Moderate KC': {'idx': 2, 'name_cn': '中度圆锥角膜',  'color': '#E07A3A', 'severity': 2},
    'Severe KC':   {'idx': 3, 'name_cn': '重度圆锥角膜',  'color': '#BC4749', 'severity': 3},
}

CLASS_CONFIG_2 = {
    'Normal': {'idx': 0, 'name_cn': '正常角膜', 'color': '#588157', 'severity': 0},
    'KC':     {'idx': 1, 'name_cn': '圆锥角膜（异常）', 'color': '#BC4749', 'severity': 2},
}


class ModelService:
    """模型服务类 v2.0 - 支持4分类+可解释性"""

    _instance = None
    _model = None
    _device = None
    _transform = None
    _explainability = None
    _risk_assessor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._initialize()

    def _download_from_hf(self, max_retries: int = 3) -> Path:
        from huggingface_hub import hf_hub_download
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"下载尝试 {attempt}/{max_retries}...")
                kwargs = {
                    "repo_id": HF_REPO_ID,
                    "filename": HF_MODEL_FILE,
                    "local_dir": "checkpoints",
                }
                hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
                if hf_token:
                    kwargs["token"] = hf_token
                cached_path = hf_hub_download(**kwargs)
                downloaded_path = Path(cached_path)
                if downloaded_path.exists() and downloaded_path.stat().st_size > 0:
                    logger.info(f"模型下载成功：{downloaded_path}")
                    return downloaded_path
                else:
                    raise FileNotFoundError(f"下载的文件为空")
            except Exception as e:
                last_error = e
                logger.warning(f"下载失败（尝试 {attempt}/{max_retries}）：{e}")
                if attempt < max_retries:
                    time.sleep(3 * attempt)
        raise FileNotFoundError(f"模型下载失败（已重试 {max_retries} 次）：{last_error}")

    def _initialize(self):
        logger.info("正在初始化模型服务 (v2.0)...")
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备：{self._device}")

        self._model_path = Path('checkpoints/best_model.pth')
        if not self._model_path.exists():
            logger.info("本地模型文件不存在，正在从 Hugging Face 下载...")
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            self._model_path = self._download_from_hf()

        logger.info(f"模型文件路径：{self._model_path}（大小：{self._model_path.stat().st_size / 1024 / 1024:.1f} MB）")

        # 检测模型类别数
        self.num_classes = self._detect_num_classes()

        self._model = create_model(
            'convnextv2_base',
            pretrained=False,
            num_classes=self.num_classes,
            in_chans=3
        )

        logger.info(f"正在加载模型权重：{self._model_path}")
        try:
            checkpoint = torch.load(self._model_path, map_location=self._device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self._model_path, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 配置类别映射
        if self.num_classes == 4:
            self.class_config = CLASS_CONFIG_4
            self.idx_to_class = {v['idx']: k for k, v in CLASS_CONFIG_4.items()}
        else:
            self.class_config = CLASS_CONFIG_2
            self.idx_to_class = {v['idx']: k for k, v in CLASS_CONFIG_2.items()}

        self.model_info = {
            'name': 'ConvNeXt V2 Base',
            'version': '2.0.0',
            'input_size': 224,
            'num_classes': self.num_classes,
            'val_accuracy': checkpoint.get('val_acc', 0.0),
            'trained_epoch': checkpoint.get('epoch', 0),
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': '4分类' if self.num_classes == 4 else '2分类',
        }

        # 初始化可解释性分析器
        self._explainability = ExplainabilityAnalyzer(self._model, self._device)
        self._risk_assessor = RiskAssessmentReport()

        logger.info(f"模型加载成功：{self.model_info['name']} ({self.model_info['mode']})")
        logger.info(f"验证准确率：{self.model_info['val_accuracy']:.2f}%")

    def _detect_num_classes(self) -> int:
        """检测模型权重中的类别数"""
        try:
            checkpoint = torch.load(self._model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(self._model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 查找最后一个全连接层
        for key in ['head.fc.weight', 'head.fc.bias', 'classifier.weight', 'classifier.bias',
                     'head.weight', 'head.bias']:
            if key in state_dict:
                return state_dict[key].shape[0]

        # 遍历查找
        for key, tensor in state_dict.items():
            if tensor.dim() == 1 and tensor.shape[0] in (2, 4):
                return tensor.shape[0]

        logger.warning("无法自动检测类别数，默认使用2分类")
        return 2

    def predict(self, image: Union[Image.Image, str, Path],
                enable_explainability: bool = True) -> Dict:
        """
        单张图片预测 (支持可解释性分析)

        Args:
            image: PIL Image / 路径
            enable_explainability: 是否启用可解释性分析
        """
        try:
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"图片不存在：{image}")
                image = Image.open(image_path).convert('RGB')
            elif isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                raise ValueError("不支持的图片类型")

            image_tensor = self._transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                outputs = self._model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            pred_idx = predicted.item()
            pred_class = self.idx_to_class[pred_idx]
            pred_config = self.class_config[pred_class]

            prob_dict = {}
            for cls_name, cfg in self.class_config.items():
                prob_dict[cfg['name_cn']] = float(probabilities[0][cfg['idx']].item())

            suggestion = self._get_suggestion(pred_class)

            result = {
                'success': True,
                'prediction': pred_class,
                'class_name': pred_config['name_cn'],
                'confidence': float(confidence.item()),
                'probabilities': prob_dict,
                'severity': pred_config['severity'],
                'color': pred_config['color'],
                'suggestion': suggestion,
                'timestamp': datetime.now().isoformat(),
            }

            # 可解释性分析
            if enable_explainability and self._explainability:
                try:
                    logger.info("开始可解释性分析...")
                    explain_report = self._explainability.analyze(
                        image=image,
                        image_tensor=image_tensor,
                        prediction_class=pred_class,
                        pred_idx=pred_idx,
                        confidence=float(confidence.item()),
                        probabilities=prob_dict,
                    )
                    result['explainability'] = explain_report
                    result['heatmap_overlay_bytes'] = explain_report.get('heatmap_overlay_bytes')
                    logger.info(f"可解释性分析完成: indicators={len(explain_report.get('indicators', []))}, regions={len(explain_report.get('regions', []))}")

                    # 生成风险评估报告
                    risk_report = self._risk_assessor.generate(
                        prediction_result=result,
                        explainability_report=explain_report
                    )
                    result['risk_report'] = risk_report
                except Exception as e:
                    logger.error(f"可解释性分析失败: {e}", exc_info=True)
                    # 构建最小化解释报告，确保 UI 不报空
                    result['explainability'] = {
                        'indicators': [],
                        'regions': [],
                        'decision_path': {'steps': [], 'explanation': f'AI已完成{pred_config["name_cn"]}诊断'},
                        'abnormal_indicator_count': 0,
                        'total_indicators': 0,
                    }
                    result['risk_report'] = None

            logger.info(f"预测完成：{result['class_name']} (置信度：{result['confidence']:.2%})")
            return result

        except Exception as e:
            logger.error(f"预测失败：{str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """批量预测"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            if isinstance(img_path, (str, Path)):
                result['image_path'] = str(img_path)
            results.append(result)
        return results

    def _get_suggestion(self, prediction_class: str) -> str:
        suggestions = {
            'Normal': '角膜形态正常，符合手术条件',
            'Mild KC': '轻度异常，建议进一步检查并密切随访',
            'Moderate KC': '中度圆锥角膜，不建议激光手术，建议角膜交联术',
            'Severe KC': '重度圆锥角膜，需角膜移植评估',
            'KC': '异常，建议进一步检查',
        }
        return suggestions.get(prediction_class, '请咨询眼科医生')

    def get_model_info(self) -> Dict:
        return {
            'success': True,
            'model': self.model_info,
            'device': str(self._device),
            'num_classes': self.num_classes,
            'class_names': {k: v['name_cn'] for k, v in self.class_config.items()},
            'timestamp': datetime.now().isoformat()
        }

    def health_check(self) -> Dict:
        return {
            'success': True,
            'status': 'healthy',
            'model_loaded': self._model is not None,
            'device': str(self._device),
            'version': '2.0.0',
            'timestamp': datetime.now().isoformat()
        }


_service_instance = None

def get_model_service() -> ModelService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelService()
    return _service_instance

def initialize_service():
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelService()
    return _service_instance
