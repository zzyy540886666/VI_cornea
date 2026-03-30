"""
2026 年角膜地形图分类模型 - 模型服务封装
提供统一的模型服务接口，支持 Web 和 API 调用
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from PIL import Image
from pathlib import Path
from typing import List, Dict, Union
import logging
from datetime import datetime
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelService:
    """模型服务类 - 单例模式"""
    
    _instance = None
    _model = None
    _device = None
    _transform = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化模型服务"""
        if self._model is None:
            self._initialize()
    
    def _initialize(self):
        """加载模型和配置"""
        logger.info("正在初始化模型服务...")
        
        # 设置设备
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备：{self._device}")
        
        # 加载模型
        self._model_path = Path('checkpoints/best_model.pth')
        if not self._model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{self._model_path}")
        
        # 创建模型
        self._model = create_model(
            'convnextv2_base',
            pretrained=False,
            num_classes=2,
            in_chans=3
        )
        
        # 加载权重
        checkpoint = torch.load(self._model_path, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()
        
        # 数据预处理
        self._transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 类别映射
        self.idx_to_class = {0: 'KC', 1: 'Normal'}
        self.class_names = {
            'KC': '圆锥角膜（异常）',
            'Normal': '正常角膜'
        }
        
        # 模型信息
        self.model_info = {
            'name': 'ConvNeXt V2 Base',
            'version': '1.0.0',
            'input_size': 224,
            'num_classes': 2,
            'val_accuracy': checkpoint.get('val_acc', 0.0),
            'trained_epoch': checkpoint.get('epoch', 0),
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"模型加载成功：{self.model_info['name']}")
        logger.info(f"验证准确率：{self.model_info['val_accuracy']:.2f}%")
    
    def predict(self, image: Union[Image.Image, str, Path]) -> Dict:
        """
        单张图片预测
        
        参数:
            image: PIL Image 对象，或图片路径字符串/Path 对象
            
        返回:
            dict: 包含预测结果、置信度、概率分布
        """
        try:
            # 加载图片
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"图片不存在：{image}")
                image = Image.open(image_path).convert('RGB')
            elif isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                raise ValueError("不支持的图片类型")
            
            # 预处理
            image_tensor = self._transform(image).unsqueeze(0).to(self._device)
            
            # 推理
            with torch.no_grad():
                outputs = self._model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # 结果
            pred_idx = predicted.item()
            pred_class = self.idx_to_class[pred_idx]
            
            result = {
                'success': True,
                'prediction': pred_class,
                'class_name': self.class_names[pred_class],
                'confidence': float(confidence.item()),
                'probabilities': {
                    self.class_names[self.idx_to_class[i]]: float(probabilities[0][i].item())
                    for i in range(2)
                },
                'suggestion': '建议进一步检查' if pred_class == 'KC' else '符合手术条件',
                'timestamp': datetime.now().isoformat()
            }
            
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
        """
        批量预测
        
        参数:
            image_paths: 图片路径列表
            
        返回:
            list: 所有图片的预测结果
        """
        logger.info(f"开始批量预测，共 {len(image_paths)} 张图片")
        
        results = []
        for i, img_path in enumerate(image_paths, 1):
            logger.info(f"处理第 {i}/{len(image_paths)} 张图片")
            result = self.predict(img_path)
            if isinstance(img_path, (str, Path)):
                result['image_path'] = str(img_path)
            results.append(result)
        
        # 统计
        kc_count = sum(1 for r in results if r.get('prediction') == 'KC')
        normal_count = sum(1 for r in results if r.get('prediction') == 'Normal')
        
        logger.info(f"批量预测完成：异常 {kc_count} 张，正常 {normal_count} 张")
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
            dict: 模型详细信息
        """
        return {
            'success': True,
            'model': self.model_info,
            'device': str(self._device),
            'timestamp': datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict:
        """
        健康检查
        
        返回:
            dict: 服务状态
        """
        return {
            'success': True,
            'status': 'healthy',
            'model_loaded': self._model is not None,
            'device': str(self._device),
            'timestamp': datetime.now().isoformat()
        }


# 全局服务实例
_service_instance = None

def get_model_service() -> ModelService:
    """获取模型服务单例"""
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelService()
    return _service_instance


def initialize_service():
    """预初始化服务（用于启动时加载）"""
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelService()
    return _service_instance
