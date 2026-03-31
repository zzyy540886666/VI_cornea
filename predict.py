"""
角膜地形图智能诊断系统 - 预测脚本 (v2.0 增强版)
支持: 4分类 / 可解释性分析 / 风险评估报告
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torchvision import transforms
from timm import create_model
from PIL import Image
from pathlib import Path
import numpy as np
import argparse
import json
from datetime import datetime


class CornealPredictor:
    """角膜地形图预测器 v2.0"""

    def __init__(self, model_path='checkpoints/best_model.pth', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

        # 检测类别数
        self.num_classes = self._detect_classes()
        if self.num_classes == 4:
            self.idx_to_class = {0: 'Normal', 1: 'Mild KC', 2: 'Moderate KC', 3: 'Severe KC'}
            self.class_names_cn = ['正常角膜', '轻度圆锥角膜', '中度圆锥角膜', '重度圆锥角膜']
        else:
            self.idx_to_class = {0: 'KC', 1: 'Normal'}
            self.class_names_cn = ['圆锥角膜（异常）', '正常角膜']

        print(f"设备：{self.device} | 类别数：{self.num_classes}")
        self._load_model()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 可解释性
        self.explainability = None
        if self.num_classes == 4:
            try:
                from explainability import ExplainabilityAnalyzer
                from risk_assessment import RiskAssessmentReport
                self.explainability = ExplainabilityAnalyzer(self.model, self.device)
                self.risk_assessor = RiskAssessmentReport()
                print("可解释性模块加载成功")
            except ImportError:
                print("可解释性模块未安装")

    def _detect_classes(self) -> int:
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        for key in ['head.fc.weight', 'head.fc.bias', 'classifier.weight', 'classifier.bias',
                     'head.weight', 'head.bias']:
            if key in state_dict:
                return state_dict[key].shape[0]
        for key, tensor in state_dict.items():
            if tensor.dim() == 1 and tensor.shape[0] in (2, 4):
                return tensor.shape[0]
        return 2

    def _load_model(self):
        print(f"加载模型：{self.model_path}")
        self.model = create_model('convnextv2_base', pretrained=False, num_classes=self.num_classes, in_chans=3)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"模型加载成功 | 验证准确率：{checkpoint['val_acc']:.2f}%")

    def predict(self, image_path, explain=False):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在：{image_path}")

        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        pred_idx = predicted.item()
        pred_class = self.idx_to_class[pred_idx]

        result = {
            'image_path': str(image_path),
            'prediction': pred_class,
            'class_name': self.class_names_cn[pred_idx],
            'confidence': confidence.item(),
            'probabilities': {self.class_names_cn[i]: float(probabilities[0][i].item()) for i in range(self.num_classes)},
            'timestamp': datetime.now().isoformat()
        }

        if explain and self.explainability:
            try:
                report = self.explainability.analyze(
                    image=image, image_tensor=image_tensor,
                    prediction_class=pred_class, pred_idx=pred_idx,
                    confidence=float(confidence.item()),
                    probabilities=result['probabilities']
                )
                result['explainability'] = {
                    'regions': report.get('regions', [])[:3],
                    'indicators': report.get('indicators', []),
                    'abnormal_count': report.get('abnormal_indicator_count', 0),
                    'total_indicators': report.get('total_indicators', 0),
                }
                risk = self.risk_assessor.generate(prediction_result=result, explainability_report=report)
                result['risk_report'] = risk
            except Exception as e:
                print(f"可解释性分析失败: {e}")

        return result

    def predict_batch(self, image_dir, explain=False):
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        print(f"发现 {len(image_files)} 张图片")

        results = []
        for img_path in image_files:
            try:
                r = self.predict(img_path, explain=explain)
                results.append(r)
                print(f"  {img_path.name}: {r['class_name']} ({r['confidence']:.1%})")
            except Exception as e:
                print(f"  {img_path.name}: 失败 - {e}")

        return results

    def print_report(self, result):
        """打印完整诊断报告"""
        print("\n" + "=" * 56)
        print("        角膜地形图AI辅助诊断报告")
        print("=" * 56)
        print(f"\n【诊断结果】")
        print(f"  分类: {result['class_name']}")
        print(f"  置信度: {result['confidence']:.1%}")
        print(f"  检查时间: {result['timestamp']}")

        print(f"\n【概率分布】")
        for name, prob in result['probabilities'].items():
            bar = "█" * int(prob * 20)
            print(f"  {name:12s} {bar} {prob:.1%}")

        if 'explainability' in result and result['explainability']:
            exp = result['explainability']
            if exp.get('indicators'):
                print(f"\n【临床指标】")
                for ind in exp['indicators']:
                    mark = "✅" if not ind['abnormal'] else "❌"
                    print(f"  {mark} {ind['name']}: {ind['value']} {ind['unit']} (正常: {ind['normal_range']}) - {ind['status']}")
                print(f"  异常指标: {exp.get('abnormal_count', 0)}/{exp.get('total_indicators', 0)}项")

            if exp.get('regions'):
                print(f"\n【关键区域】")
                for r in exp['regions'][:3]:
                    print(f"  • {r['region_type']} - 关注度: {r['avg_attention']:.2f} 严重度: {r['severity']}")

        if 'risk_report' in result and result['risk_report']:
            risk = result['risk_report']
            ra = risk.get('risk_assessment', {})
            sr = ra.get('surgery_risk', {})
            print(f"\n【风险评估】")
            print(f"  手术风险: {sr.get('level', '--')} - {sr.get('description', '--')}")
            print(f"  随访建议: {ra.get('follow_up', {}).get('interval', '--')}")

            print(f"\n【临床建议】")
            for i, rec in enumerate(risk.get('clinical_recommendations', []), 1):
                print(f"  {i}. {rec}")

        print("\n【免责声明】本报告仅供参考，最终诊断请以临床医生判断为准。")
        print("=" * 56)


def main():
    parser = argparse.ArgumentParser(description='角膜地形图预测 v2.0')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--dir', type=str, help='批量预测文件夹')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--explain', action='store_true', help='启用可解释性分析')
    parser.add_argument('--save_json', type=str, help='保存结果为JSON')
    args = parser.parse_args()

    print("=" * 60)
    print(f"角膜地形图智能诊断系统 v2.0 ({'4分类' if True else '2分类'})")
    print("=" * 60)

    predictor = CornealPredictor(model_path=args.model)

    if args.image:
        result = predictor.predict(args.image, explain=args.explain)
        predictor.print_report(result)
        if args.save_json:
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n结果已保存: {args.save_json}")

    elif args.dir:
        results = predictor.predict_batch(args.dir, explain=args.explain)
        if args.save_json:
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n结果已保存: {args.save_json}")
    else:
        print("\n使用示例:")
        print("  python predict.py --image data/kc/1.jpg --explain")
        print("  python predict.py --dir data/ --explain --save_json results.json")


if __name__ == "__main__":
    main()
