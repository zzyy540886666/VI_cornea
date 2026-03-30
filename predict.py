"""
2026 年最新角膜地形图分类模型 - 预测脚本
功能：加载训练好的模型，对新的角膜地形图进行分类预测
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import argparse

class CornealPredictor:
    """角膜地形图预测器"""
    
    def __init__(self, model_path='checkpoints/best_model.pth', device=None):
        """
        初始化预测器
        
        参数:
            model_path: 模型权重文件路径
            device: 推理设备
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.class_names = ['圆锥角膜（异常）', '正常角膜']
        self.idx_to_class = {0: 'KC', 1: 'Normal'}
        
        print(f"🚀 使用设备：{self.device}")
        
        # 加载模型
        self._load_model()
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_model(self):
        """加载训练好的模型"""
        print(f"📦 加载模型：{self.model_path}")
        
        # 创建模型（与训练时相同）
        self.model = create_model(
            'convnextv2_base',
            pretrained=False,
            num_classes=2,
            in_chans=3
        )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型加载成功")
        print(f"   验证准确率：{checkpoint['val_acc']:.2f}%")
        print(f"   训练轮次：{checkpoint['epoch']}")
    
    def predict(self, image_path):
        """
        预测单张图片
        
        参数:
            image_path: 图片路径
            
        返回:
            dict: 包含预测结果、置信度等信息
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在：{image_path}")
        
        # 读取图片
        image = Image.open(image_path).convert('RGB')
        
        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # 结果
        pred_class = predicted.item()
        conf_score = confidence.item()
        
        result = {
            'image_path': str(image_path),
            'prediction': self.idx_to_class[pred_class],
            'class_name': self.class_names[pred_class],
            'confidence': conf_score,
            'probabilities': probabilities.cpu().numpy()[0]
        }
        
        return result
    
    def predict_batch(self, image_dir):
        """
        批量预测文件夹中的所有图片
        
        参数:
            image_dir: 图片文件夹路径
            
        返回:
            list: 所有图片的预测结果
        """
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            raise FileNotFoundError(f"文件夹不存在：{image_dir}")
        
        # 获取所有图片
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"❌ 文件夹中没有图片：{image_dir}")
            return []
        
        print(f"📁 发现 {len(image_files)} 张图片，开始批量预测...")
        
        results = []
        for img_path in image_files:
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                print(f"⚠️ 预测失败 {img_path}: {e}")
        
        # 统计
        kc_count = sum(1 for r in results if r['prediction'] == 'KC')
        normal_count = sum(1 for r in results if r['prediction'] == 'Normal')
        
        print(f"\n📊 批量预测结果：")
        print(f"   总图片数：{len(results)}")
        print(f"   圆锥角膜（异常）: {kc_count} 张 ({kc_count/len(results)*100:.1f}%)")
        print(f"   正常角膜：{normal_count} 张 ({normal_count/len(results)*100:.1f}%)")
        
        return results
    
    def visualize(self, image_path, save_path=None):
        """
        可视化预测结果
        
        参数:
            image_path: 图片路径
            save_path: 保存路径（可选）
        """
        result = self.predict(image_path)
        
        # 读取图片
        image = Image.open(image_path)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 显示原图
        ax1.imshow(image)
        ax1.set_title(f'角膜地形图\n{Path(image_path).name}', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 显示预测结果
        classes = self.class_names
        probs = result['probabilities']
        colors = ['#FF6B6B' if result['prediction'] == 'KC' else '#4ECDC4',
                  '#4ECDC4' if result['prediction'] == 'KC' else '#FF6B6B']
        
        bars = ax2.barh(classes, probs, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('置信度', fontsize=12)
        ax2.set_title('预测结果', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', fontsize=12, fontweight='bold')
        
        # 添加总体判断
        judgment = "⚠️ 异常 - 建议进一步检查" if result['prediction'] == 'KC' else "✅ 正常 - 符合手术条件"
        judgment_color = 'red' if result['prediction'] == 'KC' else 'green'
        
        fig.suptitle(
            f'{result["class_name"]} (置信度：{result["confidence"]*100:.1f}%)\n{judgment}',
            fontsize=16,
            fontweight='bold',
            color=judgment_color,
            y=1.05
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 结果已保存：{save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='2026 角膜地形图分类预测')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--dir', type=str, help='批量预测文件夹路径')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='模型权重文件路径')
    parser.add_argument('--save', type=str, help='保存可视化结果的路径')
    
    args = parser.parse_args()
    
    print("="*60)
    print("2026 角膜地形图分类预测系统")
    print("="*60)
    
    # 创建预测器
    predictor = CornealPredictor(model_path=args.model)
    
    if args.image:
        # 单张图片预测
        print(f"\n📸 预测图片：{args.image}")
        result = predictor.predict(args.image)
        
        print(f"\n预测结果:")
        print(f"   类别：{result['class_name']}")
        print(f"   置信度：{result['confidence']*100:.2f}%")
        
        if args.save:
            predictor.visualize(args.image, save_path=args.save)
        else:
            predictor.visualize(args.image)
    
    elif args.dir:
        # 批量预测
        print(f"\n📁 批量预测文件夹：{args.dir}")
        results = predictor.predict_batch(args.dir)
        
        # 保存结果为 CSV
        if len(results) > 0:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
            print(f"\n📄 预测结果已保存：prediction_results.csv")
    
    else:
        # 交互式预测（测试用）
        print("\n💡 使用示例:")
        print("   python predict.py --image data/kc/1.jpg")
        print("   python predict.py --dir data/kc --save result.png")
        print("   python predict.py --image data/normal/1.jpg --model checkpoints/best_model.pth")
        
        # 测试一张图片
        test_image = Path('data/kc/1.jpg')
        if test_image.exists():
            print(f"\n🧪 测试图片：{test_image}")
            result = predictor.predict(test_image)
            print(f"   预测：{result['class_name']}")
            print(f"   置信度：{result['confidence']*100:.2f}%")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
