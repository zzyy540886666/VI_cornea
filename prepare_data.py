"""
2026 年最新角膜地形图分类模型 - 数据准备脚本
功能：自动划分训练集、验证集、测试集
"""

import os
import shutil
import random
from pathlib import Path

def prepare_dataset(data_dir="data", output_dir="dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    准备数据集，按比例划分训练集、验证集和测试集
    
    参数:
        data_dir: 原始数据目录（包含 kc 和 normal 文件夹）
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    
    # 设置随机种子，保证结果可复现
    random.seed(42)
    
    # 创建输出目录
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        for class_name in ['kc', 'normal']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # 处理每个类别
    classes = ['kc', 'normal']
    class_names = {
        'kc': '圆锥角膜（异常）',
        'normal': '正常角膜'
    }
    
    print("=" * 60)
    print("2026 角膜地形图数据集准备")
    print("=" * 60)
    
    for class_name in classes:
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"❌ 找不到文件夹：{class_dir}")
            continue
        
        # 获取所有图片
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        # 划分数据集
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        print(f"\n📁 类别：{class_names[class_name]} ({class_name})")
        print(f"   总图片数：{total}")
        
        # 复制到对应目录
        for split_name, img_list in splits.items():
            for img in img_list:
                dst = output_path / split_name / class_name / img.name
                shutil.copy2(img, dst)
            
            print(f"   {split_name}: {len(img_list)} 张")
    
    print("\n" + "=" * 60)
    print("✅ 数据集准备完成！")
    print(f"📂 输出目录：{output_dir}")
    print("=" * 60)
    
    # 统计信息
    print("\n📊 数据集统计：")
    for split in ['train', 'val', 'test']:
        split_path = output_path / split
        kc_count = len(list((split_path / 'kc').glob("*.jpg"))) + len(list((split_path / 'kc').glob("*.png")))
        normal_count = len(list((split_path / 'normal').glob("*.jpg"))) + len(list((split_path / 'normal').glob("*.png")))
        print(f"   {split}: {kc_count + normal_count} 张 (KC: {kc_count}, Normal: {normal_count})")

if __name__ == "__main__":
    prepare_dataset()
    print("\n下一步：运行 train.py 开始训练模型")
