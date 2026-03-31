"""
角膜地形图智能诊断系统 - 数据准备脚本
支持4分类数据集（normal / mild_kc / moderate_kc / severe_kc）
以及传统2分类数据集（normal / kc）
"""

import os
import shutil
import random
from pathlib import Path


def prepare_dataset_2class(data_dir="data", output_dir="dataset",
                           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    准备2分类数据集 (传统模式)
    数据结构: data/{kc, normal}/ -> dataset/{train,val,test}/{kc, normal}/
    """
    random.seed(42)
    output_path = Path(output_dir)

    for split in ['train', 'val', 'test']:
        for cls in ['kc', 'normal']:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    class_names = {'kc': '圆锥角膜', 'normal': '正常角膜'}
    print("=" * 60)
    print("角膜地形图数据集准备 (2分类)")
    print("=" * 60)

    for cls in ['kc', 'normal']:
        class_dir = Path(data_dir) / cls
        if not class_dir.exists():
            print(f"找不到文件夹: {class_dir}")
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        print(f"\n类别: {class_names[cls]} ({cls}) - 总计: {total} 张")
        for split_name, img_list in splits.items():
            for img in img_list:
                dst = output_path / split_name / cls / img.name
                shutil.copy2(img, dst)
            print(f"  {split_name}: {len(img_list)} 张")

    print("\n" + "=" * 60)
    print("数据集准备完成!")
    _print_stats(output_path)


def prepare_dataset_4class(data_dir="data_4class", output_dir="dataset_4class",
                           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    准备4分类数据集 (增强模式)
    数据结构: data_4class/{normal, mild_kc, moderate_kc, severe_kc}/
              -> dataset_4class/{train,val,test}/{normal, mild_kc, moderate_kc, severe_kc}/
    """
    random.seed(42)
    output_path = Path(output_dir)

    CLASSES = ['normal', 'mild_kc', 'moderate_kc', 'severe_kc']
    CLASS_NAMES = {
        'normal': '正常角膜',
        'mild_kc': '轻度圆锥角膜',
        'moderate_kc': '中度圆锥角膜',
        'severe_kc': '重度圆锥角膜'
    }

    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("角膜地形图数据集准备 (4分类)")
    print("=" * 60)

    for cls in CLASSES:
        class_dir = Path(data_dir) / cls
        if not class_dir.exists():
            print(f"找不到文件夹: {class_dir} (将跳过)")
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)
        total = len(images)

        if total == 0:
            print(f"\n类别: {CLASS_NAMES[cls]} ({cls}) - 无图片")
            continue

        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        print(f"\n类别: {CLASS_NAMES[cls]} ({cls}) - 总计: {total} 张")
        for split_name, img_list in splits.items():
            for img in img_list:
                dst = output_path / split_name / cls / img.name
                shutil.copy2(img, dst)
            print(f"  {split_name}: {len(img_list)} 张")

    print("\n" + "=" * 60)
    print("4分类数据集准备完成!")
    _print_stats_4class(output_path, CLASSES, CLASS_NAMES)


def split_kc_to_4class(data_dir="data", output_dir="data_4class",
                        mild_ratio=0.4, moderate_ratio=0.35, severe_ratio=0.25):
    """
    将2分类数据中的KC样本按比例随机分配到3个严重程度类别。
    注意：这是模拟分配，真实场景应由眼科医生标注。

    Args:
        data_dir: 原始数据目录 (含 kc/ 和 normal/)
        output_dir: 输出的4分类数据目录
        mild_ratio: 轻度占比
        moderate_ratio: 中度占比
        severe_ratio: 重度占比
    """
    random.seed(42)
    output_path = Path(output_dir)

    kc_dir = Path(data_dir) / 'kc'
    normal_dir = Path(data_dir) / 'normal'

    if not kc_dir.exists() or not normal_dir.exists():
        print(f"错误: 找不到原始数据目录 {data_dir}")
        return

    # 复制正常样本
    out_normal = output_path / 'normal'
    out_normal.mkdir(parents=True, exist_ok=True)
    for img in list(normal_dir.glob("*.jpg")) + list(normal_dir.glob("*.png")):
        shutil.copy2(img, out_normal / img.name)

    # 随机分配KC样本
    kc_images = list(kc_dir.glob("*.jpg")) + list(kc_dir.glob("*.png"))
    random.shuffle(kc_images)
    total = len(kc_images)

    mild_end = int(total * mild_ratio)
    moderate_end = mild_end + int(total * moderate_ratio)

    splits = {
        'mild_kc': kc_images[:mild_end],
        'moderate_kc': kc_images[mild_end:moderate_end],
        'severe_kc': kc_images[moderate_end:]
    }

    print("=" * 60)
    print("KC样本分配到3个严重程度类别 (模拟)")
    print("=" * 60)

    total_normal = len(list(out_normal.glob("*.jpg"))) + len(list(out_normal.glob("*.png")))
    print(f"\n正常样本: {total_normal} 张")

    for cls, imgs in splits.items():
        out_dir = output_path / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, out_dir / img.name)
        cls_names = {'mild_kc': '轻度', 'moderate_kc': '中度', 'severe_kc': '重度'}
        print(f"{cls_names[cls]}圆锥角膜: {len(imgs)} 张")

    print(f"\n总计: {total + total_normal} 张")
    print(f"输出目录: {output_path}")
    print("\n注意: 此为模拟分配，真实场景应由眼科医生根据临床标准标注")


def _print_stats(output_path: Path):
    print(f"\n数据集统计:")
    for split in ['train', 'val', 'test']:
        kc = len(list((output_path / split / 'kc').glob("*.jpg"))) + len(list((output_path / split / 'kc').glob("*.png")))
        normal = len(list((output_path / split / 'normal').glob("*.jpg"))) + len(list((output_path / split / 'normal').glob("*.png")))
        print(f"  {split}: {kc + normal} 张 (KC: {kc}, Normal: {normal})")


def _print_stats_4class(output_path: Path, classes, class_names):
    print(f"\n数据集统计:")
    for split in ['train', 'val', 'test']:
        counts = {}
        total = 0
        for cls in classes:
            cnt = len(list((output_path / split / cls).glob("*.jpg"))) + len(list((output_path / split / cls).glob("*.png")))
            counts[cls] = cnt
            total += cnt
        parts = [f"{class_names[c]}: {counts[c]}" for c in classes]
        print(f"  {split}: {total} 张 ({', '.join(parts)})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='角膜地形图数据集准备')
    parser.add_argument('--mode', choices=['2class', '4class', 'split_kc'], default='2class',
                        help='模式: 2class(传统) / 4class(4分类) / split_kc(拆分KC)')
    parser.add_argument('--data_dir', default='data', help='原始数据目录')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    args = parser.parse_args()

    if args.mode == '2class':
        prepare_dataset_2class(data_dir=args.data_dir)
    elif args.mode == '4class':
        out = args.output_dir or 'dataset_4class'
        prepare_dataset_4class(data_dir=args.data_dir, output_dir=out)
    elif args.mode == 'split_kc':
        out = args.output_dir or 'data_4class'
        split_kc_to_4class(data_dir=args.data_dir, output_dir=out)
