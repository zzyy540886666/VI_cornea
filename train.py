"""
角膜地形图智能诊断系统 - 训练脚本 (v2.0 增强版)
支持: 4分类 / 多模型集成 / 医学影像专用增强 / MixUp / 类别权重
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from timm import create_model
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from collections import Counter
import random

# ══════════════════════════════════════════
#  数据集
# ══════════════════════════════════════════

class CornealDataset(torch.utils.data.Dataset):
    """角膜地形图数据集 - 支持2分类和4分类"""

    def __init__(self, root_dir, transform=None, num_classes=4):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_classes = num_classes
        self.samples = []

        if num_classes == 4:
            self.class_to_idx = {'normal': 0, 'mild_kc': 1, 'moderate_kc': 2, 'severe_kc': 3}
        else:
            self.class_to_idx = {'kc': 0, 'normal': 1}

        for class_name, idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")):
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_distribution(self):
        labels = [s[1] for s in self.samples]
        return Counter(labels)


# ══════════════════════════════════════════
#  MixUp 数据增强
# ══════════════════════════════════════════

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════
#  训练器
# ══════════════════════════════════════════

class Trainer:
    """训练器 v2.0 - 支持4分类 + MixUp + 类别权重"""

    def __init__(self, model_name='convnextv2_base', num_classes=4, device=None,
                 use_mixup=True, mixup_alpha=0.2):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.use_mixup = use_mixup and num_classes == 4  # 4分类时启用MixUp
        self.mixup_alpha = mixup_alpha

        print(f"使用设备：{self.device}")
        print(f"模型：{model_name} | 类别数：{num_classes} | MixUp: {self.use_mixup}")

        self.model = create_model(
            model_name, pretrained=True, num_classes=num_classes, in_chans=3
        )
        self.model = self.model.to(self.device)

        self.input_size = 256 if 'swin' in model_name.lower() else 224

        # 医学影像专用数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 损失函数（类别权重在加载数据后设置）
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.05)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        print(f"参数量：{sum(p.numel() for p in self.model.parameters()):,}")

    def load_data(self, data_dir='dataset', batch_size=16):
        train_dir = Path(data_dir) / 'train'
        val_dir = Path(data_dir) / 'val'
        test_dir = Path(data_dir) / 'test'

        train_dataset = CornealDataset(train_dir, transform=self.train_transform, num_classes=self.num_classes)
        val_dataset = CornealDataset(val_dir, transform=self.val_transform, num_classes=self.num_classes)
        test_dataset = CornealDataset(test_dir, transform=self.val_transform, num_classes=self.num_classes)

        # 计算类别权重
        dist = train_dataset.get_class_distribution()
        total = sum(dist.values())
        class_weights = torch.tensor(
            [total / (self.num_classes * dist.get(i, 1)) for i in range(self.num_classes)],
            dtype=torch.float32
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"类别权重: {class_weights.cpu().numpy()}")

        # 加权采样器（处理类别不平衡）
        sample_weights = [class_weights[label].item() for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        print(f"\n数据集统计：")
        print(f"  训练集：{len(train_dataset)} 张")
        print(f"  验证集：{len(val_dataset)} 张")
        print(f"  测试集：{len(test_dataset)} 张")
        print(f"  类别分布(训练): {dict(dist)}")

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)

            if self.use_mixup and random.random() > 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, targets, self.mixup_alpha)
                outputs = self.model(images)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))
            acc1, = accuracy(outputs, targets, topk=(1,))
            top1.update(acc1.item(), images.size(0))

            pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{top1.avg:.2f}%'})

        return losses.avg, top1.avg

    def validate(self, loader, desc='Validation'):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        all_preds, all_targets = [], []

        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                losses.update(loss.item(), images.size(0))
                acc1, = accuracy(outputs, targets, topk=(1,))
                top1.update(acc1.item(), images.size(0))

                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{top1.avg:.2f}%'})

        return losses.avg, top1.avg, all_preds, all_targets

    def train(self, epochs=50, save_dir='checkpoints'):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        print("\n" + "=" * 60)
        print(f"开始训练 ({self.num_classes}分类, {'MixUp' if self.use_mixup else '标准'})")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, _, _ = self.validate(self.val_loader, desc=f'Val Epoch {epoch}')

            self.scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'num_classes': self.num_classes,
                }, save_path / 'best_model.pth')
                print(f"  * 保存最佳模型 (准确率：{val_acc:.2f}%)")

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc, 'num_classes': self.num_classes,
                }, save_path / f'checkpoint_epoch_{epoch}.pth')

        print(f"\n训练完成！最佳验证准确率：{best_acc:.2f}%")
        self.plot_history(history, save_path)
        return history

    def plot_history(self, history, save_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves'); ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=150)
        print(f"训练曲线已保存")

    def evaluate_test(self, model_path='checkpoints/best_model.pth'):
        print("\n" + "=" * 60)
        print("测试集评估")
        print("=" * 60)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"验证准确率：{checkpoint['val_acc']:.2f}%")

        test_loss, test_acc, all_preds, all_targets = self.validate(self.test_loader, desc='Testing')

        if self.num_classes == 4:
            class_names = ['Normal', 'Mild KC', 'Moderate KC', 'Severe KC']
        else:
            class_names = ['KC', 'Normal']

        print("\n分类报告:")
        print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix'); plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red', fontsize=12)
        plt.tight_layout()
        plt.savefig('checkpoints/confusion_matrix.png', dpi=150)
        print("混淆矩阵已保存")

        return test_acc, all_preds, all_targets


# ══════════════════════════════════════════
#  集成模型训练
# ══════════════════════════════════════════

class EnsembleTrainer:
    """多模型集成训练器"""

    MODEL_CONFIGS = [
        {'name': 'convnextv2_base', 'size': 'large'},
        {'name': 'tf_efficientnetv2_m', 'size': 'medium'},
        {'name': 'maxvit_tiny_tf_224', 'size': 'small'},
    ]

    def __init__(self, num_classes=4, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.trainers = []

    def train_all(self, data_dir='dataset', epochs=30, save_dir='checkpoints'):
        results = {}
        for config in self.MODEL_CONFIGS:
            name = config['name']
            print(f"\n{'='*60}")
            print(f"训练模型: {name}")
            print(f"{'='*60}")
            trainer = Trainer(model_name=name, num_classes=self.num_classes, device=self.device)
            trainer.load_data(data_dir=data_dir)
            history = trainer.train(epochs=epochs, save_dir=save_dir)
            model_path = Path(save_dir) / f'best_model_{name}.pth'
            torch.save({
                'epoch': len(history['val_acc']),
                'model_state_dict': trainer.model.state_dict(),
                'val_acc': max(history['val_acc']),
                'model_name': name,
                'num_classes': self.num_classes,
            }, model_path)
            results[name] = {'val_acc': max(history['val_acc']), 'path': str(model_path)}
            self.trainers.append(trainer)

        print("\n集成训练结果:")
        for name, r in results.items():
            print(f"  {name}: {r['val_acc']:.2f}%")
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='角膜地形图分类模型训练 v2.0')
    parser.add_argument('--mode', choices=['single', 'ensemble'], default='single')
    parser.add_argument('--num_classes', type=int, default=4, choices=[2, 4])
    parser.add_argument('--model', type=str, default='convnextv2_base')
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_mixup', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print(f"角膜地形图分类模型训练 v2.0 ({args.num_classes}分类)")
    print("=" * 60)

    if args.mode == 'ensemble':
        et = EnsembleTrainer(num_classes=args.num_classes)
        et.train_all(data_dir=args.data_dir, epochs=args.epochs)
    else:
        trainer = Trainer(
            model_name=args.model,
            num_classes=args.num_classes,
            use_mixup=not args.no_mixup
        )
        trainer.load_data(data_dir=args.data_dir, batch_size=args.batch_size)
        trainer.train(epochs=args.epochs)
        trainer.evaluate_test('checkpoints/best_model.pth')

    print("\n训练完成!")


if __name__ == "__main__":
    main()
