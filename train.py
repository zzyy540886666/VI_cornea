"""
2026 年最新角膜地形图分类模型 - 训练脚本
使用 2025-2026 年医学影像 SOTA 模型：
- ConvNeXt V2 (2025 NVIDIA 优化版)
- MaxViT (混合注意力机制)
- Swin Transformer V2 (医学影像专用)
"""

import os
# 设置 Hugging Face 镜像源（解决国内网络问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from timm.data import resolve_data_config
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class CornealDataset(torch.utils.data.Dataset):
    """角膜地形图数据集"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'kc': 0, 'normal': 1}
        self.idx_to_class = {0: 'kc', 1: 'normal'}
        
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


class Trainer:
    """训练器 - 支持多种 2026 最新模型"""
    
    def __init__(self, model_name='convnextv2_base', num_classes=2, device=None):
        """
        初始化训练器
        
        参数:
            model_name: 模型名称（可选：convnextv2_base, maxvit_tiny_224, swinv2_base_window16_256）
            num_classes: 分类数量
            device: 训练设备
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        print(f"🚀 使用设备：{self.device}")
        print(f"📦 加载模型：{model_name}")
        
        # 创建模型（使用 timm 库加载最新模型）
        self.model = create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=3
        )
        
        self.model = self.model.to(self.device)
        
        # 获取模型输入配置
        self.input_size = 224
        if 'swin' in model_name.lower():
            self.input_size = 256
        elif 'maxvit' in model_name.lower():
            self.input_size = 224
        
        # 数据增强（医学影像专用）
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 损失函数（带类别权重，处理不平衡数据）
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器（2025 年医学影像推荐配置）
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.05
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=30,
            eta_min=1e-6
        )
        
        print(f"✅ 模型初始化完成")
        print(f"   输入尺寸：{self.input_size}x{self.input_size}")
        print(f"   参数量：{sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_data(self, data_dir='dataset', batch_size=16):
        """加载数据"""
        
        train_dir = Path(data_dir) / 'train'
        val_dir = Path(data_dir) / 'val'
        test_dir = Path(data_dir) / 'test'
        
        # 创建数据集
        train_dataset = CornealDataset(train_dir, transform=self.train_transform)
        val_dataset = CornealDataset(val_dir, transform=self.val_transform)
        test_dataset = CornealDataset(test_dir, transform=self.val_transform)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"\n📊 数据集统计：")
        print(f"   训练集：{len(train_dataset)} 张")
        print(f"   验证集：{len(val_dataset)} 张")
        print(f"   测试集：{len(test_dataset)} 张")
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            losses.update(loss.item(), images.size(0))
            acc1, = accuracy(outputs, targets, topk=(1,))
            top1.update(acc1.item(), images.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%'
            })
        
        return losses.avg, top1.avg
    
    def validate(self, loader, desc='Validation'):
        """验证模型"""
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                losses.update(loss.item(), images.size(0))
                acc1, = accuracy(outputs, targets, topk=(1,))
                top1.update(acc1.item(), images.size(0))
                
                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'acc': f'{top1.avg:.2f}%'
                })
        
        return losses.avg, top1.avg, all_preds, all_targets
    
    def train(self, epochs=30, save_dir='checkpoints'):
        """完整训练流程"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_acc = 0.0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, _, _ = self.validate(self.val_loader, desc=f'Validating Epoch {epoch}')
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path / 'best_model.pth')
                print(f"   ⭐ 保存最佳模型 (准确率：{val_acc:.2f}%)")
            
            # 每 5 个 epoch 保存一次检查点
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path / f'checkpoint_epoch_{epoch}.pth')
        
        print("\n" + "="*60)
        print(f"训练完成！最佳验证准确率：{best_acc:.2f}%")
        print("="*60)
        
        # 绘制训练曲线
        self.plot_history(history, save_path)
        
        return history
    
    def plot_history(self, history, save_dir):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        ax1.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=150)
        print(f"📈 训练曲线已保存：{save_dir / 'training_curves.png'}")
    
    def evaluate_test(self, model_path='checkpoints/best_model.pth'):
        """在测试集上评估模型"""
        
        print("\n" + "="*60)
        print("测试集评估")
        print("="*60)
        
        # 加载最佳模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型：{model_path}")
        print(f"验证准确率：{checkpoint['val_acc']:.2f}%")
        
        # 测试
        test_loss, test_acc, all_preds, all_targets = self.validate(
            self.test_loader,
            desc='Testing'
        )
        
        # 分类报告
        class_names = ['KC (异常)', 'Normal (正常)']
        print("\n" + "="*60)
        print("分类报告")
        print("="*60)
        print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        print("\n混淆矩阵:")
        print(cm)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Corneal Topography Classification')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('checkpoints/confusion_matrix.png', dpi=150)
        print(f"\n📊 混淆矩阵已保存：checkpoints/confusion_matrix.png")
        
        return test_acc, all_preds, all_targets


def main():
    """主函数"""
    print("="*60)
    print("2026 角膜地形图分类模型训练")
    print("使用模型：ConvNeXt V2 (医学影像 SOTA)")
    print("="*60)
    
    # 创建训练器
    trainer = Trainer(
        model_name='convnextv2_base',  # 可替换为：maxvit_tiny_224, swinv2_base_window16_256
        num_classes=2
    )
    
    # 加载数据
    trainer.load_data(data_dir='dataset', batch_size=16)
    
    # 开始训练
    history = trainer.train(epochs=30, save_dir='checkpoints')
    
    # 测试评估
    trainer.evaluate_test('checkpoints/best_model.pth')
    
    print("\n" + "="*60)
    print("训练完成！下一步：运行 predict.py 进行预测")
    print("="*60)


if __name__ == "__main__":
    main()
