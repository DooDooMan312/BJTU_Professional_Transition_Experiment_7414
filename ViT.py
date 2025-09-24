# -*- coding: utf-8 -*-
# @Time    : 2025/5/13 19:58
# @Author  : XXX
# @Site    : 
# @File    : ViT.py
# @Software: PyCharm 
# @Comment :

# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings
import multiprocessing  # 新增多进程支持

warnings.filterwarnings("ignore")

# 配置参数
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 3
pretrain = True
epochs = 20
batch_size = 32  # 根据显存调整（建议16/32/64）
num_workers = min(4, os.cpu_count())  # 自动检测CPU核心数
model_save_path = './saved_models'
os.makedirs(model_save_path, exist_ok=True)

# 数据增强配置
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_datasets():
    """加载所有数据集"""
    train_dataset = datasets.ImageFolder('./recurrent_map_DAS_Res/train/', transform=train_transform)
    val_dataset = datasets.ImageFolder('./recurrent_map_DAS_Res/val/', transform=val_test_transform)
    test_dataset = datasets.ImageFolder('./recurrent_map_DAS_Res/test/', transform=val_test_transform)
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_ds, val_ds, test_ds):
    """创建数据加载器"""
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True)
    )


def create_model():
    """创建并返回Vision Transformer模型"""
    # 使用预训练的ViT模型
    model = models.vision_transformer.vit_b_16(pretrained=pretrain)
    if pretrain:
        for param in model.parameters():
            param.requires_grad = False
    # model.heads = nn.Linear(model.heads.in_features, n_classes)  # 修改输出层以适应3类问题
    # Access the last Linear layer in the 'heads' block
    num_ftrs = model.heads[-1].in_features  # Get the in_features of the last layer in heads
    model.heads = nn.Linear(num_ftrs, n_classes)  # Modify the output layer to match n_classes

    return model.to(device)



def train_epoch(model, loader, optimizer, criterion, scaler):
    """单轮训练"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    elapsed = time.time() - start_time
    return epoch_loss, accuracy, elapsed


def evaluate(model, loader, criterion):
    """模型评估"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    return epoch_loss, accuracy, all_labels, all_preds


def save_loss_acc_to_csv(epoch, train_loss, train_acc, val_loss, val_acc, elapsed_time, model_save):
    """保存训练指标到CSV"""
    file_path = os.path.join(model_save, 'training_metrics_ViT.csv')
    new_row = pd.DataFrame({
        'epoch': [epoch],
        'train_loss': [train_loss],
        'train_acc': [train_acc],
        'val_loss': [val_loss],
        'val_acc': [val_acc],
        'time': [elapsed_time]
    })

    mode = 'a' if os.path.exists(file_path) else 'w'
    new_row.to_csv(file_path, mode=mode, header=(mode == 'w'), index=False)


def plot_confusion_matrix(cm, title, classes, normalize=False):
    """绘制混淆矩阵（优化版V2）"""
    plt.figure(figsize=(8, 8))
    # plt.xlim(-0.5, cm.shape[1] - 0.5)
    # plt.ylim(cm.shape[0] - 0.5, -0.5)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=22, pad=20)  # 增加标题与图形的间距

    # 设置colorbar字体大小
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20, pad=5)  # 调整colorbar间距

    tick_marks = np.arange(len(classes))

    # 设置刻度参数（内侧刻度）
    plt.tick_params(
        axis='x',
        which='both',
        direction='in',  # 刻度线朝内
        labelrotation=45,
        labelsize=18,
        pad=12  # 刻度标签与轴的距离
    )
    plt.tick_params(
        axis='y',
        which='both',
        direction='in',
        labelsize=18,
        pad=12
    )

    # 设置刻度标签
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # 处理归一化并转换数据类型
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm * 100).astype(int)  # 转换为百分比整数
        print("Normalized confusion matrix (percentage)")
    else:
        cm = cm.astype(int)
        print('Confusion matrix, without normalization')

    if cm.size > 0:
        thresh = cm.max() / 2.
    else:
        print("Error: Confusion matrix is empty!")
        return

    # 在单元格中心添加文本（优化对齐方式）
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j, i,  # 单元格中心坐标
            format(cm[i, j], 'd'),
            ha='center',  # 水平居中
            va='center',  # 垂直居中
            color="white" if cm[i, j] > thresh else "black",
            fontsize=20,
            fontweight='bold'  # 加粗文本
        )

    plt.xlabel('Predicted label', fontsize=22, labelpad=15)  # 增加标签间距
    plt.ylabel('True label', fontsize=22, labelpad=15)
    plt.tight_layout()
    plt.show()


def main():
    # 初始化组件
    train_ds, val_ds, test_ds = load_datasets()
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds)
    model = create_model()

    # 混合精度训练配置
    scaler = torch.cuda.amp.GradScaler()

    # 优化器配置
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(epochs):
        start_time = time.time()

        # 训练阶段
        train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, criterion, scaler)

        # 验证阶段
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model_ViT.pth'))

        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 保存CSV
        elapsed = time.time() - start_time
        save_loss_acc_to_csv(epoch + 1, train_loss, train_acc, val_loss, val_acc, elapsed, model_save_path)

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.6f}s")

    # 最终测试
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'best_model_ViT.pth')))
    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion)

    print(f"\nFinal Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Size: {total_params / 1e6:.2f}M parameters")
    # 生成混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    classes = ['Wind', 'Man-made', 'Excavation']
    plot_confusion_matrix(cm, 'ViT',classes=classes, normalize=False)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows系统必需
    main()
