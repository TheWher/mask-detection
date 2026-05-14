"""
训练结束后自动生成完整报告图表：
  1. 数据集样本图    2. 模型结构图
  3. 训练曲线        4. 混淆矩阵

用法：
    python report.py                          # 使用当前模型和日志
    python report.py --model models/custom.h5 # 指定模型
"""

import os
import sys
import math
import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (MODEL_PATH, TRAIN_DIR, IMG_SIZE, CLASSES_LIST,
                    CLASS_NAMES, SAVE_DIR, LOG_DIR)
from utils.logger import get_logger

logger = get_logger()

VAL_SPLIT = 0.2
BATCH_SIZE = 16
REPORT_DIR = SAVE_DIR
os.makedirs(REPORT_DIR, exist_ok=True)


# ═══════════════ 1. 数据集样本图 ═══════════════

def plot_dataset_samples(save_path):
    """从每类随机抽取 9 张样本，排列成网格"""
    samples_per_class = 9
    n_classes = len(CLASSES_LIST)
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(14, 4))
    fig.suptitle('Dataset Samples', fontsize=15, fontweight='bold', y=1.02)

    for row, cls_name in enumerate(CLASSES_LIST):
        cls_dir = os.path.join(TRAIN_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted([f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        picked = rng.choice(files, min(samples_per_class, len(files)), replace=False)

        for col, fname in enumerate(picked):
            img = plt.imread(os.path.join(cls_dir, fname))
            axes[row, col].imshow(img)
            axes[row, col].axis('off')

        axes[row, 0].set_ylabel(cls_name, fontsize=12, fontweight='bold',
                                 rotation=0, labelpad=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"样本图: {save_path}")


# ═══════════════ 2. 模型结构图 ═══════════════

def plot_model_architecture(model, save_path):
    """绘制 CNN 层结构示意图（手工绘制，不依赖 graphviz）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Model Architecture — MaskClassifier', fontsize=14, fontweight='bold')

    layers_data = []
    for layer in model.layers:
        name = layer.name
        cls_name = layer.__class__.__name__
        if hasattr(layer, 'filters'):
            detail = f"{cls_name}\n{layer.filters}@{layer.kernel_size[0]}x{layer.kernel_size[1]}"
        elif cls_name == 'MaxPooling2D':
            detail = f"MaxPool\n{layer.pool_size[0]}x{layer.pool_size[1]}"
        elif cls_name == 'GlobalAveragePooling2D':
            detail = "GAP"
        elif cls_name == 'Flatten':
            detail = "Flatten"
        elif cls_name == 'Dense':
            detail = f"Dense\n{layer.units}"
        elif cls_name == 'Dropout':
            detail = f"Dropout\n{layer.rate:.0%}"
        elif cls_name == 'BatchNormalization':
            detail = "BatchNorm"
        elif cls_name == 'InputLayer':
            detail = f"Input\n{layer.input_shape[0][1]}x{layer.input_shape[0][2]}x{layer.input_shape[0][3]}"
        else:
            detail = cls_name
        layers_data.append(detail)

    n = len(layers_data)
    for i, text in enumerate(layers_data):
        x = 1 + i * (8.0 / max(n - 1, 1))
        y = 3
        # 框
        rect = FancyBboxPatch((x - 0.7, y - 0.55), 1.4, 1.1,
                              boxstyle="round,pad=0.1",
                              facecolor='#e8f0fe', edgecolor='#1967d2',
                              linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7,
                fontfamily='monospace')
        # 箭头
        if i < n - 1:
            ax.annotate('', xy=(x + 0.7, y), xytext=(x + 0.75, y),
                        arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

    # 参数统计（load_model(compile=False) 后 trainable_weights 为空，用 count_params）
    total = model.count_params()
    ax.text(5, 0.5, f'Total params: {total:,}', ha='center', fontsize=10, color='#555')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"模型结构图: {save_path}")


# ═══════════════ 3. 训练曲线 ═══════════════

def plot_training_curves(history, save_path):
    """准确率 + 损失双曲线"""
    if history is None:
        logger.warn("无训练历史数据，跳过曲线")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.history['loss']) + 1)
    best_epoch = int(np.argmax(history.history['val_accuracy']))

    # 准确率
    ax1.plot(epochs, history.history['accuracy'], 'b-', label='Train', linewidth=1.5)
    ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Val', linewidth=1.5)
    ax1.axvline(best_epoch + 1, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate(f'Best: {history.history["val_accuracy"][best_epoch]:.2%}',
                 xy=(best_epoch + 1, history.history['val_accuracy'][best_epoch]),
                 fontsize=9, color='red')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 损失
    ax2.plot(epochs, history.history['loss'], 'b-', label='Train', linewidth=1.5)
    ax2.plot(epochs, history.history['val_loss'], 'r-', label='Val', linewidth=1.5)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"训练曲线: {save_path}")


# ═══════════════ 4. 混淆矩阵 ═══════════════

def plot_confusion_matrix(model, save_path):
    """在验证集上生成混淆矩阵（Counts + Normalised 双图）"""
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VAL_SPLIT,
    )
    val_gen = val_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation',
        shuffle=False, seed=42,
    )

    y_true, y_pred = [], []
    n_samples = val_gen.samples
    for i in range(len(val_gen)):
        x_batch, y_batch = val_gen[i]
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = np.zeros((len(CLASSES_LIST), len(CLASSES_LIST)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    title_fmt = [('Counts', 'd'), ('Normalised', '.2%')]
    data_list = [cm, cm_norm]
    class_names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]

    for ax, data, (title, fmt) in zip([ax1, ax2], data_list, title_fmt):
        im = ax.imshow(data, cmap=plt.cm.Blues, vmin=0, vmax=data.max() if fmt == 'd' else 1.0)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        thresh = data.max() / 2
        for i, j in itertools.product(range(len(class_names)), range(len(class_names))):
            ax.text(j, i, fmt.format(data[i, j]),
                    ha='center', va='center',
                    color='white' if data[i, j] > thresh else 'black',
                    fontsize=13)

    acc = cm.diagonal().sum() / cm.sum()
    plt.suptitle(f'Confusion Matrix — Accuracy {acc:.2%}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"混淆矩阵: {save_path}")


# ═══════════════ 主入口 ═══════════════

def generate_report(model_path=None, history=None):
    """一键生成全部图表"""
    model_p = model_path or MODEL_PATH
    if not os.path.exists(model_p):
        logger.error(f"模型不存在: {model_p}")
        return

    logger.section("生成完整报告图表")

    # 加载模型
    from tensorflow.keras.models import load_model
    model = load_model(model_p, compile=False)
    logger.info(f"模型: {model_p} ({model.count_params():,} params)")

    # 图1: 数据集样本
    logger.info("[1/4] 数据集样本网格...")
    plot_dataset_samples(os.path.join(REPORT_DIR, 'report_samples.png'))

    # 图2: 模型结构
    logger.info("[2/4] 模型结构图...")
    plot_model_architecture(model, os.path.join(REPORT_DIR, 'report_architecture.png'))

    # 图3: 训练曲线
    logger.info("[3/4] 训练曲线...")
    train_curve = os.path.join(REPORT_DIR, 'training_curve.png')
    if history is not None:
        plot_training_curves(history, train_curve)
    elif os.path.exists(train_curve):
        logger.info(f"训练曲线已存在: {train_curve}")

    # 图4: 混淆矩阵
    logger.info("[4/4] 混淆矩阵...")
    plot_confusion_matrix(model, os.path.join(REPORT_DIR, 'report_confusion_matrix.png'))

    logger.section("报告生成完毕")
    for f in ['report_samples.png', 'report_architecture.png',
              'training_curve.png', 'report_confusion_matrix.png']:
        p = os.path.join(REPORT_DIR, f)
        if os.path.exists(p):
            logger.result(f"  {f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成完整训练报告图表')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='模型路径')
    args = parser.parse_args()
    generate_report(model_path=args.model)
