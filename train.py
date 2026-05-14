"""
口罩佩戴检测 — 模型训练脚本

功能：
  1. 加载数据集（支持合成数据和真实数据）
  2. 数据增强（提高泛化能力）
  3. 训练 CNN 分类模型
  4. EarlyStopping + ReduceLROnPlateau 防过拟合
  5. 保存最佳模型 + 训练曲线 + TensorBoard 日志

用法：
    python train.py
    python train.py --batch-size 32 --epochs 30 --lr 0.0005
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau, TensorBoard, CSVLogger)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_mask_classifier, compile_model
from utils.logger import get_logger
from config import IMG_SIZE, TRAIN_DIR, SAVE_DIR, LOG_DIR, MODEL_PATH, CLASSES_LIST

PLOT_PATH = os.path.join(SAVE_DIR, 'training_curve.png')
TB_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard')
CSV_LOG_PATH = os.path.join(SAVE_DIR, 'training_log.csv')

# 验证集切分比例
VAL_SPLIT = 0.2

# 初始化日志
logger = get_logger(os.path.join(LOG_DIR, 'train.log'))

def check_dataset():
    """检查数据集是否存在并打印统计信息"""
    if not os.path.exists(TRAIN_DIR):
        logger.error("数据集目录不存在！")
        logger.info(f"期望路径: {TRAIN_DIR}")
        logger.info("请先运行: python label_images.py 完成标注")
        sys.exit(1)

    logger.info("数据集统计：")
    total = 0
    for cls in CLASSES_LIST:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            logger.info(f"  {cls}: {count} 张")
            total += count
        else:
            logger.info(f"  {cls}: 目录不存在")
    logger.info(f"  总计: {total} 张")
    logger.info(f"  验证集比例: {VAL_SPLIT:.0%} (训练 {total * (1 - VAL_SPLIT):.0f} / 验证 {total * VAL_SPLIT:.0f})")


def create_data_generators(batch_size):
    """
    创建训练和验证数据生成器
    - 训练集：随机增强（旋转、平移、翻转、亮度调整）
    - 验证集：仅归一化（无增强，确保评估准确）
    seed=42 保证两个生成器的 VAL_SPLIT 切分完全一致
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=VAL_SPLIT,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VAL_SPLIT,
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42,
    )

    val_generator = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42,
    )

    logger.info(f"类别索引: {train_generator.class_indices}")
    logger.info(f"训练批次: {len(train_generator)} | 验证批次: {len(val_generator)}")
    return train_generator, val_generator


def compute_class_weights():
    """计算类别权重，用 sqrt 缩放 + 上限 3.0，避免极端权重"""
    counts = {}
    for cls in CLASSES_LIST:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.exists(cls_dir):
            counts[cls] = len([f for f in os.listdir(cls_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            counts[cls] = 0

    total = sum(counts.values())
    # flow_from_directory 字母序: mask(0), nomask(1)
    sorted_classes = sorted(CLASSES_LIST)
    weights = {}
    for idx, cls in enumerate(sorted_classes):
        count = counts.get(cls, 0)
        if count > 0:
            # sqrt 缩放使极端权重更温和
            import math
            w = math.sqrt(total / count)
            w = min(w, 3.0)       # 上限 3.0
        else:
            w = 1.0
        weights[idx] = w

    logger.info(f"类别权重 (total={total}):")
    for idx, w in weights.items():
        logger.info(f"  [{idx}] {sorted_classes[idx]}: {counts[sorted_classes[idx]]} 张 → 权重 {w:.3f}")
    return weights


def plot_training_history(history):
    """绘制训练曲线（准确率 + 损失）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 准确率曲线
    ax1.plot(history.history['accuracy'], 'b-', label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], 'r-', label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 损失曲线
    ax2.plot(history.history['loss'], 'b-', label='Train Loss')
    ax2.plot(history.history['val_loss'], 'r-', label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    logger.info(f"训练曲线已保存至: {PLOT_PATH}")


def train(batch_size=16, epochs=15, learning_rate=0.001):
    """主训练流程"""
    logger.section("口罩佩戴检测 - 模型训练")

    # 1. 检查数据集
    logger.info("[1/5] 检查数据集...")
    check_dataset()

    # 2. 加载数据
    logger.info("[2/5] 加载数据并配置增强...")
    train_gen, val_gen = create_data_generators(batch_size)

    # 3. 构建模型
    logger.info("[3/5] 构建模型...")
    model = build_mask_classifier(input_shape=(*IMG_SIZE, 3))
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()

    # 4. 回调配置（早停 + 学习率衰减 + TensorBoard）
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(CSV_LOG_PATH),
    ]
    try:
        import tensorboard
        callbacks.append(TensorBoard(log_dir=TB_LOG_DIR, histogram_freq=1))
    except ImportError:
        logger.warn("TensorBoard 未安装，跳过。安装: pip install tensorboard")

    # 计算类别权重（缓解样本不均衡）
    class_weight = compute_class_weights()

    # 5. 开始训练
    logger.info("[5/5] 开始训练...")
    logger.info(f"轮数: {epochs} | 批次: {batch_size} | 学习率: {learning_rate}")
    logger.info(f"TensorBoard: tensorboard --logdir {TB_LOG_DIR}")

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # 6. 保存结果
    plot_training_history(history)

    final_acc = max(history.history['val_accuracy'])
    final_loss = min(history.history['val_loss'])
    actual_epochs = len(history.history['loss'])

    logger.section("训练完成")
    logger.result(f"实际训练轮数: {actual_epochs}/{epochs}")
    logger.result(f"最佳验证准确率: {final_acc:.4f}")
    logger.result(f"最低验证损失: {final_loss:.4f}")
    logger.result(f"模型已保存至: {MODEL_PATH}")

    # 7. 自动生成完整报告图表
    logger.info("[7/7] 生成报告图表...")
    try:
        from report import generate_report
        generate_report(model_path=MODEL_PATH, history=history)
    except Exception as e:
        logger.warn(f"报告生成失败: {e}")

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='口罩检测 — CNN 模型训练')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小 (默认: 16)')
    parser.add_argument('--epochs', type=int, default=15, help='最大训练轮数 (默认: 15)')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率 (默认: 0.001)')
    args = parser.parse_args()
    train(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr)
