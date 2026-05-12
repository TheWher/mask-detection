"""
口罩佩戴检测 — 模型训练脚本

功能：
  1. 加载数据集（支持合成数据和真实数据）
  2. 数据增强（提高泛化能力）
  3. 训练 CNN 分类模型
  4. EarlyStopping + ReduceLROnPlateau 防过拟合
  5. 保存最佳模型 + 训练曲线

用法：
    python train.py
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_mask_classifier, compile_model
from utils.logger import get_logger

# ==================== 配置参数 ====================
IMG_SIZE = (224, 224)           # 统一 224x224 输入
BATCH_SIZE = 16                 # 批次大小（224 图像更大，减小批次）
EPOCHS = 15                     # 最大训练轮数（早停机制会自动提前结束）
LEARNING_RATE = 0.001           # 初始学习率

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
SAVE_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, 'mask_classifier.h5')
PLOT_PATH = os.path.join(SAVE_DIR, 'training_curve.png')

# 初始化日志
logger = get_logger(os.path.join(LOG_DIR, 'train.log'))


def check_dataset():
    """检查数据集是否存在并打印统计信息"""
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        logger.error("数据集目录不存在！")
        logger.info("请先运行: python dataset/prepare_dataset.py")
        sys.exit(1)

    logger.info("数据集统计：")
    for cls in ['with_mask', 'without_mask']:
        train_count = len(os.listdir(os.path.join(TRAIN_DIR, cls)))
        test_count = len(os.listdir(os.path.join(TEST_DIR, cls)))
        logger.info(f"  {cls}: 训练 {train_count} 张 / 测试 {test_count} 张")


def create_data_generators():
    """
    创建训练和验证数据生成器
    - 训练集：随机增强（旋转、平移、翻转、亮度调整）
    - 测试集：仅归一化
    """
    # 训练数据增强（针对小数据集做了适度增强）
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,             # 随机旋转 ±20°
        width_shift_range=0.15,        # 水平平移 15%
        height_shift_range=0.15,       # 垂直平移 15%
        zoom_range=0.15,               # 随机缩放 15%
        horizontal_flip=True,          # 水平翻转
        brightness_range=[0.8, 1.2],   # 亮度调整
        fill_mode='nearest'
    )

    # 测试数据仅归一化
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    logger.info(f"类别索引: {train_generator.class_indices}")
    logger.info(f"训练批次: {len(train_generator)} | 验证批次: {len(test_generator)}")
    return train_generator, test_generator


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


def train():
    """主训练流程"""
    logger.section("口罩佩戴检测 - 模型训练")

    # 1. 检查数据集
    logger.info("[1/4] 检查数据集...")
    check_dataset()

    # 2. 加载数据
    logger.info("[2/4] 加载数据并配置增强...")
    train_gen, test_gen = create_data_generators()

    # 3. 构建模型
    logger.info("[3/4] 构建模型...")
    model = build_mask_classifier(input_shape=(*IMG_SIZE, 3))
    model = compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()

    # 4. 训练回调配置（早停 + 学习率衰减）
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # 验证损失 5 轮不降则提前停止，恢复最佳权重
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 验证损失 3 轮不降则学习率减半
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. 开始训练
    logger.info("[4/4] 开始训练...")
    logger.info(f"轮数: {EPOCHS} | 批次: {BATCH_SIZE} | 学习率: {LEARNING_RATE}")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks,
        verbose=1
    )

    # 6. 保存结果
    plot_training_history(history)

    # 最终评估
    final_acc = max(history.history['val_accuracy'])
    final_loss = min(history.history['val_loss'])
    actual_epochs = len(history.history['loss'])

    logger.section("训练完成")
    logger.result(f"实际训练轮数: {actual_epochs}/{EPOCHS}")
    logger.result(f"最佳验证准确率: {final_acc:.4f}")
    logger.result(f"最低验证损失: {final_loss:.4f}")
    logger.result(f"模型已保存至: {MODEL_PATH}")

    return model, history


if __name__ == '__main__':
    train()
