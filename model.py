"""
口罩佩戴检测 CNN 模型定义
二分类任务：戴口罩(with_mask) vs 不戴口罩(without_mask)

设计思路：
  - 4层卷积 + BatchNormalization 加速收敛
  - 仅使用 64 神经元全连接层，控制参数量
  - Dropout 防过拟合，适合小数据集
  - 相比原方案（850万参数）减少约 60%
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_mask_classifier(input_shape=(224, 224, 3)):
    """
    构建轻量级 CNN 二分类模型

    参数:
        input_shape: 输入图像尺寸, 默认 224x224x3

    返回:
        未编译的 Keras Sequential 模型
    """
    model = models.Sequential(name="MaskClassifier")

    # ---- 第1个卷积块: 224 -> 112 ----
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # ---- 第2个卷积块: 112 -> 56 ----
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # ---- 第3个卷积块: 56 -> 28 ----
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # ---- 第4个卷积块: 28 -> 14 ----
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # ---- 分类头 ----
    model.add(layers.Flatten())
    # 14*14*128 = 25088 -> 64 神经元，大幅减少参数量
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def compile_model(model, learning_rate=0.001):
    """
    编译模型：配置损失函数、优化器与评估指标
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    m = build_mask_classifier()
    compile_model(m)
    m.summary()
