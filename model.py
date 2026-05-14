"""
口罩佩戴检测 CNN 模型定义 — 二分类 (Mask / NoMask)
4层卷积 + BatchNorm + GlobalAveragePooling + Dense(64) + softmax
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_mask_classifier(input_shape=(224, 224, 3), num_classes=2):
    """
    构建轻量级 CNN 二分类模型

    参数:
        input_shape: 输入图像尺寸, 默认 224x224x3
        num_classes: 分类数, 默认 2

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
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, learning_rate=0.001):
    """
    编译模型：配置损失函数、优化器与评估指标
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    m = build_mask_classifier()
    compile_model(m)
    m.summary()
