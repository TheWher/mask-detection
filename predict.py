"""
口罩佩戴检测 — 单张图片预测推理

用法：
    python predict.py <图片路径>
    python predict.py test_face.jpg

特性：
  - 异常图片容错（损坏文件、非图像文件友好提示）
  - 输出置信度 + 预测类别
"""

import os
import sys
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==================== 配置 ====================
IMG_SIZE = (224, 224)              # 与训练时保持一致
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mask_classifier.h5')

# 类别名称（flow_from_directory 按字母顺序分配: with_mask(0) -> without_mask(1)）
CLASS_NAMES = {0: '戴口罩', 1: '未戴口罩'}


def load_trained_model(path=None):
    """加载训练好的模型"""
    model_path = path or MODEL_PATH
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("[提示] 请先运行: python train.py")
        sys.exit(1)
    model = load_model(model_path)
    print(f"[成功] 模型已加载: {model_path}")
    return model


def preprocess_image(image_path):
    """
    预处理输入图片
    - 打开图片并转为 RGB
    - 缩放到模型输入尺寸
    - 归一化到 [0, 1]
    - 异常图片给出友好错误提示
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    # 检查文件扩展名
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    if not image_path.lower().endswith(valid_exts):
        print(f"[警告] 文件扩展名可能不支持: {image_path}")
        print(f"[提示] 支持的格式: {', '.join(valid_exts)}")

    try:
        img = Image.open(image_path).convert('RGB')
    except UnidentifiedImageError:
        raise ValueError(
            f"无法识别该图片文件: {image_path}\n"
            f"[提示] 文件可能已损坏或不是有效的图片格式"
        )
    except Exception as e:
        raise ValueError(f"打开图片时出错: {image_path}\n[详情] {e}")

    # 检查图片有效性（尺寸不为 0）
    if img.width == 0 or img.height == 0:
        raise ValueError(f"图片尺寸异常: {img.size}\n[提示] 文件可能已损坏")

    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


def predict(model, image_path):
    """
    对单张图片进行预测

    参数:
        model: 加载好的 Keras 模型
        image_path: 图片路径

    返回:
        (类别名称, 置信度, 原始概率)
    """
    img_array, _ = preprocess_image(image_path)

    try:
        prob = model.predict(img_array, verbose=0)[0][0]
    except Exception as e:
        raise RuntimeError(f"模型推理失败: {e}")

    if prob > 0.5:
        predicted_class = 1        # 不戴口罩
        confidence = prob
    else:
        predicted_class = 0        # 戴口罩
        confidence = 1 - prob

    class_name = CLASS_NAMES[predicted_class]
    return class_name, confidence, prob


def main():
    if len(sys.argv) < 2:
        print("=" * 45)
        print("  口罩佩戴检测 — 单张预测")
        print("=" * 45)
        print("用法: python predict.py <图片路径>")
        print("示例: python predict.py dataset/test/with_mask/with_mask_0001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        model = load_trained_model()
        class_name, confidence, raw_prob = predict(model, image_path)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)

    # 输出结果
    print("\n" + "=" * 45)
    print(f"  图片: {image_path}")
    print(f"  预测结果: {class_name}")
    print(f"  置信度: {confidence:.2%}")
    print(f"  原始概率(不戴口罩): {raw_prob:.4f}")
    print("=" * 45)


if __name__ == '__main__':
    main()
