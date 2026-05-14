"""
口罩佩戴检测 — 单张图片预测推理 (二分类: Mask / NoMask)

用法：
    python predict.py <图片路径>
    python predict.py test.jpg
"""

import os
import sys
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mask_classifier_binary.h5')

CLASS_NAMES = {0: 'Mask', 1: 'NoMask'}


def load_trained_model(path=None):
    model_path = path or MODEL_PATH
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("[提示] 请先运行: python train.py")
        sys.exit(1)
    return load_model(model_path, compile=False)


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
    except UnidentifiedImageError:
        raise ValueError(f"无法识别该图片文件: {image_path}\n[提示] 文件可能已损坏")
    if img.width == 0 or img.height == 0:
        raise ValueError(f"图片尺寸异常: {img.size}")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(model, image_path):
    img_array = preprocess_image(image_path)
    probs = model.predict(img_array, verbose=0)[0]
    cls_idx = int(probs.argmax())
    return CLASS_NAMES[cls_idx], float(probs[cls_idx]), \
        {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}


def main():
    if len(sys.argv) < 2:
        print("=" * 45)
        print("  口罩佩戴检测 — 单张预测 (二分类)")
        print("=" * 45)
        print("用法: python predict.py <图片路径>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_trained_model()

    try:
        class_name, confidence, all_probs = predict(model, image_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[错误] {e}")
        sys.exit(1)

    print("\n" + "=" * 45)
    print(f"  图片: {image_path}")
    print(f"  预测结果: {class_name}")
    print(f"  置信度: {confidence:.2%}")
    print(f"  各类概率:")
    for name, p in all_probs.items():
        bar = '█' * int(p * 20)
        print(f"    {name}: {p:.4f} {bar}")
    print("=" * 45)


if __name__ == '__main__':
    main()
