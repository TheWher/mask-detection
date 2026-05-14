"""
口罩佩戴检测 — 单张图片预测推理 (二分类: Mask / NoMask)

用法：
    python predict.py <图片路径>
    python predict.py test.jpg
"""

import sys
import numpy as np
from config import load_trained_model, preprocess_image, CLASS_NAMES


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
