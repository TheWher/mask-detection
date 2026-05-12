"""
口罩佩戴检测 — 批量预测脚本

功能：
  1. 对单个目录下的所有图片进行批量预测
  2. 输出 CSV 格式的预测结果文件
  3. 显示统计摘要（戴口罩/未戴口罩数量）

用法：
    python batch_predict.py <图片目录> [--output result.csv]
    python batch_predict.py ./test_images/
    python batch_predict.py ./test_images/ --output predictions.csv
"""

import os
import sys
import argparse
import time
import csv
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==================== 配置 ====================
IMG_SIZE = (224, 224)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mask_classifier.h5')

CLASS_NAMES = {0: '戴口罩', 1: '未戴口罩'}


def load_trained_model():
    """加载训练好的模型"""
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        print("[提示] 请先运行: python train.py")
        sys.exit(1)
    print(f"[成功] 模型已加载: {MODEL_PATH}")
    return load_model(MODEL_PATH)


def collect_images(input_dir):
    """收集目录下所有图片文件"""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    images = []
    skipped = []

    for fname in sorted(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            images.append(fpath)
        else:
            skipped.append(fname)

    if skipped:
        print(f"[警告] 跳过 {len(skipped)} 个非图片文件: {skipped[:5]}...")

    return images


def preprocess_batch(image_paths):
    """批量预处理图片，返回 numpy 数组"""
    from PIL import Image

    batch = []
    valid_paths = []
    errors = []

    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            batch.append(arr)
            valid_paths.append(path)
        except Exception as e:
            errors.append((path, str(e)))

    if errors:
        print(f"[警告] {len(errors)} 张图片读取失败，已跳过")
        for p, err in errors[:3]:
            print(f"  - {os.path.basename(p)}: {err}")

    return np.array(batch), valid_paths, errors


def save_results(results, output_path):
    """保存预测结果为 CSV 文件"""
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['文件名', '预测类别', '置信度', '原始概率(不戴口罩)'])
        for r in results:
            fname = os.path.basename(r['path'])
            writer.writerow([
                fname, r['class'], f"{r['confidence']:.4f}", f"{r['raw_prob']:.4f}"
            ])
    print(f"\n[完成] 预测结果已保存至: {output_path}")


def print_statistics(results):
    """打印批量预测统计摘要"""
    total = len(results)
    with_mask = sum(1 for r in results if r['class'] == '戴口罩')
    without_mask = total - with_mask

    print("\n" + "=" * 45)
    print("  批量预测统计")
    print("=" * 45)
    print(f"  总图片数: {total}")
    print(f"  戴口罩:   {with_mask} 张 ({with_mask / total * 100:.1f}%)" if total else "")
    print(f"  未戴口罩: {without_mask} 张 ({without_mask / total * 100:.1f}%)" if total else "")
    print("=" * 45)


def main():
    parser = argparse.ArgumentParser(description='口罩佩戴检测 — 批量预测')
    parser.add_argument('input_dir', help='包含图片的目录路径')
    parser.add_argument('--output', '-o', default='batch_result.csv',
                        help='输出 CSV 文件路径 (默认: batch_result.csv)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[错误] 目录不存在: {args.input_dir}")
        sys.exit(1)

    # 1. 加载模型
    model = load_trained_model()

    # 2. 收集图片
    image_paths = collect_images(args.input_dir)
    if not image_paths:
        print("[错误] 目录中没有找到图片文件")
        sys.exit(1)
    print(f"[信息] 找到 {len(image_paths)} 张图片")

    # 3. 批量预处理
    batch, valid_paths, errors = preprocess_batch(image_paths)
    if len(batch) == 0:
        print("[错误] 所有图片都无法读取")
        sys.exit(1)

    # 4. 批量推理
    print(f"\n[信息] 正在对 {len(batch)} 张图片进行推理...")
    start_time = time.time()

    try:
        probs = model.predict(batch, verbose=0).flatten()
    except Exception as e:
        print(f"[错误] 模型推理失败: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # 5. 整理结果
    results = []
    for i, prob in enumerate(probs):
        predicted_class = 1 if prob > 0.5 else 0
        confidence = prob if prob > 0.5 else 1 - prob
        results.append({
            'path': valid_paths[i],
            'class': CLASS_NAMES[predicted_class],
            'confidence': float(confidence),
            'raw_prob': float(prob),
        })

    # 6. 保存 + 统计
    save_results(results, args.output)
    print_statistics(results)
    print(f"\n[耗时] 批量推理: {elapsed:.2f}s | 平均每张: {elapsed / len(batch):.3f}s")


if __name__ == '__main__':
    main()
