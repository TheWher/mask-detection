"""
口罩佩戴检测 — 批量预测 (二分类: Mask / NoMask)

用法：
    python batch_predict.py <图片目录> [--output result.csv]
"""

import os
import sys
import argparse
import time
import csv
import numpy as np
from PIL import Image
from config import load_trained_model, IMG_SIZE, CLASS_NAMES


def collect_images(input_dir):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    images = []
    for fname in sorted(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if os.path.splitext(fname)[1].lower() in valid_exts:
            images.append(fpath)
    return images


def preprocess_batch(image_paths):
    batch, valid = [], []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB').resize(IMG_SIZE)
            batch.append(np.array(img, dtype=np.float32) / 255.0)
            valid.append(path)
        except Exception:
            pass
    return np.array(batch), valid


def main():
    parser = argparse.ArgumentParser(description='口罩检测 — 批量预测')
    parser.add_argument('input_dir', help='包含图片的目录路径')
    parser.add_argument('--output', '-o', default='batch_result.csv', help='输出 CSV (默认: batch_result.csv)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[错误] 目录不存在: {args.input_dir}")
        sys.exit(1)

    model = load_trained_model()
    image_paths = collect_images(args.input_dir)
    if not image_paths:
        print("[错误] 目录中没有找到图片文件")
        sys.exit(1)
    print(f"[信息] 找到 {len(image_paths)} 张图片")

    batch, valid_paths = preprocess_batch(image_paths)
    if len(batch) == 0:
        print("[错误] 所有图片都无法读取")
        sys.exit(1)

    print(f"[信息] 正在推理 {len(batch)} 张...")
    t0 = time.time()
    probs = model.predict(batch, verbose=0)
    elapsed = time.time() - t0

    results = []
    mask_count = 0
    for i, row in enumerate(probs):
        cls_idx = int(row.argmax())
        if cls_idx == 0:
            mask_count += 1
        results.append({
            'path': valid_paths[i],
            'class': CLASS_NAMES[cls_idx],
            'confidence': float(row[cls_idx]),
            'mask_prob': float(row[0]),
            'nomask_prob': float(row[1]),
        })

    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['文件名', '预测', '置信度', 'Mask概率', 'NoMask概率'])
        for r in results:
            writer.writerow([os.path.basename(r['path']), r['class'],
                             f"{r['confidence']:.4f}", f"{r['mask_prob']:.4f}",
                             f"{r['nomask_prob']:.4f}"])

    total = len(results)
    print(f"\n  总图片: {total}  |  Mask: {mask_count} ({mask_count/total*100:.1f}%)  |  "
          f"NoMask: {total - mask_count} ({(total-mask_count)/total*100:.1f}%)")
    print(f"  耗时: {elapsed:.2f}s (平均 {elapsed/total:.3f}s/张)")
    print(f"  结果已保存: {args.output}")


if __name__ == '__main__':
    main()
