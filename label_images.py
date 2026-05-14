"""
交互式图片标注工具 — 二分类口罩检测
逐张显示图片，按键分类到 images_2class/

用法：
    python label_images.py         # 从源目录标注

类别：
    1 - mask      戴口罩（所有类型）
    2 - nomask    未佩戴口罩

控制：
    1/2  分类
    s    跳过
    d    无效（非人脸/非口罩图，永久排除）
    q    保存进度并退出
    b    回退上一张
"""

import os
import sys
import shutil
import json
import cv2
import numpy as np

# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIRS = [
    os.path.join(BASE_DIR, 'dataset', 'train', 'data', 'labeled'),
]
OUT_DIR = os.path.join(BASE_DIR, 'dataset', 'train', 'data', 'images_2class')
PROGRESS_FILE = os.path.join(BASE_DIR, 'label_progress.json')

CATEGORIES = {
    ord('1'): 'mask',
    ord('2'): 'nomask',
}

CATEGORY_LABELS = {
    '1': '戴口罩',
    '2': '未戴口罩',
}

# 创建目标目录
for cat in CATEGORIES.values():
    os.makedirs(os.path.join(OUT_DIR, cat), exist_ok=True)


def collect_all_images():
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    for src_dir in SRC_DIRS:
        if not os.path.isdir(src_dir):
            continue
        for fname in sorted(os.listdir(src_dir)):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                images.append(os.path.join(src_dir, fname))
    return images


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'labeled': {}, 'current_index': 0}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def print_stats(progress):
    counts = {cat: 0 for cat in CATEGORIES.values()}
    skipped = 0
    invalid = 0
    for path, label in progress['labeled'].items():
        if label in counts:
            counts[label] += 1
        elif label == 'skip':
            skipped += 1
        elif label == 'invalid':
            invalid += 1
    total_labeled = sum(counts.values())
    print("\n" + "=" * 50)
    print(f"  已标注: {total_labeled} | 跳过: {skipped} | 无效: {invalid}")
    for key_str, cat_label in CATEGORY_LABELS.items():
        cat = CATEGORIES[ord(key_str)]
        print(f"  [{key_str}] {cat_label}: {counts[cat]}")
    print("=" * 50)


def label_images():
    images = collect_all_images()
    progress = load_progress()
    idx = progress['current_index']

    print(f"\n共 {len(images)} 张图片待标注，从第 {idx + 1} 张开始\n")
    print("按键说明：")
    for key, label in CATEGORY_LABELS.items():
        print(f"  [{key}] - {label}")
    print("  [s] - 跳过  [b] - 回退  [q] - 保存并退出\n")

    WIN_NAME = 'Label Tool'
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 700, 700)

    while idx < len(images):
        img_path = images[idx]

        # 跳过已标注的
        if img_path in progress['labeled']:
            idx += 1
            continue

        # 读取图片（用 numpy 绕过 OpenCV 中文路径问题）
        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            print(f"  [错误] 无法读取: {os.path.basename(img_path)}")
            idx += 1
            continue

        # 缩放到显示窗口
        h, w = img.shape[:2]
        scale = min(700 / w, 700 / h, 1.0)
        display = cv2.resize(img, (int(w * scale), int(h * scale)))

        # 叠加信息
        fname = os.path.basename(img_path)
        src_folder = os.path.basename(os.path.dirname(img_path))
        info = f"[{idx + 1}/{len(images)}] {src_folder}/{fname}"
        cv2.putText(display, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        # 叠加按键提示（英文/数字，渲染可靠）
        hints = [
            "[1]Mask [2]NoMask",
            "[S]Skip [D]Invalid [B]Back [Q]Quit",
        ]
        for i, hint in enumerate(hints):
            y = display.shape[0] - 25 + i * 20
            cv2.putText(display, hint, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)

        cv2.imshow(WIN_NAME, display)

        # 等待按键
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            progress['current_index'] = idx
            save_progress(progress)
            print(f"\n进度已保存。下次从第 {idx + 1} 张继续。")
            break

        elif key in CATEGORIES:
            label = CATEGORIES[key]
            dest = os.path.join(OUT_DIR, label, fname)
            shutil.copy2(img_path, dest)
            progress['labeled'][img_path] = label
            save_progress(progress)
            print(f"  [{idx + 1}/{len(images)}] {fname} -> {CATEGORY_LABELS[chr(key)]}")
            idx += 1

        elif key == ord('s'):
            progress['labeled'][img_path] = 'skip'
            save_progress(progress)
            print(f"  [{idx + 1}/{len(images)}] {fname} -> 跳过")
            idx += 1

        elif key == ord('d'):
            progress['labeled'][img_path] = 'invalid'
            save_progress(progress)
            print(f"  [{idx + 1}/{len(images)}] {fname} -> 无效（排除）")
            idx += 1

        elif key == ord('b'):
            # 回退
            undone = False
            for prev_idx in range(idx - 1, -1, -1):
                prev_path = images[prev_idx]
                if prev_path in progress['labeled']:
                    old_label = progress['labeled'].pop(prev_path)
                    if old_label in CATEGORIES.values():
                        old_dest = os.path.join(OUT_DIR, old_label,
                                                os.path.basename(prev_path))
                        if os.path.exists(old_dest):
                            os.remove(old_dest)
                    print(f"  回退: {os.path.basename(prev_path)} 撤销'{old_label}'")
                    idx = prev_idx
                    progress['current_index'] = idx
                    undone = True
                    break
            if not undone:
                print("  没有可回退的标注")
            save_progress(progress)

    cv2.destroyAllWindows()
    print_stats(progress)

    remaining = len(images) - len(progress['labeled'])
    if remaining == 0:
        print("\n全部标注完成！")
        print(f"标注结果: {OUT_DIR}")
    else:
        print(f"\n还有 {remaining} 张未标注。再跑 python label_images.py 继续。")


if __name__ == '__main__':
    label_images()
