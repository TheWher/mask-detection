"""
数据集准备脚本
功能：生成 800 张合成人脸图像（戴口罩 / 不戴口罩 各 400 张）
     自动划分训练集(80%) 和测试集(20%)

说明：
  - 本项目使用合成数据演示完整流程
  - 如使用真实数据，请将图片按分类放入对应文件夹即可
  - 合成图像为简单的几何人脸模拟，确保模型可学习区分性特征

标注规范：
  with_mask/     -> 标签 0（戴口罩）
  without_mask/  -> 标签 1（不戴口罩）
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

# ==================== 配置 ====================
IMAGE_SIZE = (224, 224)          # 输出图像尺寸（统一 224x224）
NUM_SAMPLES = 800                # 总样本数
SEED = 42                        # 随机种子，保证可复现
TRAIN_RATIO = 0.8                # 训练集比例

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

os.makedirs(os.path.join(TRAIN_DIR, 'with_mask'), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'without_mask'), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, 'with_mask'), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, 'without_mask'), exist_ok=True)


def draw_face_with_mask(draw, cx, cy, r):
    """在画布上绘制戴口罩的人脸（椭圆脸 + 蓝色口罩）"""
    # 肤色椭圆脸
    draw.ellipse([cx - r, cy - r - 5, cx + r, cy + r + 5],
                 fill=(255, 200, 150), outline=(200, 150, 100))
    # 眼睛（两个黑点）
    eye_r = r // 6
    draw.ellipse([cx - r // 3 - eye_r, cy - r // 3 - eye_r,
                  cx - r // 3 + eye_r, cy - r // 3 + eye_r], fill=(50, 50, 50))
    draw.ellipse([cx + r // 3 - eye_r, cy - r // 3 - eye_r,
                  cx + r // 3 + eye_r, cy - r // 3 + eye_r], fill=(50, 50, 50))
    # 蓝色口罩（覆盖嘴部区域）
    draw.rectangle([cx - r + 5, cy + r // 4, cx + r - 5, cy + r // 2 - 5],
                   fill=(100, 150, 255), outline=(70, 120, 200))
    # 口罩挂耳线
    draw.line([cx - r + 5, cy + r // 4, cx - r - 2, cy - r // 4],
              fill=(150, 150, 150), width=1)
    draw.line([cx + r - 5, cy + r // 4, cx + r + 2, cy - r // 4],
              fill=(150, 150, 150), width=1)


def draw_face_without_mask(draw, cx, cy, r):
    """在画布上绘制不戴口罩的人脸（椭圆脸 + 完整五官）"""
    # 肤色椭圆脸
    draw.ellipse([cx - r, cy - r - 5, cx + r, cy + r + 5],
                 fill=(255, 200, 150), outline=(200, 150, 100))
    # 眼睛
    eye_r = r // 6
    draw.ellipse([cx - r // 3 - eye_r, cy - r // 3 - eye_r,
                  cx - r // 3 + eye_r, cy - r // 3 + eye_r], fill=(50, 50, 50))
    draw.ellipse([cx + r // 3 - eye_r, cy - r // 3 - eye_r,
                  cx + r // 3 + eye_r, cy - r // 3 + eye_r], fill=(50, 50, 50))
    # 鼻子
    draw.ellipse([cx - 3, cy - 1, cx + 3, cy + 5], fill=(220, 160, 110))
    # 嘴巴
    draw.arc([cx - r // 3, cy + r // 4, cx + r // 3, cy + r // 2],
             start=0, end=180, fill=(180, 80, 80), width=2)


def generate_single_image(size, with_mask):
    """生成一张随机构造的人脸图像"""
    img = Image.new('RGB', size, (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # 随机人脸位置与大小（适配 224x224 画布）
    w, h = size
    r_min, r_max = w // 5, w // 3
    r = np.random.randint(r_min, r_max)
    cx = np.random.randint(r + 5, w - r - 5)
    cy = np.random.randint(r + 5, h - r - 5)

    if with_mask:
        draw_face_with_mask(draw, cx, cy, r)
    else:
        draw_face_without_mask(draw, cx, cy, r)

    # 添加少量随机噪点（模拟真实图像噪声）
    pixels = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.float32)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)

    return img


def generate_dataset():
    """生成全部数据集并划分训练/测试集"""
    np.random.seed(SEED)

    print(f"开始生成 {NUM_SAMPLES} 张合成人脸图像 (尺寸: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]})...")

    # 生成两类图像的文件名列表
    with_mask_files = []
    without_mask_files = []

    for i in range(NUM_SAMPLES // 2):
        # 戴口罩
        img = generate_single_image(IMAGE_SIZE, with_mask=True)
        fname = f"with_mask_{i:04d}.jpg"
        with_mask_files.append((fname, img))

        # 不戴口罩
        img = generate_single_image(IMAGE_SIZE, with_mask=False)
        fname = f"without_mask_{i:04d}.jpg"
        without_mask_files.append((fname, img))

        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1} 对...")

    # 划分训练集和测试集（80/20）
    wm_train, wm_test = train_test_split(
        with_mask_files, test_size=1 - TRAIN_RATIO, random_state=SEED
    )
    wom_train, wom_test = train_test_split(
        without_mask_files, test_size=1 - TRAIN_RATIO, random_state=SEED
    )

    # 写入训练集
    print("\n写入训练集...")
    for fname, img in wm_train:
        img.save(os.path.join(TRAIN_DIR, 'with_mask', fname))
    for fname, img in wom_train:
        img.save(os.path.join(TRAIN_DIR, 'without_mask', fname))
    print(f"  训练集: with_mask={len(wm_train)}张, without_mask={len(wom_train)}张")

    # 写入测试集
    print("写入测试集...")
    for fname, img in wm_test:
        img.save(os.path.join(TEST_DIR, 'with_mask', fname))
    for fname, img in wom_test:
        img.save(os.path.join(TEST_DIR, 'without_mask', fname))
    print(f"  测试集: with_mask={len(wm_test)}张, without_mask={len(wom_test)}张")

    print(f"\n数据集生成完毕！")
    print(f"  训练集总计: {len(wm_train) + len(wom_train)} 张")
    print(f"  测试集总计: {len(wm_test) + len(wom_test)} 张")
    print(f"  图像尺寸: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")


if __name__ == '__main__':
    generate_dataset()
