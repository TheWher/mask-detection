"""
共享配置和工具函数
"""
import os
import sys
import numpy as np
from PIL import Image


# ═══════════════ 路径配置 ═══════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mask_classifier_binary.h5')
SAVE_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train', 'data', 'images_2class')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# ═══════════════ 模型参数 ═══════════════
IMG_SIZE = (224, 224)
NUM_CLASSES = 2
CLASS_NAMES = {0: 'Mask', 1: 'NoMask'}
CLASSES_LIST = ['mask', 'nomask']

# ═══════════════ 检测参数 ═══════════════
CONF_THRESH = 0.5
IOU_THRESH = 0.4
SSD_INPUT_SIZE = (260, 260)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def load_trained_model(path=None):
    """加载训练好的 CNN 模型（compile=False 跳过重编译，仅推理）"""
    from tensorflow.keras.models import load_model
    model_path = path or MODEL_PATH
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("[提示] 请先运行: python train.py")
        sys.exit(1)
    return load_model(model_path, compile=False)


def preprocess_image(image_path):
    """预处理单张图片 → (1, 224, 224, 3) float32 [0,1]"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception:
        raise ValueError(f"无法识别该图片文件: {image_path}")
    if img.width < 10 or img.height < 10:
        raise ValueError(f"图片尺寸异常: {img.size}")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)
