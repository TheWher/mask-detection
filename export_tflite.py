"""
模型导出 — H5 → TFLite (FP16 + INT8 量化)

用法：
    python export_tflite.py                    # 导出 FP16 + INT8 + 动态量化
    python export_tflite.py --no-int8          # 跳过 INT8 (无训练样本时)

输出：
    models/mask_classifier_fp16.tflite         # ~11MB
    models/mask_classifier_int8.tflite         # ~5MB
    models/mask_classifier_dynamic.tflite      # ~6MB (动态量化，无需校准数据)
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from config import MODEL_PATH, SAVE_DIR, TRAIN_DIR, IMG_SIZE


def representative_dataset_gen():
    """INT8 量化校准数据集 — 从训练数据中随机采样 100 张"""
    valid_exts = {'.jpg', '.jpeg', '.png'}
    image_paths = []
    for cls in ['mask', 'nomask']:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.isdir(cls_dir):
            for f in os.listdir(cls_dir):
                if os.path.splitext(f)[1].lower() in valid_exts:
                    image_paths.append(os.path.join(cls_dir, f))
    np.random.seed(42)
    np.random.shuffle(image_paths)
    sample_paths = image_paths[:100]

    for path in sample_paths:
        try:
            from PIL import Image
            img = Image.open(path).convert('RGB').resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            yield [np.expand_dims(arr, axis=0)]
        except Exception:
            continue


def export_tflite(model, path, converter_fn, description):
    """通用导出流程"""
    try:
        converter = converter_fn()
        tflite_model = converter.convert()
        with open(path, 'wb') as f:
            f.write(tflite_model)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f'  {description}: {path} ({size_mb:.1f}MB)')
    except Exception as e:
        print(f'  [跳过] {description}: {e}')


def main():
    parser = argparse.ArgumentParser(description='导出 TFLite 模型')
    parser.add_argument('--no-int8', action='store_true', help='跳过 INT8 (无校准数据时)')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f'[错误] 模型文件不存在: {MODEL_PATH}')
        print('[提示] 请先运行: python train.py')
        return

    print(f'[加载] {MODEL_PATH}')
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    h5_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f'[原始] H5 模型: {h5_size:.1f}MB | 参数量: {model.count_params():,}\n')

    # FP16 量化
    export_tflite(model,
                  os.path.join(SAVE_DIR, 'mask_classifier_fp16.tflite'),
                  lambda: _fp16_converter(model),
                  'FP16')

    # 动态范围量化（无需校准数据）
    export_tflite(model,
                  os.path.join(SAVE_DIR, 'mask_classifier_dynamic.tflite'),
                  lambda: _dynamic_converter(model),
                  '动态量化')

    # INT8 量化（需校准数据）
    if not args.no_int8:
        export_tflite(model,
                      os.path.join(SAVE_DIR, 'mask_classifier_int8.tflite'),
                      lambda: _int8_converter(model),
                      'INT8')

    print('\n[完成] 推荐移动端/嵌入式使用 INT8 (最小)，通用场景用 动态量化')


def _fp16_converter(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter


def _dynamic_converter(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter


def _int8_converter(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter


if __name__ == '__main__':
    main()
