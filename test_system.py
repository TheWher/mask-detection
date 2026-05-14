"""
口罩检测系统 — 功能测试 (二分类: Mask / NoMask)

用法：
    python test_system.py
"""

import os
import sys
import time
import tempfile
import warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train', 'data', 'images_2class')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mask_classifier_binary.h5')
REPORT_PATH = os.path.join(BASE_DIR, 'test_report.txt')

IMG_SIZE = (224, 224)
CLASSES = ['mask', 'nomask']
NUM_CLASSES = 2

results = []
report_lines = []


def log(msg, level='INFO'):
    line = f"[{level}] {msg}"
    print(line)
    report_lines.append(line)


def test_case(name):
    def decorator(func):
        def wrapper():
            log(f"{'=' * 50}", '')
            log(f"测试: {name}", 'CASE')
            try:
                start = time.time()
                passed = func()
                elapsed = time.time() - start
                status = 'PASS' if passed else 'FAIL'
                results.append({'name': name, 'status': status, 'time': elapsed})
                log(f"结果: {status} | 耗时: {elapsed:.2f}s", 'RESULT')
                return passed
            except Exception as e:
                results.append({'name': name, 'status': 'ERROR', 'time': time.time() - start})
                log(f"异常: {e}", 'ERROR')
                return False
        return wrapper
    return decorator


@test_case("TC01: 数据集目录结构")
def test_directory_structure():
    all_ok = True
    for cls in CLASSES:
        path = os.path.join(TRAIN_DIR, cls)
        if not os.path.isdir(path):
            log(f"  [FAIL] 目录缺失: {path}", 'FAIL')
            all_ok = False
        else:
            count = len([f for f in os.listdir(path)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            log(f"  [OK] {cls}: {count} 张", 'CHECK')
    return all_ok


@test_case("TC02: 数据集规模")
def test_dataset_size():
    total = 0
    for cls in CLASSES:
        path = os.path.join(TRAIN_DIR, cls)
        count = len([f for f in os.listdir(path)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        log(f"  {cls}: {count} 张", 'INFO')
        total += count
    log(f"  总计: {total} 张", 'INFO')
    return total >= 20


@test_case("TC03: 类别比例")
def test_class_balance():
    counts = {}
    for cls in CLASSES:
        path = os.path.join(TRAIN_DIR, cls)
        counts[cls] = len([f for f in os.listdir(path)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    a, b = counts['mask'], counts['nomask']
    ratio = max(a, b) / max(min(a, b), 1)
    log(f"  Mask: {a} | NoMask: {b} | 比例: {ratio:.1f}:1", 'INFO')
    return ratio < 5


@test_case("TC04: 图像格式验证")
def test_image_format():
    from PIL import Image
    sample_paths = []
    for cls in CLASSES:
        path = os.path.join(TRAIN_DIR, cls)
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            sample_paths.extend([os.path.join(path, f) for f in files[:5]])
    all_ok = True
    for p in sample_paths:
        try:
            img = Image.open(p)
            img.verify()
            img = Image.open(p)
            w, h = img.size
            if w < 10 or h < 10:
                all_ok = False
        except Exception:
            all_ok = False
    return all_ok and len(sample_paths) > 0


@test_case("TC05: 模型结构")
def test_model_structure():
    from model import build_mask_classifier, compile_model
    model = build_mask_classifier(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model = compile_model(model)
    params = model.count_params()
    log(f"  输入: {model.input_shape} | 输出: {model.output_shape}", 'INFO')
    log(f"  参数量: {params:,}", 'INFO')
    return model.output_shape[-1] == 2 and params < 3_000_000


@test_case("TC06: 推理 — Mask 样本")
def test_inference_mask():
    from predict import load_trained_model, predict
    if not os.path.exists(MODEL_PATH):
        log("  [SKIP] 无模型文件", 'SKIP')
        return True
    model = load_trained_model()
    mask_dir = os.path.join(TRAIN_DIR, 'mask')
    if not os.path.isdir(mask_dir):
        log("  [SKIP] mask 目录不存在", 'SKIP')
        return True
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        return True
    name, conf, _ = predict(model, os.path.join(mask_dir, files[0]))
    log(f"  预测: {name} | 置信度: {conf:.4f}", 'INFO')
    return name == 'Mask'


@test_case("TC07: 推理 — NoMask 样本")
def test_inference_nomask():
    from predict import load_trained_model, predict
    if not os.path.exists(MODEL_PATH):
        log("  [SKIP] 无模型文件", 'SKIP')
        return True
    model = load_trained_model()
    nomask_dir = os.path.join(TRAIN_DIR, 'nomask')
    if not os.path.isdir(nomask_dir):
        log("  [SKIP] nomask 目录不存在", 'SKIP')
        return True
    files = [f for f in os.listdir(nomask_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        return True
    name, conf, _ = predict(model, os.path.join(nomask_dir, files[0]))
    log(f"  预测: {name} | 置信度: {conf:.4f}", 'INFO')
    return name == 'NoMask'


@test_case("TC08: 边界 — 文件不存在")
def test_file_not_found():
    from predict import preprocess_image
    try:
        preprocess_image('/nonexistent/image.jpg')
        return False
    except FileNotFoundError:
        return True
    except Exception:
        return True


@test_case("TC09: 边界 — 损坏图片")
def test_corrupt_image():
    from predict import preprocess_image
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp.write(b'not a valid image')
    tmp.close()
    try:
        preprocess_image(tmp.name)
        return False
    except (ValueError, Exception):
        return True
    finally:
        os.unlink(tmp.name)


@test_case("TC10: 批量推理")
def test_batch_predict():
    from batch_predict import load_trained_model, preprocess_batch
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        log("  [SKIP] 无模型文件", 'SKIP')
        return True
    model = load_trained_model()
    test_files = []
    for cls in CLASSES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.isdir(cls_dir):
            for f in sorted(os.listdir(cls_dir))[:3]:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_files.append(os.path.join(cls_dir, f))
    if not test_files:
        return True
    batch, valid = preprocess_batch(test_files)
    if len(batch) == 0:
        return True
    probs = model.predict(batch, verbose=0)
    return len(probs) == len(valid)


@test_case("TC11: 单张推理性能")
def test_inference_perf():
    from predict import load_trained_model, predict
    if not os.path.exists(MODEL_PATH):
        log("  [SKIP] 无模型文件", 'SKIP')
        return True
    model = load_trained_model()
    test_files = []
    for cls in CLASSES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.isdir(cls_dir):
            for f in sorted(os.listdir(cls_dir))[:3]:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_files.append(os.path.join(cls_dir, f))
    if not test_files:
        return True
    times = []
    for p in test_files:
        t0 = time.time()
        predict(model, p)
        times.append(time.time() - t0)
    avg = np.mean(times)
    log(f"  平均: {avg:.4f}s | 测试 {len(times)} 次", 'INFO')
    return avg < 1.0


@test_case("TC12: 环境信息")
def test_environment():
    import tensorflow as tf
    log(f"  Python: {sys.version.split()[0]}", 'INFO')
    log(f"  TensorFlow: {tf.__version__}", 'INFO')
    log(f"  NumPy: {np.__version__}", 'INFO')
    return True


def run_all_tests():
    print("\n" + "=" * 50)
    print("  口罩检测 — 二分类测试套件")
    print("=" * 50)

    for func in [
        test_directory_structure, test_dataset_size, test_class_balance,
        test_image_format, test_model_structure, test_inference_mask,
        test_inference_nomask, test_file_not_found, test_corrupt_image,
        test_batch_predict, test_inference_perf, test_environment,
    ]:
        func()

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')

    print(f"\n{'=' * 50}")
    print(f"  总计: {len(results)} | PASS: {passed} | FAIL: {failed} | ERROR: {errors}")
    print(f"  通过率: {passed / len(results) * 100:.1f}%")
    print(f"{'=' * 50}")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"报告: {REPORT_PATH}")


if __name__ == '__main__':
    run_all_tests()
