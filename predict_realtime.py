"""
实时口罩检测 — 人脸检测 + CNN二分类 + Grad-CAM可视化

用法:
    python predict_realtime.py                    # 摄像头实时检测 (SSD)
    python predict_realtime.py --image test.jpg   # 单张图片检测
    python predict_realtime.py --image test.jpg --save result.jpg  # 保存结果
    python predict_realtime.py --detector haar    # 回退到 Haar 级联
    python predict_realtime.py --detector ssd-e2e # SSD 端到端（跳过 CNN）
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL = os.path.join(BASE_DIR, 'models', 'mask_classifier_binary.h5')
IMG_SIZE = (224, 224)
CONF_THRESH = 0.5

ID2CLASS = {0: ('MASK', (0, 255, 0)), 1: ('NO MASK', (0, 0, 255))}


# ═══════════════ 人脸检测器 ═══════════════

class HaarFaceDetector:
    """Haar 级联人脸检测（多模型 + 非极大抑制）"""

    def __init__(self):
        cascades = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml',
        ]
        self.cascades = []
        for c in cascades:
            path = cv2.data.haarcascades + c
            if os.path.exists(path):
                self.cascades.append(cv2.CascadeClassifier(path))

    def detect(self, image):
        """返回 [(x1, y1, x2, y2, conf), ...]"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_faces = []
        for cascade in self.cascades:
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4,
                minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
            )
            all_faces.extend([(x, y, w, h, 0.9) for (x, y, w, h) in faces])

        if not all_faces:
            return []

        all_faces.sort(key=lambda r: r[2] * r[3], reverse=True)
        merged = []
        for x1, y1, w1, h1, c1 in all_faces:
            overlap = False
            for x2, y2, w2, h2, c2 in merged:
                ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                iou = ix * iy / (w1 * h1 + w2 * h2 - ix * iy + 1e-6)
                if iou > 0.3:
                    overlap = True
                    break
            if not overlap:
                merged.append((x1, y1, x1 + w1, y1 + h1, c1))
        return merged


class SsdFaceDetector:
    """SSD 人脸检测器 — 只取人脸框，分类交给 CNN"""

    def __init__(self, conf_thresh=0.5):
        from ssd_detector import SSDDetector
        self.ssd = SSDDetector(conf_thresh=conf_thresh, iou_thresh=0.4)

    def detect(self, image):
        """返回 [(x1, y1, x2, y2, conf), ...]"""
        return self.ssd.detect_faces_only(image)


class SsdE2EDetector:
    """SSD 端到端检测器 — 人脸框 + 口罩分类一次完成"""

    def __init__(self, conf_thresh=0.5):
        from ssd_detector import SSDDetector
        self.ssd = SSDDetector(conf_thresh=conf_thresh, iou_thresh=0.4)

    def detect(self, image):
        """返回 [(x1, y1, x2, y2, conf, class_id), ...]"""
        dets = self.ssd.detect(image)
        return [(d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3],
                 d['confidence'], d['class']) for d in dets]


# ═══════════════ Grad-CAM ═══════════════

def make_gradcam_heatmap(model, img_array, class_idx):
    last_conv = None
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


# ═══════════════ 主推理类 ═══════════════

class MaskDetector:
    def __init__(self, detector_type='ssd'):
        self.detector_type = detector_type
        self.e2e_mode = (detector_type == 'ssd-e2e')

        print(f'[加载] 人脸检测器 ({detector_type})...')
        if detector_type == 'haar':
            self.face_detector = HaarFaceDetector()
        elif detector_type == 'ssd-e2e':
            self.face_detector = SsdE2EDetector(conf_thresh=CONF_THRESH)
        else:
            self.face_detector = SsdFaceDetector(conf_thresh=CONF_THRESH)

        if self.e2e_mode:
            self.cnn = None
            print('[跳过] SSD 端到端模式，不加载 CNN')
        else:
            print('[加载] CNN 二分类模型...')
            self.cnn = tf.keras.models.load_model(CNN_MODEL, compile=False)
            print('[预热] 首次推理编译...')
            dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
            self.cnn.predict(dummy, verbose=0)

        print('[就绪] 图片: --image  |  摄像头: 直接运行  |  退出: Q')

    def _cnn_classify(self, face_roi):
        """CNN 对裁剪的人脸区域进行分类"""
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb).resize(IMG_SIZE)
        face_arr = np.array(face_pil, dtype=np.float32) / 255.0
        face_arr_exp = np.expand_dims(face_arr, axis=0)
        pred = self.cnn.predict(face_arr_exp, verbose=0)[0]
        cls_idx = int(np.argmax(pred))
        conf = float(pred[cls_idx])
        return cls_idx, conf, face_arr_exp

    def predict(self, image, with_gradcam=True):
        result = image.copy()

        try:
            faces = self.face_detector.detect(image)
        except Exception as e:
            print(f'[SSD 检测异常] {e}')
            faces = []

        detections = []

        for face_data in faces:
            if self.e2e_mode:
                x1, y1, x2, y2, fd_conf, cls_idx = face_data
                conf = fd_conf
            else:
                x1, y1, x2, y2, fd_conf = face_data
                face = image[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                cls_idx, conf, _ = self._cnn_classify(face)

            label, color = ID2CLASS[cls_idx]
            text = f'{label} {conf:.0%}'

            # 绘制边框 + 标签
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(result, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Grad-CAM（仅 CNN 混合模式）
            if with_gradcam and not self.e2e_mode:
                face = image[y1:y2, x1:x2]
                if face.shape[0] > 30 and face.shape[1] > 30:
                    try:
                        _, _, face_arr_exp = self._cnn_classify(face)
                        heatmap = make_gradcam_heatmap(self.cnn, face_arr_exp, cls_idx)
                        if heatmap is not None:
                            heatmap = cv2.resize(heatmap, (x2 - x1, y2 - y1))
                            roi = result[y1:y2, x1:x2]
                            result[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.55, heatmap, 0.45, 0)
                    except Exception:
                        pass

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': cls_idx,
                'label': label,
                'confidence': conf,
            })

        return result, detections


# ═══════════════ 入口 ═══════════════

def main():
    parser = argparse.ArgumentParser(description='实时口罩检测 — CNN二分类 + Grad-CAM')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--save', type=str, help='保存结果路径')
    parser.add_argument('--no-cam', action='store_true', help='禁用 Grad-CAM')
    parser.add_argument('--headless', action='store_true', help='无GUI模式（不弹窗口）')
    parser.add_argument('--detector', type=str, default='ssd',
                        choices=['ssd', 'ssd-e2e', 'haar'],
                        help='人脸检测器: ssd (SSD+CNN), ssd-e2e (SSD端到端), haar (Haar+CNN)')
    args = parser.parse_args()

    detector = MaskDetector(detector_type=args.detector)
    use_cam = not args.no_cam

    if args.image:
        frame = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            frame = cv2.imread(args.image)
        if frame is None:
            print(f'无法读取图片: {args.image}')
            sys.exit(1)

        t0 = time.time()
        result, dets = detector.predict(frame, with_gradcam=use_cam)
        elapsed = time.time() - t0
        print(f'检测到 {len(dets)} 张人脸 (耗时 {elapsed:.2f}s)')
        for d in dets:
            print(f'  {d["label"]} conf={d["confidence"]:.2%} bbox={d["bbox"]}')

        if args.save:
            cv2.imwrite(args.save, result)
            print(f'结果已保存: {args.save}')

        if not args.headless:
            cv2.imshow('Mask Detection', result)
            print('按任意键退出...')
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        # 依次尝试不同后端，有些摄像头在 DirectShow/MSMF 会返回纯黑帧
        backends = [
            (cv2.CAP_ANY, 'Default'),
            (cv2.CAP_DSHOW, 'DirectShow'),
        ]
        cap = None
        for bid, bname in backends:
            c = cv2.VideoCapture(0, bid)
            if not c.isOpened():
                c.release()
                continue
            # 读取一帧验证是否为纯黑帧
            for _ in range(10):
                ret, test = c.read()
                if ret and test.mean() > 10:
                    cap = c
                    print(f'[摄像头] 后端: {bname} | 首帧 OK (mean={test.mean():.0f})')
                    break
            if cap is not None:
                break
            c.release()
            print(f'[摄像头] {bname} 返回纯黑帧，尝试下一个...')

        if cap is None:
            print('无法打开可用摄像头 (索引 0)')
            sys.exit(1)

        fps_hist = []
        frame_idx = 0
        print('[摄像头] 按 Q 退出')
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f'[摄像头] 读取失败 (第{frame_idx}帧)')
                break

            if frame_idx == 0:
                print(f'[摄像头] 首帧: shape={frame.shape} mean={frame.mean():.0f} '
                      f'min={frame.min()} max={frame.max()}')
            frame_idx += 1
            t0 = time.time()
            result, dets = detector.predict(frame, with_gradcam=use_cam)
            fps = 1.0 / (time.time() - t0 + 1e-6)
            fps_hist.append(fps)
            avg_fps = np.mean(fps_hist[-30:])

            cv2.putText(result, f'FPS: {avg_fps:.1f} | Faces: {len(dets)} | {args.detector}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('Mask Detection - Press Q to quit', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
