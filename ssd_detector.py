"""
SSD 人脸+口罩检测器 — 基于 Caffe SSD 模型，OpenCV DNN 推理

一次推理同时输出人脸边界框和口罩分类（Mask / NoMask）。
模型来源: FaceMaskDetection-master/models/face_mask_detection.{prototxt,caffemodel}
"""

import os
import numpy as np
import cv2

from config import BASE_DIR

PROTOTXT = os.path.join(BASE_DIR, 'models', 'face_mask_detection.prototxt')
CAFFEMODEL = os.path.join(BASE_DIR, 'models', 'face_mask_detection.caffemodel')

INPUT_SIZE = (260, 260)
ID2CLASS = {0: 'Mask', 1: 'NoMask'}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}

# ── Anchor 配置（与 prototxt 一致） ──
FEATURE_MAP_SIZES = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
ANCHOR_SIZES = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
ANCHOR_RATIOS = [[1, 0.62, 0.42]] * 5


def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios):
    """生成 SSD 先验框，与参考项目 FaceMaskDetection-master 一致"""
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        center = np.stack((np.expand_dims(cx_grid, -1),
                            np.expand_dims(cy_grid, -1)), axis=-1)

        num_anchors = len(anchor_sizes[idx]) + len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
        anchor_width_heights = []

        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0]
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0,
                                          width / 2.0, height / 2.0])

        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0]
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0,
                                          width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        anchor_bboxes.append(bbox_coords.reshape((-1, 4)))
    return np.concatenate(anchor_bboxes, axis=0)


def decode_bbox(anchors, raw_outputs, variances=(0.1, 0.1, 0.2, 0.2)):
    """将 SSD 原始输出解码为实际边界框坐标 [xmin, ymin, xmax, ymax]"""
    anchor_centers_x = (anchors[:, 0:1] + anchors[:, 2:3]) / 2
    anchor_centers_y = (anchors[:, 1:2] + anchors[:, 3:4]) / 2
    anchors_w = anchors[:, 2:3] - anchors[:, 0:1]
    anchors_h = anchors[:, 3:4] - anchors[:, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, 3:4]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    return np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)


def nms_boxes(bboxes, confidences, conf_thresh=0.5, iou_thresh=0.4):
    """OpenCV DNN NMS 包装 — 比纯 Python 实现快一个数量级"""
    if len(bboxes) == 0:
        return []
    # 转换 [xmin,ymin,xmax,ymax] → [x,y,w,h]
    boxes = []
    scores = []
    for i, b in enumerate(bboxes):
        if confidences[i] > conf_thresh:
            boxes.append([float(b[0]), float(b[1]),
                          float(b[2] - b[0]), float(b[3] - b[1])])
            scores.append(float(confidences[i]))
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()


class SSDDetector:
    """SSD 人脸+口罩检测器，一次推理同时输出边界框和分类"""

    def __init__(self, conf_thresh=0.5, iou_thresh=0.4):
        if not os.path.exists(PROTOTXT):
            raise FileNotFoundError(f"prototxt 不存在: {PROTOTXT}")
        if not os.path.exists(CAFFEMODEL):
            raise FileNotFoundError(f"caffemodel 不存在: {CAFFEMODEL}")

        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
        # 强制 CPU 后端，避免 Windows 上 OpenCL/GPU 与摄像头资源竞争
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # 预生成 anchor
        self.anchors = generate_anchors(FEATURE_MAP_SIZES, ANCHOR_SIZES, ANCHOR_RATIOS)

    def detect(self, image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, INPUT_SIZE,
                                      (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        y_bboxes_raw, y_cls_raw = self.net.forward(
            ['loc_branch_concat', 'cls_branch_concat'])

        y_bboxes = decode_bbox(self.anchors, y_bboxes_raw[0])
        y_cls = y_cls_raw[0]

        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_classes = np.argmax(y_cls, axis=1)

        keep_idxs = nms_boxes(y_bboxes, bbox_max_scores,
                              conf_thresh=self.conf_thresh,
                              iou_thresh=self.iou_thresh)

        detections = []
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = int(bbox_max_classes[idx])
            bbox = y_bboxes[idx]
            x1 = max(0, int(bbox[0] * w))
            y1 = max(0, int(bbox[1] * h))
            x2 = min(int(bbox[2] * w), w)
            y2 = min(int(bbox[3] * h), h)

            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': class_id,
                'label': ID2CLASS[class_id],
                'confidence': conf,
            })

        return detections

    def detect_faces_only(self, image):
        """仅返回人脸边界框列表 [(x1, y1, x2, y2, conf), ...]"""
        dets = self.detect(image)
        return [(d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3],
                 d['confidence']) for d in dets]


# 单例缓存
_detector_instance = None


def get_ssd_detector(conf_thresh=0.5, iou_thresh=0.4):
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SSDDetector(conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    return _detector_instance


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法: python ssd_detector.py <图片路径>")
        sys.exit(1)

    detector = SSDDetector()
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"无法读取图片: {sys.argv[1]}")
        sys.exit(1)

    detections = detector.detect(img)
    print(f"检测到 {len(detections)} 张人脸:")
    for d in detections:
        print(f"  {d['label']} conf={d['confidence']:.2%} bbox={d['bbox']}")

    # 绘制结果
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        color = CLASS_COLORS[d['class']]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{d['label']} {d['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('SSD Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
