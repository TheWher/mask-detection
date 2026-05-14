# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 标注图片（1=mask 2=nomask, s跳过 d无效 b回退 q保存）
python label_images.py

# 训练二分类模型 → models/mask_classifier_binary.h5
python train.py

# 单张预测
python predict.py <图片路径>

# 批量预测 + CSV
python batch_predict.py <图片目录> --output result.csv

# 实时检测（人脸检测 + CNN + Grad-CAM）
python predict_realtime.py                                    # 摄像头 (SSD+CNN)
python predict_realtime.py --image test.jpg                   # 图片
python predict_realtime.py --image test.jpg --save result.jpg --no-cam --headless
python predict_realtime.py --detector ssd-e2e                 # SSD端到端（跳过CNN，更快）
python predict_realtime.py --detector haar                    # Haar回退模式

# SSD 检测器（独立使用）
python ssd_detector.py <图片路径>                             # 单张测试

# 测试套件（12项）
python test_system.py

# 合成数据集（仅演示）
python dataset/prepare_dataset.py
```

## 架构概览

**任务**：图像二分类 — 判断人脸是否佩戴口罩（Mask / NoMask）。

**人脸检测**：默认使用 SSD Caffe 模型（OpenCV DNN 推理），可选 Haar 级联回退。SSD 支持端到端模式（人脸+口罩一次完成）和混合模式（SSD 取人脸框 + CNN 分类）。

**二类标签**（flow_from_directory 按字母序分配）：

| 目录 | 标签 | 含义 |
|------|------|------|
| `mask/` | 0 | 佩戴口罩 |
| `nomask/` | 1 | 未佩戴口罩 |

**模型**：4 层 CNN + Flatten + Dense(64) + softmax，定义在 `model.py`。
- 输入 224×224×3 → Conv(32)→BN→Pool → Conv(64)→BN→Pool → Conv(128)→BN→Pool → Conv(128)→BN→Pool → Flatten → Dense(64)→Dropout(0.5) → Dense(2, softmax)
- ~185 万参数，Adam(lr=0.001)，categorical_crossentropy
- 当前最佳验证准确率 **98.5%**（3669 张训练数据）

**目录结构**：

```
dataset/train/data/
├── labeled/                       # 原始图片 3539 张
├── label/                         # VOC XML 标注 3537 个（<object><name>mask/nomask</name></object>）
└── images_2class/                 # 训练数据
    ├── mask/                      # 656 张
    └── nomask/                    # 3013 张

models/
├── mask_classifier_binary.h5      # 训练好的 CNN 模型 (98.5%)
├── training_curve.png
├── face_mask_detection.prototxt   # SSD 模型结构 (Caffe)
└── face_mask_detection.caffemodel # SSD 预训练权重 (Caffe)

FaceMaskDetection-master/          # 参考项目（SSD 目标检测，含 Caffe/Keras/PyTorch/TF 预训练模型）
```

**数据流**：

```
labeled/ + label/ (XML)
       ↓ 自动导入或 label_images.py 标注
images_2class/ (mask/ + nomask/)
       ↓ train.py (20% validation_split + 数据增强)
models/mask_classifier_binary.h5
       ↓
predict.py / batch_predict.py / predict_realtime.py

SSD 检测流（端到端）：
摄像头/图片 → SSD Caffe 模型 (OpenCV DNN) → 人脸框 + Mask/NoMask
SSD 检测流（混合模式）：
摄像头/图片 → SSD 人脸框 → CNN 分类 (mask_classifier_binary.h5) → Mask/NoMask
```

## 关键细节

- **样本不均衡**：mask 656 张 vs nomask 3013 张（约 1:4.6），`train.py` 中 `compute_class_weights()` 已启用（mask ~2.36× 权重）。
- **训练数据来源**：3537 个 VOC XML 标注文件（`label/` 目录）对应 `labeled/` 中的图片，`<name>mask</name>` → mask，`<name>nomask</name>` → nomask。`label_images.py` 通过交互式 GUI 从 `labeled/` 标注并复制到 `images_2class/`。
- **SSD 人脸检测**：FaceMaskDetection-master 的 Caffe 模型可做人脸检测，模型文件在 `models/face_mask_detection.{prototxt,caffemodel}`。OpenCV 的 C++ 层不兼容中文路径，使用时需把模型复制到无中文路径。
- **验证集**：`validation_split=0.2` 从 `images_2class/` 自动切分，无单独验证目录。
- **数据增强**：训练时自动旋转(±20°)、平移(±15%)、缩放(±15%)、水平翻转、亮度(0.8~1.2)。
- **回调**：EarlyStopping(patience=5, monitor=val_loss, restore_best_weights) + ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)。
- **摄像头后端**：`predict_realtime.py` 打开摄像头时先尝试默认后端（MSMF），读 10 帧验证非纯黑（`mean > 10`），若黑屏则自动切到 DirectShow。不同摄像头的兼容后端不同。
  - `--detector ssd`（默认）：SSD 检测人脸框 → CNN 分类
  - `--detector ssd-e2e`：SSD 端到端（人脸+分类一次完成，0.03s/帧，跳过 CNN）
  - `--detector haar`：Haar 级联人脸框 → CNN 分类（传统方案）
- **prepare_dataset.py** 仅生成合成数据演示流程，使用真实数据时无需运行。
- **SSD 检测器**：`ssd_detector.py` 用 OpenCV DNN 加载 Caffe SSD 模型，输入 260×260，5 层特征图多尺度检测，输出人脸框 + 二分类。可独立使用：`python ssd_detector.py <图片>`。
- **日志系统**：`utils/logger.py` 提供统一日志（`get_logger()`），训练/测试脚本均使用。日志输出到 `logs/` 目录。
