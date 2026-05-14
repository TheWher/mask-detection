# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 图形化集成界面（训练/预测/实时检测/报告/导出）
python gui_app.py

# 训练（自动生成报告图表 + TensorBoard + CSV 日志）
python train.py
python train.py --batch-size 32 --epochs 30 --lr 0.0005

# 单张预测
python predict.py <图片路径>

# 批量预测 → CSV
python batch_predict.py <图片目录> --output result.csv

# 实时检测
python predict_realtime.py                           # 摄像头 (SSD+CNN)
python predict_realtime.py --detector ssd-e2e        # SSD端到端（最快）
python predict_realtime.py --detector haar           # Haar回退
python predict_realtime.py --image test.jpg --save result.jpg --no-cam --headless

# 报告图表（训练后自动生成，也可手动）
python report.py

# TFLite 导出（FP16 / 动态量化 / INT8）
python export_tflite.py

# 标注工具（1=mask 2=nomask, s跳过 d无效 b回退 q保存）
python label_images.py

# 测试套件
python test_system.py

# TensorBoard
tensorboard --logdir logs/tensorboard
```

## 架构概览

**任务**：图像二分类 — 判断人脸是否佩戴口罩（Mask / NoMask）。

**人脸检测**：默认使用 SSD Caffe 模型（OpenCV DNN 推理），可选 Haar 级联回退。SSD 支持端到端模式（人脸+口罩一次完成）和混合模式（SSD 取人脸框 + CNN 分类）。

**二类标签**（flow_from_directory 按字母序分配）：

| 目录 | 标签 | 含义 |
|------|------|------|
| `mask/` | 0 | 佩戴口罩 |
| `nomask/` | 1 | 未佩戴口罩 |

**模型**：4 层 CNN + GlobalAveragePooling + Dense(64) + softmax，定义在 `model.py`。
- 输入 224×224×3 → Conv(32)→BN→Pool → Conv(64)→BN→Pool → Conv(128)→BN→Pool → Conv(128)→BN→Pool → GAP → Dense(64)→Dropout(0.5) → Dense(2, softmax)
- ~25 万参数（GAP 替代 Flatten，比原方案减少 87%），Adam(lr=0.001)，categorical_crossentropy
- 最佳验证准确率 **98.5%**（3669 张训练数据，733 张验证集）

**模块关系**：

```
config.py              # 共享路径/常量(CONF_THRESH/IOU_THRESH/SSD_INPUT_SIZE等)/load_trained_model/preprocess_image — 所有脚本从这里导入
model.py               # 模型定义 + 编译（build_mask_classifier / compile_model）
train.py               # 训练入口 → 自动调 report.py 生成全部报告图表
predict.py             # 单张推理（从 config 导入模型加载和预处理）
batch_predict.py       # 批量推理 + CSV 导出
predict_realtime.py    # 实时检测（MaskDetector 类封装三种检测器）
ssd_detector.py        # SSD Caffe 推理 + NMS（cv2.dnn.NMSBoxes）
report.py              # 报告生成：样本网格 + 模型结构图 + 混淆矩阵
export_tflite.py       # H5 → TFLite 转换（FP16/动态量化/INT8）
gui_app.py             # tkinter 集成界面（5 标签页）
label_images.py        # OpenCV 交互式标注工具
test_system.py         # 12 项功能测试
utils/logger.py        # 统一日志工具
```

**目录结构**：

```
dataset/train/data/
├── labeled/                       # 原始图片 3539 张
├── label/                         # VOC XML 标注 3537 个
└── images_2class/                 # 训练数据
    ├── mask/                      # 656 张
    └── nomask/                    # 3013 张

models/
├── mask_classifier_binary.h5      # CNN 模型 (~3MB)
├── training_curve.png             # 训练曲线
├── training_log.csv               # 每轮指标 CSV
├── report_*.png                   # 报告图表（样本/结构/混淆矩阵）
├── face_mask_detection.prototxt   # SSD 结构 (Caffe)
└── face_mask_detection.caffemodel # SSD 权重 (Caffe)

logs/
├── train.log                      # 最新训练日志
└── tensorboard/                   # TensorBoard 事件文件

FaceMaskDetection-master/          # SSD 模型参考实现（上游项目，非本项目的模块）
```

**数据流**：

```
labeled/ + label/ (XML)
       ↓ label_images.py 标注或自动导入
images_2class/ (mask/ + nomask/)
       ↓ train.py (20% validation_split + 数据增强)
models/mask_classifier_binary.h5
       ↓
predict.py / batch_predict.py / predict_realtime.py / gui_app.py

SSD 检测流（端到端）：
摄像头/图片 → SSD Caffe 模型 (OpenCV DNN) → 人脸框 + Mask/NoMask
SSD 检测流（混合模式）：
摄像头/图片 → SSD 人脸框 → CNN 分类 → Mask/NoMask
```

## 关键细节

- **样本不均衡**：mask 656 vs nomask 3013（1:4.6），`train.py` 中 `compute_class_weights()` 用 sqrt 缩放 + 上限 3.0（mask ~2.36× 权重）。
- **训练数据来源**：`label/` 目录 3537 个 VOC XML 对应 `labeled/` 中的图片，`<name>mask</name>` → mask，`<name>nomask</name>` → nomask。`label_images.py` 从 `labeled/` 标注并复制到 `images_2class/`。
- **验证集**：`validation_split=0.2` + `seed=42` 从 `images_2class/` 自动切分。训练和验证使用分离的 ImageDataGenerator（训练有增强，验证仅归一化）。
- **数据增强**：训练时旋转(±20°)、平移(±15%)、缩放(±15%)、水平翻转、亮度(0.8~1.2)。
- **回调**：ModelCheckpoint(best val_acc) + EarlyStopping(patience=5, restore_best) + ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6) + CSVLogger → `models/training_log.csv` + TensorBoard（如已安装则自动启用）。
- **SSD NMS**：`ssd_detector.py` 使用 `cv2.dnn.NMSBoxes`（C++ 后端），注意返回的索引必须映射回原始数组（之前有 bug 导致返回内部过滤数组索引，已修复）。
- **SSD 人脸检测**：模型文件在 `models/face_mask_detection.{prototxt,caffemodel}`。OpenCV 的 C++ 层不兼容中文路径，使用时需把模型复制到无中文路径。
- **摄像头后端**：优先 DirectShow（`cv2.CAP_DSHOW`），读 10 帧验证非纯黑（`mean > 10`），失败则尝试 Default 后端。
- **共享配置**：`config.py` 集中管理所有路径、常量、`load_trained_model()`、`preprocess_image()`。各脚本通过 `from config import ...` 引用，避免重复定义。
- **Grad-CAM**：`predict_realtime.py` 中已修复重复推理 bug — `face_arr_exp` 从第一次 `_cnn_classify()` 缓存复用，不再调两次 CNN。
- **TFLite 导出**：`export_tflite.py` 支持 FP16（~11MB）、动态量化（~6MB）、INT8（~5MB）三种格式。INT8 需校准数据（自动从训练集采样 100 张）。
- **GUI 架构**（`gui_app.py`）：**主线程**只跑 tkinter UI 刷新（`after(33)` ~30fps），**工作线程**独立跑摄像头读取 + 模型推理 + 画框，通过 `queue.Queue(maxsize=1)` 传画面（只保留最新帧，消除延迟）。`threading.Event` 控制线程安全退出。识别结果缓存 `cached_dets` 确保非推理帧也显示框。推理间隔可通过界面旋钮实时调节（1=每帧推理，2=隔1帧，默认2）。切换检测器无需重启摄像头。
