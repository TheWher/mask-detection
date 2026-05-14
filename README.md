# 口罩佩戴检测系统

基于深度学习的口罩佩戴检测——判断人脸是否佩戴口罩（二分类）。

## 项目简介

- **任务**：图像二分类 —— Mask / NoMask
- **模型**：4 层 CNN + GAP + Dense(2, softmax)，~25 万参数
- **人脸检测**：SSD Caffe 模型（OpenCV DNN，默认）+ Haar 级联（回退）
- **数据**：3669 张真实人脸图像（VOC XML 标注），mask 656 张 / nomask 3013 张
- **准确率**：最佳验证准确率 98.5%
- **推理模式**：单张 / 批量 / 实时摄像头

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.7+ |
| 框架 | TensorFlow / Keras |
| 人脸检测 | OpenCV DNN (SSD) / Haar 级联 |
| 图像处理 | PIL、OpenCV |
| 数据处理 | NumPy |
| 可视化 | Matplotlib、Grad-CAM |

## 项目结构

```
mask-detection/
├── gui_app.py                    # 图形化集成界面
├── model.py                      # CNN 模型定义
├── config.py                     # 共享配置 + 工具函数
├── train.py                      # 训练（含 CLI 参数）
├── predict.py                    # 单张预测
├── batch_predict.py              # 批量预测 + CSV
├── predict_realtime.py           # 实时检测（摄像头/图片）
├── ssd_detector.py               # SSD 检测器模块
├── report.py                     # 报告图表生成
├── export_tflite.py              # TFLite 模型导出
├── label_images.py               # 交互式标注工具
├── test_system.py                # 12 项功能测试
├── dataset/train/data/
│   ├── labeled/                  # 原始图片 3539 张
│   ├── label/                    # VOC XML 标注 3537 个
│   └── images_2class/            # 训练数据 (mask 656 / nomask 3013)
├── models/
│   ├── mask_classifier_binary.h5     # CNN 模型 (~3MB)
│   ├── face_mask_detection.prototxt  # SSD 结构 (Caffe)
│   ├── face_mask_detection.caffemodel# SSD 权重 (Caffe)
│   └── report_*.png / training_*.png # 报告图表
├── utils/
│   └── logger.py                 # 统一日志工具
└── logs/                         # 训练日志 + TensorBoard
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 图形化界面（推荐）

```bash
python gui_app.py
```

一个窗口搞定：训练、单张/批量预测、实时摄像头检测、报告图表、TFLite 导出。

### 3. 标注数据（如已有 images_2class 则跳过）

```bash
python label_images.py
```

按键：1 → mask，2 → nomask，s 跳过，q 保存退出。

### 4. 训练模型

```bash
python train.py
python train.py --batch-size 32 --epochs 30 --lr 0.0005
```

- 最大 15 轮，EarlyStopping 自动提前停止
- 自动保存最佳模型到 `models/mask_classifier_binary.h5`
- 自动生成训练曲线、混淆矩阵、数据集样本图、模型结构图
- 每轮指标自动写入 `models/training_log.csv`
- 已启用类别权重（mask ~2.36×）缓解样本不均衡

### 5. 单张预测

```bash
python predict.py <图片路径>
```

### 5. 批量预测

```bash
python batch_predict.py <图片目录> --output result.csv
```

### 6. 实时检测

```bash
# 摄像头 (SSD + CNN，默认)
python predict_realtime.py

# SSD 端到端（跳过 CNN，最快 ~0.03s/帧）
python predict_realtime.py --detector ssd-e2e

# Haar 回退模式
python predict_realtime.py --detector haar

# 单张图片
python predict_realtime.py --image test.jpg

# 保存结果 + 无 GUI
python predict_realtime.py --image test.jpg --save result.jpg --no-cam --headless
```

### 8. 生成报告

```bash
python report.py
```

生成：数据集样本网格 + 模型结构图 + 混淆矩阵（`models/report_*.png`）。

### 9. 导出 TFLite

```bash
python export_tflite.py
```

导出 FP16 / 动态量化 / INT8 三种格式，适合边缘设备部署。

### 10. 运行测试

```bash
python test_system.py
```

## 模型结构

| 层 | 输出尺寸 | 说明 |
|----|----------|------|
| Conv2D(32) + BN + MaxPool | 112×112×32 | 第 1 卷积块 |
| Conv2D(64) + BN + MaxPool | 56×56×64 | 第 2 卷积块 |
| Conv2D(128) + BN + MaxPool | 28×28×128 | 第 3 卷积块 |
| Conv2D(128) + BN + MaxPool | 14×14×128 | 第 4 卷积块 |
| GlobalAveragePooling2D | 128 | 替代 Flatten，大幅减少参数 |
| Dense(64) + Dropout(0.5) | 64 | 防过拟合 |
| Dense(2, softmax) | 2 | 二分类输出 |

## SSD 检测器

`ssd_detector.py` 使用 OpenCV DNN 加载 Caffe SSD 模型，一次推理同时输出人脸框和口罩分类。

| 模式 | 说明 | 速度 |
|------|------|------|
| `--detector ssd-e2e` | SSD 端到端（人脸+分类一次完成） | ~23 FPS |
| `--detector ssd` | SSD 取人脸框 → CNN 分类 | ~12 FPS |
| `--detector haar` | Haar 级联人脸框 → CNN 分类 | ~5 FPS |

## 训练配置

| 参数 | 值 |
|------|-----|
| 图像尺寸 | 224 × 224 |
| 批次大小 | 16 |
| 初始学习率 | 0.001 |
| 优化器 | Adam |
| 损失函数 | categorical_crossentropy |
| 验证集比例 | 20% |
| 数据增强 | 旋转 ±20°、平移 ±15%、缩放 ±15%、水平翻转、亮度 0.8~1.2 |
| 早停 | patience=5 (val_loss) |
| 学习率衰减 | factor=0.5, patience=3 |

## 常见问题

**Q: 数据集目录不存在？**
A: 运行 `python label_images.py` 标注图片，或直接将图片放入 `dataset/train/data/images_2class/mask/` 和 `nomask/` 目录。

**Q: 模型文件不存在？**
A: 先运行 `python train.py` 训练模型。

**Q: 摄像头黑屏？**
A: 优先 DirectShow 后端，自动验证非纯黑帧。如仍黑屏，检查摄像头是否被其他应用占用。

**Q: 实时检测卡顿？**
A: GUI 中使用 `ssd-e2e` 模式 + 推理间隔调为 1（每帧推理可达 23 FPS）。命令行用 `python predict_realtime.py --detector ssd-e2e`。

**Q: 预测置信度低？**
A: 尝试 `--detector ssd-e2e` 端到端模式，或增加训练数据量。
