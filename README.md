# 口罩佩戴检测系统

基于深度学习的口罩佩戴检测系统（图像二分类），判断图片中的人是否佩戴口罩。

## 项目简介

- **任务**: 图像二分类 —— 戴口罩 vs 不戴口罩
- **模型**: 自定义轻量级 CNN（3层卷积 + GlobalAveragePooling + 早停）
- **数据**: 800 张合成人脸图像
- **特点**: 代码量小、注释全、一键运行、测试套件完善

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.7+ |
| 框架 | TensorFlow / Keras |
| 图像处理 | PIL (Pillow) |
| 数据处理 | NumPy、Scikit-learn |
| 可视化 | Matplotlib |

## 项目结构

```
mask-detection/
├── dataset/
│   ├── prepare_dataset.py       # 数据集生成脚本（224x224）
│   ├── train/                   # 训练集（640张，2类各320）
│   └── test/                    # 测试集（160张，2类各80）
├── models/
│   ├── mask_classifier.h5       # 训练好的模型
│   └── training_curve.png       # 训练曲线图
├── logs/                        # 运行日志
│   └── train.log
├── utils/
│   ├── __init__.py
│   └── logger.py                # 统一日志工具
├── model.py                     # CNN 模型定义
├── train.py                     # 训练脚本（15轮 + 早停）
├── predict.py                   # 单张图片预测
├── batch_predict.py             # 批量预测 + CSV导出
├── test_system.py               # 14项功能测试套件
├── test_report.txt              # 测试报告（自动生成）
├── requirements.txt             # 依赖清单
├── report.md                    # 设计报告
└── README.md                    # 本文件
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

> 如使用 Windows Store 版 Python 遇长路径错误，建议创建虚拟环境：
> ```bash
> python -m venv C:/tmp/mask-venv
> C:/tmp/mask-venv/Scripts/pip install -r requirements.txt
> ```

### 2. 生成数据集

```bash
python dataset/prepare_dataset.py
```

生成 800 张 224x224 合成人脸图像，自动划分训练/测试集。

### 3. 模型训练

```bash
python train.py
```

- 最大 15 轮，EarlyStopping 自动提前停止
- 自动保存最佳模型到 `models/mask_classifier.h5`
- 生成训练曲线 `models/training_curve.png`

### 4. 单张预测

```bash
python predict.py <图片路径>
python predict.py dataset/test/with_mask/with_mask_0001.jpg
```

### 5. 批量预测

```bash
python batch_predict.py <图片目录>
python batch_predict.py ./test_images/ --output result.csv
```

### 6. 运行测试

```bash
python test_system.py
```

## 模型结构

| 层 | 类型 | 输出尺寸 | 说明 |
|----|------|----------|------|
| Conv2D(32) + BN + MaxPool | 卷积块 | 112x112x32 | 第1组 |
| Conv2D(64) + BN + MaxPool | 卷积块 | 56x56x64 | 第2组 |
| Conv2D(128) + BN + MaxPool | 卷积块 | 28x28x128 | 第3组 |
| Conv2D(128) + BN + MaxPool | 卷积块 | 14x14x128 | 第4组 |
| Flatten | 展平 | 25088 | - |
| Dense(64) + Dropout(0.5) | 全连接 | 64 | 防过拟合 |
| Dense(1, sigmoid) | 输出 | 1 | 二分类 |

> 参数量约 185 万（相比原方案 850 万减少 78%）

## 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像尺寸 | 224 x 224 | 统一输入尺寸 |
| 批次大小 | 16 | 适配 224 图像 |
| 初始学习率 | 0.001 | Adam 优化器 |
| 学习率衰减 | ReduceLROnPlateau | 3轮不降则减半 |
| 早停策略 | patience=5 | 5轮不降即停止 |

## 测试用例说明

| 编号 | 测试项 | 类型 |
|------|--------|------|
| TC01 | 数据集目录结构 | 完整性 |
| TC02 | 数据集规模(800张) | 验证 |
| TC03 | 类别平衡性 | 验证 |
| TC04 | 图像格式与尺寸(224x224) | 验证 |
| TC05 | 模型结构(输入/输出/参数量) | 结构 |
| TC06 | 戴口罩样本推理 | 功能 |
| TC07 | 不戴口罩样本推理 | 功能 |
| TC08 | 文件不存在处理 | 边界 |
| TC09 | 损坏图片容错 | 边界 |
| TC10 | 命令行参数检查 | 边界 |
| TC11 | 批量预测功能 | 功能 |
| TC12 | 单张推理性能 | 性能 |
| TC13 | 训练收敛性(3轮>80%) | 性能 |
| TC14 | 系统环境信息 | 环境 |

## 常见问题

**Q: 数据集目录不存在？**
A: 先运行 `python dataset/prepare_dataset.py`

**Q: 模型文件不存在？**
A: 先运行 `python train.py` 训练模型

**Q: 如何使用真实数据？**
A: 将图片按分类放入 `dataset/train/with_mask/` 等对应目录，保持 JPG/PNG 格式即可

**Q: 预测置信度低？**
A: 合成数据训练出的模型泛化能力有限，真实数据建议增加样本量或使用迁移学习
