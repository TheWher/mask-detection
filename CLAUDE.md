# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

口罩佩戴检测系统 — 基于 CNN 的图像二分类（戴口罩/不戴口罩）。课程设计项目，TensorFlow/Keras 实现，实习生水平。

## 常用命令

```bash
# 虚拟环境（解决 Windows Store Python 长路径问题）
C:/tmp/mask-venv/Scripts/python <脚本>.py

# 生成 800 张 224x224 合成人脸图像
python dataset/prepare_dataset.py

# 训练模型（15轮 Max + EarlyStopping，保存最佳权重到 models/）
python train.py

# 单张预测
python predict.py <图片路径>

# 批量预测（输出 CSV）
python batch_predict.py <图片目录> --output result.csv

# 运行 14 项功能测试
python test_system.py
```

## 架构要点

**模型** (`model.py`): 4 层卷积 (32→64→128→128) + BatchNorm + MaxPool，Flatten 后接 Dense(64) + Dropout(0.5) + Dense(1, sigmoid)。参数量 185 万，相比原 Dense(256) 方案减少 78%。`build_mask_classifier()` 返回未编译模型，`compile_model()` 编译。

**数据流**:
- `prepare_dataset.py` → 生成合成图到 `dataset/train/` 和 `dataset/test/`，按 `with_mask`/`without_mask` 分目录
- `train.py` → `ImageDataGenerator.flow_from_directory` 自动按目录名分配标签（字母序：with_mask=0, without_mask=1）
- `predict.py` / `batch_predict.py` → 加载 `models/mask_classifier.h5`，prob > 0.5 判为"未戴口罩"

**防过拟合策略**: EarlyStopping(patience=5) + ReduceLROnPlateau(patience=3, factor=0.5) + 数据增强（旋转 ±20°、平移 15%、亮度调整）

**测试** (`test_system.py`): 14 项用例覆盖数据完整性(TC01-04)、模型结构(TC05)、推理功能(TC06-07,11)、边界测试(TC08-10)、性能(TC12-13)、环境(TC14)。每个测试用例用 `@test_case("名称")` 装饰器标注。

**日志** (`utils/logger.py`): `Logger` 类支持控制台 + 文件双输出，`logger.section()` 打印分隔标题，`logger.result()` 输出结果行。

## 关键约定

- 图像尺寸统一 224×224 RGB，与训练时必须一致
- 合成数据的 `random_state=42` 保证可复现
- 类别映射由 `flow_from_directory` 字母序自动决定，不可手动修改
- 预测输出中文类别名：`{0: '戴口罩', 1: '未戴口罩'}`
- GitHub 仓库: `https://github.com/TheWher/mask-detection`
