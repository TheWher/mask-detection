# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 生成合成数据集（800张人脸）
python dataset/prepare_dataset.py

# 训练模型（15轮 + 早停，模型保存到 models/mask_classifier.h5）
python train.py

# 单张预测
python predict.py <图片路径>

# 批量预测 + 导出 CSV
python batch_predict.py <图片目录> --output result.csv

# 运行测试套件（14项）
python test_system.py
```

## 架构概览

**任务**：图像二分类——判断图片中人脸是否佩戴口罩。

**模型**：4 层 CNN + Flatten + Dense(64) + sigmoid 输出。定义在 `model.py`，由 `train.py` 调用。

**标签分配**（由 `flow_from_directory` 字母序决定）：
- `with_mask/` → 标签 0
- `without_mask/` → 标签 1

**sigmoid 输出语义**：`prob > 0.5` 判为"未戴口罩"（标签1），否则"戴口罩"（标签0）。见 `predict.py:96-103`。

**数据流**：
```
prepare_dataset.py → dataset/train/ + dataset/test/
       ↓
   train.py → models/mask_classifier.h5 + training_curve.png
       ↓
predict.py / batch_predict.py → 输出预测结果
```

## 关键细节

- 训练前 `train.py` 会检查 `dataset/train/` 和 `dataset/test/` 是否存在，不存在则报错退出。
- `predict.py` 和 `batch_predict.py` 各自独立加载模型，均依赖 `models/mask_classifier.h5`。
- 合成数据训练的模型对真实照片泛化能力有限——合成图是简单几何人脸，真实照片需要更多样本或迁移学习。
- 使用真实数据时，直接将图片按类别放入 `dataset/train/with_mask/` 和 `dataset/train/without_mask/`，然后重新训练。
