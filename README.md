# Project-1: MNIST Classification with NumPy-only MLP & CNN

复旦大学《神经网络与深度学习》Project-1 提交代码。所有核心算子（`Linear`、`conv2D`、
`MaxPool2D`、`MultiCrossEntropyLoss`、`MomentGD`、`MultiStepLR`、`ExponentialLR`）
均基于 NumPy 手写实现，未调用 PyTorch / TensorFlow 等深度学习库。

- **报告**：见仓库根目录的 `报告.pdf` / `报告.html`（HTML 内嵌全部图片，单文件自包含）
- **训练好的权重**：从 ModelScope 下载 → `[ModelScope link]`
- **作者**：`[姓名]`（学号 `[学号]`）

## 目录结构

```
codes/
├── README.md, .gitignore
├── dataset_explore.ipynb              # 数据探索 notebook
│
├── mynn/                              # 框架核心
│   ├── op.py                          #   Linear / conv2D / MaxPool2D / Flatten / ReLU / MultiCrossEntropyLoss
│   ├── models.py                      #   Model_MLP / Model_CNN
│   ├── optimizer.py                   #   SGD / MomentGD
│   ├── lr_scheduler.py                #   StepLR / MultiStepLR / ExponentialLR
│   ├── runner.py                      #   训练循环 + batched eval
│   ├── metric.py                      #   accuracy
│   └── __init__.py
├── draw_tools/                        # 绘图工具（plot.py + 手写画板 draw.py）
├── dataset/MNIST/*.gz                 # 数据（gitignored；自行准备）
│
├── test_train_mlp_fair.py             # 训练 MLP 基线
├── test_train_cnn.py                  # 训练 CNN（默认 MomentGD + MultiStepLR）
├── test_train.py                      # PDF starter 提供的 MLP 训练入口（保留备用，本提交未使用）
├── test_model.py                      # 在测试集上评估任意 --tag
├── analysis.py                        # Part C 方向 5：混淆矩阵 / 误分类 / 权重 / 卷积核
├── generate_report.py                 # 一键生成 ../报告.html
├── upload_to_modelscope.py            # 一键上传 4 个 checkpoint + 报告
├── weight_visualization.py            # PDF optional 模板，保留未用（实际可视化走 analysis.py）
├── hyperparameter_search.py           # PDF optional 模板，保留未用
│
└── outputs/                           # 所有训练/分析产物（gitignored）
    ├── idx.pickle                     #   60k → 50k train + 10k val 的固定 split
    ├── best_models/<tag>/best_model.pickle    # checkpoint（mlp_fair / cnn / cnn_sgd / cnn_sgd_ms 等）
    ├── history/history_<tag>.pickle           # 训练历史
    └── figs/                          #   curve_*.png / cm_*.png / wrong_*.png / weights_*.png / kernels_*.png / curve_all.png
```

## 复现

### 0. 准备数据
把官方 MNIST 的 4 个 `.gz` 文件放到 `dataset/MNIST/`。

### 1. 训练（4 次实验）
所有 4 组实验共享相同的训练 budget：MomentGD/SGD lr=0.05、batch=64、3 epochs、weight_decay λ=1e-4。
```bash
python test_train_mlp_fair.py                                                              # MLP 基线
python test_train_cnn.py --optimizer momentum --scheduler multistep --save-tag cnn         # CNN 主模型
python test_train_cnn.py --optimizer sgd      --scheduler none      --save-tag cnn_sgd     # 方向 1 对照
python test_train_cnn.py --optimizer sgd      --scheduler multistep --save-tag cnn_sgd_ms  # 方向 1 单变量隔离
```
checkpoint 落到 `outputs/best_models/<tag>/best_model.pickle`，history 落到 `outputs/history/`，曲线图落到 `outputs/figs/`。

### 2. 测试集评估
```bash
python test_model.py --tag mlp_fair      # 97.77%
python test_model.py --tag cnn           # 98.89%
python test_model.py --tag cnn_sgd       # 98.12%
python test_model.py --tag cnn_sgd_ms    # 97.99%
```

### 3. Part C 可视化（方向 5）
```bash
python analysis.py --tag mlp_fair
python analysis.py --tag cnn
```
图落到 `outputs/figs/`。

### 4. 重生成报告
```bash
python generate_report.py
# 然后在 PJ1/ 根目录：
python -c "from weasyprint import HTML; HTML(filename='报告.html').write_pdf('报告.pdf')"
```

## 关键结果

| 实验 | 优化器 | scheduler | best dev | test |
|---|---|---|---|---|
| MLP | MomentGD μ=0.9 lr=0.05 b=64 e=3 | MultiStepLR | 97.53% | 97.77% |
| **CNN** | 同上 | MultiStepLR | 98.96% | **98.89%** |
| CNN (SGD, no sched) | SGD lr=0.05 b=64 e=3 | — | 98.02% | 98.12% |
| CNN (SGD+MS) | SGD lr=0.05 b=64 e=3 | MultiStepLR | 97.90% | 97.99% |

详细分析见报告。
