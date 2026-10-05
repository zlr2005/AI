# 实验一：文本分类

## 仓库与数据目录

本实验代码放在课程仓库的 `Project1` 文件夹中，原始数据未上传到 GitHub。

上传代码时保留 `Project1` 这一层目录。下载仓库后，将课程提供的数据放到下面的位置即可；已有的结果文件不需要提前准备。

```text
课程仓库根目录/
├── Project1/
│   ├── run_experiment.py
│   ├── plot_training_losses.py
│   ├── README.md
│   ├── requirements.txt
│   └── output/                       # 运行脚本后自动创建
└── data-Project1/                    # 仅保存在本地，不上传到 GitHub
    └── data-Project1/
        ├── train_data.csv
        └── test_data_unlabeled.csv
```

其中，`Project1` 与外层的 `data-Project1` 必须位于同一级目录。数据压缩包解压后若已经形成两层同名目录，可直接将外层 `data-Project1` 放到课程仓库根目录；否则需要手动建立上述目录，并确认两个 CSV 文件位于内层 `data-Project1` 中。

脚本根据自身位置寻找数据，因此无须修改绝对路径。对应的相对路径为：

```text
../data-Project1/data-Project1/train_data.csv
../data-Project1/data-Project1/test_data_unlabeled.csv
```

## 环境与运行

为尽量复现实验报告中的结果，建议使用本次实验记录的 Python 3.13.9 和固定依赖版本。首次运行时可以创建独立的 Conda 环境：

```bash
conda create -n project1 python=3.13.9 -y
conda activate project1
```

进入课程仓库中的 `Project1` 文件夹后运行：

```bash
python -m pip install -r requirements.txt
python run_experiment.py
```

脚本也可以从其他工作目录启动，因为数据路径和输出路径均根据 `run_experiment.py` 所在位置确定。运行完成后，所有结果会保存在 `Project1/output/` 中。实际运行使用的依赖版本、随机种子和数据规模记录在 `output/run_metadata.json` 中。

如需根据逐轮记录额外绘制三类模型的训练目标曲线，可在主实验运行完成后执行：

```bash
python plot_training_losses.py
```

## 数据边界

- 原始有标签数据为 7,368 条，无标签测试数据为 2,457 条。
- 使用 `train_test_split(test_size=0.2, stratify=target, random_state=42)` 得到 5,894 条训练数据和 1,474 条验证数据。
- 划分后检查文本按大小写折叠和空白归一化的交集；训练、验证、测试之间的交集都必须为零，否则脚本终止。
- `TfidfVectorizer.fit_transform` 仅用于训练文本；验证文本和测试文本只经过 `transform`。
- 模型的 `partial_fit` 仅接收训练集。验证集用于逐轮评估、参数和轮次选择。无标签测试集只用于重叠检查和最终预测，从不参与模型选择。
- 最终模型使用选中的训练轮次，不把验证集或测试集重新并入训练集。

## 模型与评价

三种模型分别是 SVM（`SGDClassifier(loss="hinge")`）、逻辑回归（`SGDClassifier(loss="log_loss")`）和 MLP（`MLPClassifier`）。前两者是以随机梯度下降求解的线性分类器，因此可以逐轮记录训练过程；它们并非 `SVC` 或 `LogisticRegression` 类。每个模型尝试两组预先固定的参数，每组训练 20 轮。

每轮分别预测训练集和验证集。预测正确计 0，错误计 1，求和得到该轮的**预测错误数**。它是本实验约定的标签差距。脚本另存模型的实际训练目标（hinge loss、交叉熵或 MLP loss），不同目标值不直接横向比较。

模型选择先看验证集错误数，再看宏平均 F1，最后参考训练时间；同一组配置的最佳轮次也由验证集选择。验证集准确率、宏平均 F1、分类混淆矩阵和逐类准确率用于补充分析。创新对照仅改变 TF-IDF 的 `ngram_range`，由单词扩展为单词加二元词组；使用相同的数据划分、模型参数和训练轮数。

## 输出文件

| 文件                                                                 | 内容                                         |
| -------------------------------------------------------------------- | -------------------------------------------- |
| `output/train_indices.csv`, `validation_indices.csv`                 | 固定划分的原始行号                           |
| `output/epoch_history.csv`                                           | 所有配置逐轮的错误数、准确率、F1 和优化目标  |
| `output/model_comparison.csv`                                        | 每组配置的最佳验证轮次与性能                 |
| `output/error_curves.png`                                            | 三个模型训练集与验证集的逐轮错误数曲线       |
| `output/training_loss_curves.png`                                    | 三个模型各自训练目标的逐轮变化曲线           |
| `output/validation_confusion_matrix.csv`, `validation_per_class.csv` | 最终模型的验证集分类分析                     |
| `output/prediction.csv`                                              | 对无标签测试集的预测，单列、无表头、顺序不变 |
| `output/run_metadata.json`                                           | 随机种子、参数、环境版本、重叠检查和选择依据 |
