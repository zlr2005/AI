"""根据 output/epoch_history.csv 中的记录绘制各模型的训练损失曲线"""

from pathlib import Path

import matplotlib

# 使用非交互式后端，直接保存图片而不弹出绘图窗口
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


# 以脚本所在目录为基准，设置训练记录和输出图片的路径
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "output" / "epoch_history.csv"
OUTPUT = ROOT / "output" / "training_loss_curves.png"


def main():
    # 读取训练记录，只保留单词 TF-IDF 特征对应的实验数据
    data = pd.read_csv(INPUT)
    data = data[data["feature_set"] == "unigram"].copy()

    # 加载微软雅黑字体，让图中的中文标签正常显示
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    # 使用普通减号，避免负号在部分字体下显示异常
    plt.rcParams["axes.unicode_minus"] = False

    # 设置每个子图的模型名称、标题、纵轴标签和配置图例
    panels = [
        ("Linear SVM", "线性 SVM", "合页损失", {
            "svm_a1e-4": "α=1e-4", "svm_a1e-5": "α=1e-5"
        }),
        ("Logistic regression", "逻辑回归", "交叉熵损失", {
            "logreg_a1e-4": "α=1e-4", "logreg_a1e-5": "α=1e-5"
        }),
        ("MLP", "MLP", "MLP 训练损失", {
            "mlp_h64": "隐藏层 64", "mlp_h128": "隐藏层 128"
        }),
    ]
    colors = ["#176a95", "#cc6d22"]
    # 创建一行三个子图，分别展示三种模型的训练损失
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (family, title, ylabel, labels) in zip(axes, panels):
        # 筛选当前模型的数据，并依次绘制两组参数配置的曲线
        part = data[data["family"] == family]
        for color, (config, label) in zip(colors, labels.items()):
            # 按训练轮次排序，并检查每组配置是否包含完整的 20 轮记录
            curve = part[part["config"] == config].sort_values("epoch")
            if len(curve) != 20:
                raise ValueError(f"{config}: expected 20 epochs, found {len(curve)}")
            ax.plot(
                curve["epoch"], curve["training_objective"],
                marker="o", markersize=3, linewidth=1.8,
                color=color, label=label,
            )
        # 设置子图标题、坐标轴、网格和图例
        ax.set_title(title)
        ax.set_xlabel("训练轮次")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    # 调整整体布局，保存高清图片并释放绘图资源
    fig.suptitle("三种模型的训练损失曲线（单词 TF-IDF）", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
