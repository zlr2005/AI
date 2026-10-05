"""
本脚本可以从任意工作目录运行
无标签测试集仅用于检查数据是否重叠，并在模型选择完成后生成最终预测
"""

from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data-Project1" / "data-Project1"
OUT = ROOT / "output"
SEED = 42
EPOCHS = 20
MAX_FEATURES = 20_000


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    alpha: float
    hidden: int | None = None


CONFIGS = [
    # 每类模型设置两组候选参数，便于在相同数据划分下公平比较
    Config("svm_a1e-4", "Linear SVM", 1e-4),
    Config("svm_a1e-5", "Linear SVM", 1e-5),
    Config("logreg_a1e-4", "Logistic regression", 1e-4),
    Config("logreg_a1e-5", "Logistic regression", 1e-5),
    Config("mlp_h64", "MLP", 1e-4, 64),
    Config("mlp_h128", "MLP", 1e-4, 128),
]


def normalized(text: str) -> str:
    # 统一空白、大小写，避免格式差异影响重复文本检查
    return re.sub(r"\s+", " ", text).strip().casefold()


def audit_disjoint(train: pd.Series, val: pd.Series, test: pd.Series) -> dict:
    """Fail before fitting if any text crosses a data boundary."""
    # 分别建立训练集、验证集和测试集的规范化文本集合
    sets = [set(map(normalized, part.astype(str))) for part in (train, val, test)]
    checks = {
        "train_validation_overlap": len(sets[0] & sets[1]),
        "train_test_overlap": len(sets[0] & sets[2]),
        "validation_test_overlap": len(sets[1] & sets[2]),
    }
    # 只要任意两个集合存在重复文本就终止运行，防止数据泄漏
    if any(checks.values()):
        raise ValueError(f"Text overlap across data splits: {checks}")
    return checks


def vectorize(train: pd.Series, val: pd.Series, ngram_range: tuple[int, int]):
    # 配置 TF-IDF，将文本转换为适合分类器处理的稀疏特征矩阵
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=MAX_FEATURES,
        min_df=2,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    # 词表和 IDF 只在训练集上拟合，避免验证集信息泄漏
    x_train = vectorizer.fit_transform(train)
    # 验证集仅使用训练阶段得到的转换器生成特征
    x_val = vectorizer.transform(val)
    return vectorizer, x_train, x_val


def make_model(config: Config):
    # 根据配置创建 SVM、逻辑回归和 MLP
    if config.family == "Linear SVM":
        return SGDClassifier(loss="hinge", alpha=config.alpha, random_state=SEED)
    if config.family == "Logistic regression":
        return SGDClassifier(loss="log_loss", alpha=config.alpha, random_state=SEED)
    return MLPClassifier(
        hidden_layer_sizes=(config.hidden,),
        alpha=config.alpha,
        solver="adam",
        batch_size=128,
        learning_rate_init=0.001,
        random_state=SEED,
        early_stopping=False,
    )


def fit_record(config, x_train, y_train, x_val, y_val, classes, feature_set):
    # 为当前参数配置初始化模型、随机数生成器和训练记录
    model = make_model(config)
    rng = np.random.default_rng(SEED)
    rows = []
    best = None
    best_key = None
    best_epoch = None
    best_model = None
    start = time.perf_counter()

    # 按固定轮数迭代训练，并在每轮结束后进行验证评估
    for epoch in range(1, EPOCHS + 1):
        # 每轮随机打乱训练样本，降低固定顺序对优化过程的影响
        order = rng.permutation(len(y_train))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            # partial_fit 每次接收完整训练集，验证集不参与参数更新
            model.partial_fit(x_train[order], y_train[order], classes=classes)

        # 分别预测训练集和验证集，用于观察拟合与泛化表现
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        # 错误数采用预测标签与真实标签不一致的样本总数
        train_errors = int(np.count_nonzero(train_pred != y_train))
        val_errors = int(np.count_nonzero(val_pred != y_val))
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))

        # 根据模型类型记录实际优化目标，避免将其与预测错误数混淆
        if config.family == "Linear SVM":
            objective_name = "hinge"
            from sklearn.metrics import hinge_loss

            objective = float(hinge_loss(y_train, model.decision_function(x_train), labels=classes))
        elif config.family == "Logistic regression":
            objective_name = "cross_entropy"
            objective = float(log_loss(y_train, model.predict_proba(x_train), labels=classes))
        else:
            objective_name = "mlp_loss"
            objective = float(model.loss_)

        rows.append({
            "feature_set": feature_set,
            "config": config.name,
            "family": config.family,
            "epoch": epoch,
            "train_errors": train_errors,
            "validation_errors": val_errors,
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "validation_accuracy": float(accuracy_score(y_val, val_pred)),
            "validation_macro_f1": val_f1,
            "objective_name": objective_name,
            "training_objective": objective,
        })

        # 优先选择验证错误最少的轮次，其次选择宏平均 F1 更高的轮次
        # 若两个指标完全相同，则保留更早的轮次
        key = (val_errors, -val_f1, epoch)
        if best_key is None or key < best_key:
            import copy

            best_key = key
            best = rows[-1].copy()
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        print(f"{feature_set:8s} {config.name:15s} epoch {epoch:2d}: "
              f"train errors={train_errors:4d}, val errors={val_errors:4d}, "
              f"val F1={val_f1:.4f}", flush=True)

    elapsed = time.perf_counter() - start
    best["training_seconds"] = elapsed
    best["best_epoch"] = best_epoch
    best["alpha"] = config.alpha
    best["hidden_units"] = config.hidden
    best["n_features"] = x_train.shape[1]
    return rows, best, best_model


def plot_error_curves(history: pd.DataFrame, summary: pd.DataFrame, path: Path):
    # 加载 Windows 中文字体，保证输出图片中的中文正常显示
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    baseline = summary[summary.feature_set == "unigram"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    palette = {"Linear SVM": "#176a95", "Logistic regression": "#cc6d22", "MLP": "#558b42"}
    chinese_names = {"Linear SVM": "线性 SVM", "Logistic regression": "逻辑回归", "MLP": "MLP"}
    # 每类模型仅绘制验证表现最好的参数配置
    for family in ("Linear SVM", "Logistic regression", "MLP"):
        choice = baseline[baseline.family == family].sort_values(
            ["validation_errors", "validation_macro_f1"], ascending=[True, False]
        ).iloc[0]
        part = history[(history.feature_set == "unigram") & (history.config == choice.config)]
        for ax, column, title in zip(
            axes,
            ("train_errors", "validation_errors"),
            ("训练集错误数（5,894 条）", "验证集错误数（1,474 条）"),
        ):
            ax.plot(part.epoch, part[column], marker="o", markersize=2.5,
                    linewidth=1.8, label=chinese_names[family], color=palette[family])
            ax.set_title(title)
            ax.set_xlabel("训练轮次")
            ax.set_ylabel("预测错误数（0/1 损失之和）")
            ax.grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("三种模型的逐轮预测错误数（固定且互不重叠的数据划分）")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    # 创建结果目录并读取有标签训练集和无标签测试集
    OUT.mkdir(exist_ok=True)
    train_df = pd.read_csv(DATA / "train_data.csv")
    test_df = pd.read_csv(DATA / "test_data_unlabeled.csv")
    # 检查数据列结构和缺失值，尽早发现输入文件异常
    assert list(train_df.columns) == ["text", "target"]
    assert list(test_df.columns) == ["text"]
    assert not train_df.isna().any().any() and not test_df.isna().any().any()

    # 按类别比例进行 80% 训练集和 20% 验证集的分层划分
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, stratify=train_df.target,
        random_state=SEED,
    )
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
    train_text = train_df.text.iloc[train_idx]
    val_text = train_df.text.iloc[val_idx]
    y_train = train_df.target.iloc[train_idx].to_numpy()
    y_val = train_df.target.iloc[val_idx].to_numpy()
    classes = np.sort(train_df.target.unique())
    # 检查三个数据集合之间是否存在重复文本
    audit = audit_disjoint(train_text, val_text, test_df.text)
    print("Disjointness audit:", audit, flush=True)

    # 保存划分索引，使实验使用的数据划分可以复现和核查
    pd.DataFrame({"row_index": train_idx}).to_csv(OUT / "train_indices.csv", index=False)
    pd.DataFrame({"row_index": val_idx}).to_csv(OUT / "validation_indices.csv", index=False)

    history_rows = []
    summaries = []
    best_models = {}
    # 首先使用单词级 TF-IDF 特征训练并比较全部候选配置
    uni_vectorizer, x_train, x_val = vectorize(train_text, val_text, (1, 1))
    for config in CONFIGS:
        rows, best, model = fit_record(
            config, x_train, y_train, x_val, y_val, classes, "unigram"
        )
        history_rows.extend(rows)
        summaries.append(best)
        best_models[("unigram", config.name)] = model

    # 创新对照只改变 ngram_range，保持数据划分、模型参数和训练轮数不变
    base_summary = pd.DataFrame(summaries)
    base_winner = base_summary.sort_values(
        ["validation_errors", "validation_macro_f1", "training_seconds"],
        ascending=[True, False, True],
    ).iloc[0]
    winning_config = next(c for c in CONFIGS if c.name == base_winner.config)
    bi_vectorizer, x_train_bi, x_val_bi = vectorize(train_text, val_text, (1, 2))
    rows, best, bi_model = fit_record(
        winning_config, x_train_bi, y_train, x_val_bi, y_val, classes, "uni+bigram"
    )
    history_rows.extend(rows)
    summaries.append(best)
    best_models[("uni+bigram", winning_config.name)] = bi_model

    # 汇总并保存逐轮训练记录、模型比较结果和错误曲线
    history = pd.DataFrame(history_rows)
    summary = pd.DataFrame(summaries)
    history.to_csv(OUT / "epoch_history.csv", index=False)
    summary.to_csv(OUT / "model_comparison.csv", index=False)
    plot_error_curves(history, summary, OUT / "error_curves.png")

    # 按验证错误数、宏平均 F1 和训练时间依次选出最终模型
    winner = summary.sort_values(
        ["validation_errors", "validation_macro_f1", "training_seconds"],
        ascending=[True, False, True],
    ).iloc[0]
    vectorizer = bi_vectorizer if winner.feature_set == "uni+bigram" else uni_vectorizer
    model = best_models[(winner.feature_set, winner.config)]
    chosen_x_val = x_val_bi if winner.feature_set == "uni+bigram" else x_val
    # 计算最终模型在验证集上的混淆矩阵和逐类别准确率
    val_pred = model.predict(chosen_x_val)
    cm = confusion_matrix(y_val, val_pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT / "validation_confusion_matrix.csv")
    per_class = []
    for i, label in enumerate(classes):
        n = int(np.count_nonzero(y_val == label))
        per_class.append({"label": int(label), "validation_count": n,
                          "correct": int(cm[i, i]), "errors": n - int(cm[i, i]),
                          "accuracy": float(cm[i, i] / n)})
    pd.DataFrame(per_class).to_csv(OUT / "validation_per_class.csv", index=False)

    # 测试集从未参与训练和选模，此处仅用于生成最终预测
    x_test = vectorizer.transform(test_df.text)
    predictions = model.predict(x_test)
    assert len(predictions) == len(test_df)
    assert set(predictions).issubset(set(classes))
    pd.Series(predictions).to_csv(OUT / "prediction.csv", index=False, header=False)

    # 记录随机种子、数据规模、参数和软件版本，确保实验可复现
    metadata = {
        "seed": SEED,
        "train_count": len(train_idx),
        "validation_count": len(val_idx),
        "unlabeled_test_count": len(test_df),
        "overlap_audit": audit,
        "split": "80/20 stratified; validation and test never used for fit",
        "tfidf": {"lowercase": True, "max_features": MAX_FEATURES,
                  "min_df": 2, "sublinear_tf": True},
        "epochs_per_run": EPOCHS,
        "selection": "fewest validation 0/1 errors; then higher macro F1; then shorter training time",
        "winner": {
            key: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
            for key, value in winner.to_dict().items()
        },
        "versions": {"python": __import__("sys").version.split()[0],
                     "sklearn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "configs": [asdict(c) for c in CONFIGS],
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Winner:", winner.to_dict(), flush=True)
    print("Outputs:", OUT, flush=True)


if __name__ == "__main__":
    main()
