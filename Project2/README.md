# 实验二：搜索算法

## 环境与依赖

解题、测试和性能实验使用 Python 3.10 或更高版本，仅依赖 Python 标准库，无需运行 `pip install`。报告中的原始结果采集于 Windows 11、Python 3.12.14。解题、测试与性能实验已在 Windows 上验证。以下命令以 Windows PowerShell 为例，在本 README 所在目录执行。先用 `python --version` 核对解释器版本。

## 数据与目录

四道题的样例内置于 `question1.py` 至 `question4.py`。性能实验使用内置样例和脚本生成的对照用例，无需下载数据。自定义输入可保存为当前目录下的 UTF-8 JSON 文件，再通过 `--input` 传入。坐标统一写成从 0 开始的 `[行, 列]`。

| 题目 | JSON 字段及格式                                                                                                                                                            |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 一   | `start`、`end` 为坐标，`grid` 为矩形二维数组。使用 `S`、`E`、`.` 和障碍 `X`                                                                                                |
| 二   | `start`、`end`、`grid` 同上，普通地形填非负整数或整数字符串，表示进入代价，进入 `E` 免费                                                                                   |
| 三   | 直接传根节点对象，或将根节点放入 `tree_structure`。内部节点用 `children` 保存子节点，叶子用整数 `value` 保存得分。可用 `is_maximizing_player` 标记当前层，根节点为最大化方 |
| 四   | `grid` 为矩形二维数组，使用非负地形权重、障碍 `X` 和 `A`、`B`、`T1` 等标记。`points` 包含 A、B 和每个宝藏的坐标，`values` 只列出宝藏及其整数价值                           |

例如，将 `{"start":[0,0],"end":[0,2],"grid":[["S",".","E"]]}` 保存为 `input.json`，再运行 `python question1.py --input input.json`。

### 仓库目录

在 GitHub 仓库中，实验二的文件放在 `Project2/` 目录下。运行下文的命令前，先进入 `Project2/`。

```text
Project2/
├── README.md                  # 环境、运行命令和结果核验方法
├── .gitignore                 # 排除缓存、虚拟环境和本地复现产物
├── question1.py               # 题目一，使用 A* 求迷宫最短路径
├── question2.py               # 题目二，使用加权 A* 求最小移动代价
├── question3.py               # 题目三，构建样例博弈树并用 Alpha-beta 求最佳得分
├── question4.py               # 题目四，输出距离矩阵并用两种算法求最优首步
├── search.py                  # 共用 A*、地形计费、启发函数和搜索统计
├── test_search.py             # 验证四题样例、边界情况和随机用例
├── experiments.py             # 生成四题的基础性能对照和过程数据
├── innovation_experiments.py  # 生成收益排序与 A* 同分规则的补充实验数据
└── results/                   # 报告使用的原始实验结果
    ├── performance.csv              # 46 组基础性能记录
    ├── details.json                 # 路径累计代价、距离矩阵和最优行动序列
    ├── sorting_generalization.csv   # 100 组宝藏价值、每组 3 种配置的结果
    ├── tie_break_ablation.csv       # 2 张地图上启发函数与同分规则的 8 组对照
    └── innovation_details.json      # 补充实验配置、汇总和排序失败追踪
```

## 运行顺序

1. 核验内置样例、自定义输入和边界情况。

   ```powershell
   python -m unittest -v test_search
   ```

2. 运行四道题，取得路径、代价、博弈得分和宝藏决策。第四题另外支持输出距离矩阵。

   ```powershell
   python question1.py
   python question2.py
   python question3.py
   python question4.py
   python question4.py --matrix
   ```

   若需单独核验第四题 H=4 的 Minimax 结果，可运行 `python question4.py --algorithm minimax --depth 4`。

3. 运行基础实验和补充实验，评估搜索正确性与开销。复现数据写入 `results_reproduced/`，便于和报告原始数据对照。

   ```powershell
   python experiments.py --repeats 100 --output results_reproduced
   python innovation_experiments.py --repeats 100 --output results_reproduced
   ```

本实验直接运行搜索算法，无需划分训练集、训练模型或生成预测文件。

## 实验配置

| 实验     | 算法与关键配置                                                                                                                                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 题目一   | 四方向 A\*，步长代价为 1，比较曼哈顿距离与 `h=0`。使用样例、31×31 开放地图、交错墙体地图和无解地图                                                                                                                |
| 题目二   | 加权 A\*，比较修正加权启发函数与 `h=0`。使用样例、种子 `20261015` 生成的 25×25 加权地图及零代价样例                                                                                                               |
| 题目三   | 比较 Minimax、Alpha-beta 及正向、反向访问顺序。使用样例树和一棵对顺序敏感的对照树                                                                                                                                 |
| 题目四   | 先用 A\* 预计算点对距离，再比较 Minimax 与 Alpha-beta。实验深度为 H=1 至 4，访问顺序为默认、反向和即时收益排序。命令行默认用 Alpha-beta 输出 H=2、4，收益按宝藏价值减去移动代价计算，评价值为 A 与 B 的累计收益差 |
| 补充实验 | 排序实验固定地图、宝藏位置和 H=4，用种子 `20261022` 生成 100 组宝藏价值，每个价值独立从 1 至 20 的整数中均匀抽取。A\* 消融在开放地图和墙体地图上交叉比较 `zero/manhattan` 与 `fifo/small_h`                       |

每种配置预热 1 次，重复计时 100 次，取算法内部耗时的中位数。计时包含启发函数初始化和访问顺序排序的相应开销。距离预计算单独统计，文件输出和失败追踪在计时之外。

## 结果与提交

性能指标为最优代价或最优值、访问数 `visited`、展开数 `expanded`、生成数 `generated`、剪掉的直接子分支数 `pruned_branches` 和中位耗时 `median_ms`。A\* 的生成数包括起点和重新入队，访问数只统计有效出队。博弈搜索的访问数包括叶子。

| 产物     | 位置与格式                                                                                                                                                                 |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 四题输出 | 标准输出。依次为坐标列表、整数最小代价、整数博弈得分、含 `best_target` 和 `best_value` 的逐深度决策行。第四题加 `--matrix` 时输出 JSON 二维数组                            |
| 基础实验 | `results/performance.csv` 含 46 组性能记录，`results/details.json` 含路径累计代价、距离矩阵、首步精确值和最优行动序列                                                      |
| 补充实验 | `results/sorting_generalization.csv` 含 300 条逐局配置记录，`results/tie_break_ablation.csv` 含 8 条消融记录，`results/innovation_details.json` 含配置、汇总和失败局面追踪 |

## 复现与限制

测试应显示 `Ran 26 tests` 和 `OK`。测试用 BFS、Bellman-Ford 和朴素递归核验结果，并检查命令行输出格式。四题默认输出依次如下。

```text
[(0,0), (0,1), (0,2), (1,2), (1,3), (1,4), (2,4), (3,4)]
7
5
H = 2   =>  {"best_target": "T2", "best_value": -1}
H = 4   =>  {"best_target": "T3", "best_value":  3}
```

第四题的距离矩阵应为 6×6，行列顺序均为 `A, B, T1, T2, T3, T4`，可与 `results/details.json` 的 `distance_matrix` 核对。将复现 CSV 与原始 CSV 按配置逐行对比，最优值及节点统计应一致。耗时受机器与运行环境影响，可忽略 `median_ms` 的差异。

- 题目一无解时输出 `[]`，题目二无解时输出 `-1`。加权 A\* 支持非负整数代价，出现零代价格时启发函数可能减弱。题目四若当前玩家无可达宝藏，按当前收益差终止搜索
- H=2 只评估后续两次行动，H=4 才覆盖样例的四次行动。等值首步可能随访问顺序变化，核验时以最优收益差为准
- 收益排序的 100 组实验中，访问量减少 97 组、持平 2 组、增加 1 组。第 44 组从 41 个节点增至 43 个节点，追踪见 `innovation_details.json` 的 `sorting_failure`。统计范围为固定地图和四个宝藏
