"""复现收益排序的适用范围，以及 A* 启发函数与同分规则的消融实验"""

import argparse
import copy
import csv
import json
import platform
import random
from pathlib import Path
from statistics import mean, median

from experiments import make_maze, make_row, measure
from question4 import TreasureGame
from search import astar


SEED = 20261022
CASE_COUNT = 100


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path, repeats: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    details = {"metadata": {
        "python": platform.python_version(), "platform": platform.platform(),
        "repeats": repeats, "seed": SEED, "value_cases": CASE_COUNT,
        "value_distribution": "T1,T2,T3,T4 依次独立均匀抽取 1..20 的整数",
        "timing": "每配置预热一次，重复计时取中位数；不含距离预计算、输出和跟踪；包含排序",
        "scope": "固定地图及宝藏位置，H=4，仅改变价值；不同地图与更多宝藏未在此验证",
        "trace": "只记录非终止节点；returned_value 结合 result_kind 解释，可能仅为上下界",
    }}

    # 在同一张地图上交叉比较启发函数和同分规则，核对最优代价是否一致
    ablation = []
    for case, grid in [("open_31", [["."] * 31 for _ in range(31)]),
                       ("walls_31", make_maze())]:
        for heuristic in ("zero", "manhattan"):
            for tie_break in ("fifo", "small_h"):
                result, elapsed = measure(
                    lambda: astar(grid, (0, 0), (30, 30), lambda cell: 1,
                                  heuristic, tie_break=tie_break), repeats)
                row = make_row(case, "astar", tie_break, result.cost, result.stats, elapsed)
                row["heuristic"] = heuristic
                ablation.append(row)
        expected = 60 if case == "open_31" else 340
        if any(r["value"] != expected for r in ablation if r["case"] == case):
            raise AssertionError("同分规则改变了最优路径代价")
    write_csv(output / "tie_break_ablation.csv", ablation)

    base = TreasureGame()
    rng = random.Random(SEED)
    rows, cases = [], []
    counts = {"fewer": 0, "equal": 0, "more": 0}
    default_visits, gain_visits = [], []
    default_times, gain_times = [], []
    first_failure = None
    for index in range(CASE_COUNT):
        # 各局面共用固定地图的距离表，只替换宝藏价值
        game = copy.copy(base)
        game.values = {target: rng.randint(1, 20) for target in base.values}
        case_name = f"value_{index + 1:03d}"
        results = {}
        for algorithm, order in [("minimax", "original"),
                                 ("alpha_beta", "original"), ("alpha_beta", "gain")]:
            (decision, stats), elapsed = measure(lambda: game.solve(4, algorithm, order), repeats)
            row = make_row(case_name, algorithm, order, decision["best_value"], stats, elapsed, 4)
            row.update(game.values)
            row["best_target"] = decision["best_target"]
            rows.append(row)
            results[algorithm, order] = (decision, stats, elapsed)
        reference = results["minimax", "original"][0]["best_value"]
        if any(result[0]["best_value"] != reference for result in results.values()):
            raise AssertionError(f"{case_name}: 最优值不一致")
        default = results["alpha_beta", "original"]
        gain = results["alpha_beta", "gain"]
        key = "fewer" if gain[1].visited < default[1].visited else "more" if gain[1].visited > default[1].visited else "equal"
        counts[key] += 1
        default_visits.append(default[1].visited)
        gain_visits.append(gain[1].visited)
        default_times.append(default[2])
        gain_times.append(gain[2])
        cases.append({"case": case_name, "values": game.values, "comparison": key,
                      "reference_value": reference})
        if key == "more" and first_failure is None:
            first_failure = {"case": case_name, "values": dict(game.values)}
    write_csv(output / "sorting_generalization.csv", rows)
    details["value_cases"] = cases
    details["sorting_summary"] = {
        "comparison_counts": counts,
        "mean_visited_default": mean(default_visits), "mean_visited_gain": mean(gain_visits),
        "reduction_of_mean_visited_pct": (1 - mean(gain_visits) / mean(default_visits)) * 100,
        "median_of_case_median_ms_default": median(default_times),
        "median_of_case_median_ms_gain": median(gain_times),
        "all_values_match_minimax": True,
    }
    # 单独追踪首个排序后访问量增加的局面，避免把跟踪开销计入计时
    if first_failure:
        game = copy.copy(base)
        game.values = first_failure["values"]
        initial = game.initial_state()
        first_failure["immediate_gain_a"] = {
            target: game.values[target] - game.distances["A", target]
            for target in initial.remaining}
        first_failure["exact_root_action_values"] = {
            target: game.solve(3, "minimax", state=game.move(initial, target))[0]["best_value"]
            for target in initial.remaining}
        first_failure["searches"] = {}
        for order in ("original", "gain"):
            trace = []
            decision, stats = game.solve(4, "alpha_beta", order, trace=trace)
            first_failure["searches"][order] = {
                "decision": decision, "visited": stats.visited, "trace": trace,
                "root_branches": [node for node in trace if len(node["path"]) == 1],
            }
        details["sorting_failure"] = first_failure
    details["tie_break_ablation"] = ablation
    (output / "innovation_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(details["sorting_summary"], ensure_ascii=False, indent=2))
    print(f"已保存 {output / 'tie_break_ablation.csv'}")
    print(f"已保存 {output / 'sorting_generalization.csv'}")
    print(f"已保存 {output / 'innovation_details.json'}")
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats 必须大于 0")
    run(args.output, args.repeats)


if __name__ == "__main__":
    main()
