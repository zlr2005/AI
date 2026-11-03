"""生成报告所需的性能对照数据与搜索过程数据"""

import argparse
import csv
import json
import platform
import random
from pathlib import Path
from statistics import median
from typing import Callable

import question1
import question2
import question3
from question4 import TreasureGame
from search import PathResult, SearchStats


def measure(operation: Callable, repeats: int):
    # 预热不计入计时，重复实验取中位数以减少瞬时波动
    operation()
    durations = []
    result = None
    for _ in range(repeats):
        result = operation()
        stats = result.stats if isinstance(result, PathResult) else result[1]
        durations.append(stats.elapsed_ms)
    return result, median(durations)


def make_row(case: str, algorithm: str, order: str, value, stats: SearchStats, elapsed: float, depth="") -> dict:
    return {
        "case": case,
        "algorithm": algorithm,
        "order": order,
        "depth": depth,
        "value": value,
        "visited": stats.visited,
        "expanded": stats.expanded,
        "generated": stats.generated,
        "pruned_branches": stats.pruned,
        "median_ms": round(elapsed, 6),
    }


def path_steps(result: PathResult, grid, entry_cost: Callable) -> list[dict]:
    accumulated = 0
    steps = []
    for index, (row, col) in enumerate(result.path):
        cost = 0 if index == 0 else entry_cost(grid[row][col])
        accumulated += cost
        steps.append({"position": [row, col], "entry_cost": cost, "accumulated_cost": accumulated})
    return steps


def make_maze(size: int = 31) -> list[list[str]]:
    # 交错墙体形成需要绕行的额外地图
    grid = [["." for _ in range(size)] for _ in range(size)]
    for wall_index, col in enumerate(range(2, size - 1, 3)):
        gap = size - 2 if wall_index % 2 == 0 else 1
        for row in range(size):
            if row != gap:
                grid[row][col] = "X"
    grid[0][0], grid[-1][-1] = "S", "E"
    return grid


def make_weighted_grid(size: int = 25) -> list[list[int | str]]:
    # 固定随机种子，让每次实验使用相同的地形权重
    random_generator = random.Random(20261015)
    grid = [[random_generator.choice([1, 1, 1, 3, 8]) for _ in range(size)] for _ in range(size)]
    grid[0][0], grid[-1][-1] = "S", "E"
    return grid


def continuation(game: TreasureGame, depth: int) -> list[dict]:
    state = game.initial_state()
    sequence = []
    while depth > 0 and state.remaining:
        # 从当前局面重新求解，并将选中的一步加入最优行动序列
        result, _ = game.solve(depth, "minimax", state=state)
        target = result["best_target"]
        if target is None:
            break
        source = state.position_a if state.current_player == "A" else state.position_b
        distance = game.distances[source, target]
        next_state = game.move(state, target)
        sequence.append({
            "player": state.current_player,
            "source": source,
            "target": target,
            "distance": distance,
            "gain": game.values[target] - distance,
            "score_a": next_state.score_a,
            "score_b": next_state.score_b,
            "utility": next_state.utility,
        })
        state = next_state
        depth -= 1
    return sequence


def run_experiments(output: Path, repeats: int) -> None:
    rows = []
    details = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "repeats": repeats,
            "seed": 20261015,
            "timing": "预热一次，记录算法内部计时的中位数，博弈计时不含距离矩阵预计算",
        },
        "paths": {},
    }
    maze = make_maze()
    path_cases = [
        ("q1_sample", question1.GRID, question1.START, question1.END),
        ("q1_open_31", [["."] * 31 for _ in range(31)], (0, 0), (30, 30)),
        ("q1_walls_31", maze, (0, 0), (30, 30)),
        ("q1_unreachable", [["S", "X", "E"]], (0, 0), (0, 2)),
    ]
    for name, grid, start, end in path_cases:
        for heuristic in ("zero", "manhattan"):
            result, elapsed = measure(lambda: question1.solve(start, end, grid, heuristic), repeats)
            rows.append(make_row(name, "astar", heuristic, result.cost, result.stats, elapsed))
            if name == "q1_sample" and heuristic == "manhattan":
                details["paths"][name] = path_steps(result, grid, lambda cell: 1)

    weighted = make_weighted_grid()
    for name, grid, start, end in [
        ("q2_sample", question2.GRID, question2.START, question2.END),
        ("q2_random_25", weighted, (0, 0), (24, 24)),
        ("q2_zero_cost", [["S", 0, 0, "E"]], (0, 0), (0, 3)),
    ]:
        for heuristic in ("zero", "weighted"):
            result, elapsed = measure(lambda: question2.solve(start, end, grid, heuristic), repeats)
            rows.append(make_row(name, "astar", heuristic, result.cost, result.stats, elapsed))
            if name == "q2_sample" and heuristic == "weighted":
                details["paths"][name] = path_steps(result, grid, question2.entry_cost)

    extra_tree = question3.TreeNode(children=[
        question3.TreeNode(children=[question3.TreeNode(3), question3.TreeNode(5)]),
        question3.TreeNode(children=[question3.TreeNode(2), question3.TreeNode(9)]),
    ])
    details["extra_tree"] = {"children": [{"children": [{"value": 3}, {"value": 5}]}, {"children": [{"value": 2}, {"value": 9}]}]}
    for name, root in [("q3_sample", question3.build_sample_tree()), ("q3_order_sensitive", extra_tree)]:
        for algorithm in ("minimax", "alpha_beta"):
            for order in ("original", "reverse"):
                (value, stats), elapsed = measure(lambda: question3.solve(root, algorithm, order), repeats)
                rows.append(make_row(name, algorithm, order, value, stats, elapsed))

    # 只预计算一次距离，后续各组博弈搜索共用同一张距离表
    game = TreasureGame()
    details["distance_labels"] = game.labels
    details["distance_matrix"] = game.distance_matrix()
    details["distance_precomputation"] = {
        "astar_calls": len(game.distance_stats),
        "visited": sum(stats.visited for stats in game.distance_stats),
        "expanded": sum(stats.expanded for stats in game.distance_stats),
        "elapsed_ms": sum(stats.elapsed_ms for stats in game.distance_stats),
    }
    details["decisions"] = {}
    details["root_action_values"] = {}
    details["principal_variations"] = {}
    for depth in (1, 2, 3, 4):
        initial = game.initial_state()
        # 分别计算每个首步的精确值，便于解释深度变化导致的决策差异
        details["root_action_values"][str(depth)] = {
            target: game.solve(depth - 1, "minimax", state=game.move(initial, target))[0]["best_value"]
            for target in initial.remaining
        }
        details["principal_variations"][str(depth)] = continuation(game, depth)
        for algorithm in ("minimax", "alpha_beta"):
            for order in ("original", "reverse", "gain"):
                (decision, stats), elapsed = measure(lambda: game.solve(depth, algorithm, order), repeats)
                rows.append(make_row("q4_sample", algorithm, order, decision["best_value"], stats, elapsed, depth))
                details["decisions"][f"H{depth}_{algorithm}_{order}"] = decision

    # 可采纳性反例直接记录同一个节点的估计值与真实剩余代价
    details["heuristic_counterexample"] = {
        "grid": [["S", "E"]],
        "true_cost": 0,
        "raw_manhattan": 1,
        "corrected_heuristic": 0,
        "explanation": "进入 E 的代价为 0，原始曼哈顿距离会高估，因此第二题使用修正的代价下界",
    }
    details["extra_inputs"] = {"maze_31": maze, "weighted_25": weighted}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "performance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {output / 'performance.csv'}")
    print(f"已生成 {output / 'details.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats 必须大于 0")
    run_experiments(args.output, args.repeats)


if __name__ == "__main__":
    main()
