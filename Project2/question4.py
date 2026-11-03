"""题目四：综合题（结合 A*、Minimax 与 Alpha-beta 争夺宝藏）"""

import argparse
import json
from dataclasses import dataclass
from math import inf
from pathlib import Path
from time import perf_counter

from search import Grid, Position, SearchStats, astar, terrain_cost

# 数字表示进入代价，X 为障碍，A、B 和宝藏格的进入代价为 1
GRID = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
POINTS = {"A": (0, 0), "B": (6, 6), "T1": (0, 6), "T2": (2, 2), "T3": (4, 4), "T4": (6, 0)}
VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}


# 状态不可变，各搜索分支独立创建后继状态
@dataclass(frozen=True)
class GameState:
    position_a: str
    position_b: str
    remaining: tuple[str, ...]
    score_a: int = 0
    score_b: int = 0
    current_player: str = "A"

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


class TreasureGame:
    def __init__(
        self,
        grid: Grid | None = None,
        points: dict[str, Position] | None = None,
        values: dict[str, int] | None = None,
    ) -> None:
        self.grid = GRID if grid is None else grid
        self.points = dict(POINTS if points is None else points)
        self.values = dict(VALUES if values is None else values)
        if "A" not in self.points or "B" not in self.points or not self.values:
            raise ValueError("必须提供 A、B 与宝藏信息")
        if not set(self.values).issubset(self.points):
            raise ValueError("每个宝藏都必须提供位置")
        if any(type(value) is not int for value in self.values.values()):
            raise ValueError("宝藏价值必须为整数")
        self.labels = ("A", "B", *sorted(self.values))
        self.distances: dict[tuple[str, str], int] = {}
        self.distance_stats: list[SearchStats] = []
        # 预计算有序点对的最小移动代价，博弈搜索只查表
        for source in self.labels:
            for target in self.labels:
                if source == target:
                    self.distances[source, target] = 0
                    continue
                result = astar(self.grid, self.points[source], self.points[target], terrain_cost)
                self.distances[source, target] = result.cost
                self.distance_stats.append(result.stats)

    def distance_matrix(self) -> list[list[int]]:
        # 矩阵的行对应起点，列对应终点
        return [[self.distances[source, target] for target in self.labels] for source in self.labels]

    def initial_state(self) -> GameState:
        return GameState("A", "B", tuple(sorted(self.values)))

    def legal_targets(self, state: GameState, order: str) -> list[str]:
        position = state.position_a if state.current_player == "A" else state.position_b
        # 只考虑尚未收集且可达的宝藏，距离 -1 表示不可达
        targets = [target for target in state.remaining if self.distances[position, target] >= 0]
        if order == "reverse":
            targets.reverse()
        elif order == "gain":
            # 双方都优先尝试即时净收益高的目标，排序只影响搜索顺序
            targets.sort(key=lambda target: -(self.values[target] - self.distances[position, target]))
        return targets

    def move(self, state: GameState, target: str) -> GameState:
        if target not in state.remaining:
            raise ValueError("不能再次获取已被拿走的宝藏")
        position = state.position_a if state.current_player == "A" else state.position_b
        distance = self.distances[position, target]
        if distance < 0:
            raise ValueError("目标宝藏不可达")
        gain = self.values[target] - distance
        # 只收集选中的宝藏，途经其他宝藏时不自动收集
        remaining = tuple(item for item in state.remaining if item != target)
        if state.current_player == "A":
            return GameState(target, state.position_b, remaining, state.score_a + gain, state.score_b, "B")
        return GameState(state.position_a, target, remaining, state.score_a, state.score_b + gain, "A")

    def solve(
        self,
        depth: int,
        algorithm: str = "alpha_beta",
        order: str = "original",
        state: GameState | None = None,
        *,
        trace: list[dict] | None = None,
    ) -> tuple[dict[str, str | int | None], SearchStats]:
        # 深度按单次行动计数，H=2 表示 A 和 B 各行动一次
        if type(depth) is not int or depth < 0:
            raise ValueError("搜索深度必须为非负整数")
        if algorithm not in {"minimax", "alpha_beta"}:
            raise ValueError("算法应为 minimax 或 alpha_beta")
        if order not in {"original", "reverse", "gain"}:
            raise ValueError("未知的子节点访问顺序")
        stats = SearchStats(generated=1)
        begin = perf_counter()

        def visit(current: GameState, remaining_depth: int, alpha: float, beta: float, path: tuple[str, ...] = ()) -> tuple[int, str | None]:
            stats.visited += 1
            entered_at = stats.visited
            alpha_in, beta_in = alpha, beta
            if remaining_depth == 0 or not current.remaining:
                # 深度耗尽或宝藏取完时，返回当前收益差
                return current.utility, None
            targets = self.legal_targets(current, order)
            if not targets:
                # 剩余目标均不可达时，也按当前收益差终止
                return current.utility, None
            stats.expanded += 1
            # A 最大化收益差，B 最小化收益差
            maximizing = current.current_player == "A"
            best_value = -inf if maximizing else inf
            best_target = None
            explored_targets = [] if trace is not None else None
            skipped_targets = []
            for index, target in enumerate(targets):
                stats.generated += 1
                value, _ = visit(self.move(current, target), remaining_depth - 1, alpha, beta,
                                 path + (target,) if trace is not None else ())
                if explored_targets is not None:
                    explored_targets.append(target)
                improves = value > best_value if maximizing else value < best_value
                # 得分相同时保留先访问的目标
                if improves:
                    best_value, best_target = value, target
                if maximizing:
                    alpha = max(alpha, best_value)
                else:
                    beta = min(beta, best_value)
                # 只有 Alpha-beta 在搜索窗口闭合时跳过剩余分支
                if algorithm == "alpha_beta" and alpha >= beta:
                    stats.pruned += len(targets) - index - 1
                    skipped_targets = targets[index + 1:]
                    break
            if trace is not None:
                # 用进入子树时的搜索窗口区分上下界与精确值
                result_kind = "exact"
                if algorithm == "alpha_beta":
                    if best_value <= alpha_in:
                        result_kind = "upper_bound"
                    elif best_value >= beta_in:
                        result_kind = "lower_bound"
                trace.append({
                    "path": list(path), "player": current.current_player,
                    "remaining_depth": remaining_depth,
                    "alpha_in": alpha_in if alpha_in != -inf else "-inf",
                    "beta_in": beta_in if beta_in != inf else "+inf",
                    "returned_value": int(best_value), "result_kind": result_kind,
                    "visited_subtree": stats.visited - entered_at + 1,
                    "explored_targets": explored_targets, "skipped_targets": skipped_targets,
                })
            return int(best_value), best_target

        value, target = visit(self.initial_state() if state is None else state, depth, -inf, inf)
        stats.elapsed_ms = (perf_counter() - begin) * 1000
        return {"best_target": target, "best_value": value}, stats


def format_decision(depth: int, result: dict) -> str:
    target = json.dumps(result["best_target"], ensure_ascii=False)
    # 数值宽度为 2，使正数前保留样例要求的对齐空格
    return f'H = {depth}   =>  {{"best_target": {target}, "best_value": {result["best_value"]:2d}}}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="包含 grid、points、values 的 JSON 文件")
    parser.add_argument("--matrix", action="store_true", help="输出距离矩阵")
    parser.add_argument("--algorithm", choices=("minimax", "alpha_beta"), default="alpha_beta")
    parser.add_argument("--depth", type=int, choices=(2, 4), help="只输出指定深度的结果")
    args = parser.parse_args()
    if args.input:
        data = json.loads(args.input.read_text(encoding="utf-8-sig"))
        game = TreasureGame(data["grid"], {key: tuple(value) for key, value in data["points"].items()}, data["values"])
    else:
        game = TreasureGame()
    if args.matrix:
        print(json.dumps(game.distance_matrix()))
    else:
        for depth in ([args.depth] if args.depth is not None else [2, 4]):
            result, _ = game.solve(depth, args.algorithm)
            print(format_decision(depth, result))


if __name__ == "__main__":
    main()
