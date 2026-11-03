"""题目二：加权地形（使用 A* 求加权地图的最小移动代价）"""

import argparse
import json
from pathlib import Path

from search import Grid, PathResult, Position, astar, terrain_cost

START = (0, 0)
END = (4, 4)
# 数字表示进入该格的代价
GRID = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]


def entry_cost(cell: int | str) -> int:
    # 进入 E 的最后一步免费，加权启发函数单独计算这一步
    return terrain_cost(cell, free_end=True)


def solve(start: Position, end: Position, grid: Grid, heuristic: str = "weighted") -> PathResult:
    return astar(grid, start, end, entry_cost, heuristic)


def astar_cost(start: Position, end: Position, grid: Grid) -> int:
    # 只返回最小总代价，无解时返回 -1
    return solve(start, end, grid).cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="包含 start、end、grid 的 JSON 文件")
    args = parser.parse_args()
    if args.input:
        data = json.loads(args.input.read_text(encoding="utf-8-sig"))
        result = astar_cost(tuple(data["start"]), tuple(data["end"]), data["grid"])
    else:
        result = astar_cost(START, END, GRID)
    print(result)


if __name__ == "__main__":
    main()
