"""题目一：迷宫寻路（使用 A* 返回迷宫最短路径）"""

import argparse
import json
from pathlib import Path

from search import Grid, PathResult, Position, astar

START = (0, 0)
END = (3, 4)
# S 为起点，E 为终点，X 为障碍，坐标从 0 开始
GRID = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]


def solve(start: Position, end: Position, grid: Grid, heuristic: str = "manhattan") -> PathResult:
    # 每步代价为 1，起点自身不计费
    return astar(grid, start, end, lambda cell: 1, heuristic)


def astar_path(start: Position, end: Position, grid: Grid) -> list[Position]:
    # 路径包含起终点，无解时返回空列表
    return solve(start, end, grid).path


def format_path(path: list[Position]) -> str:
    # 保持样例格式，坐标内不加空格，坐标之间用逗号和空格分隔
    return "[" + ", ".join(f"({row},{col})" for row, col in path) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="包含 start、end、grid 的 JSON 文件")
    args = parser.parse_args()
    if args.input:
        data = json.loads(args.input.read_text(encoding="utf-8-sig"))
        path = astar_path(tuple(data["start"]), tuple(data["end"]), data["grid"])
    else:
        path = astar_path(START, END, GRID)
    print(format_path(path))


if __name__ == "__main__":
    main()
