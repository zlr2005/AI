"""四道题共用的 A* 搜索与性能统计"""

from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from time import perf_counter
from typing import Callable

Position = tuple[int, int]
Grid = Sequence[Sequence[int | str]]


@dataclass
class SearchStats:
    # 访问数包括终点或叶子，跳过的过期堆记录不计入
    visited: int = 0
    # 展开数只统计实际生成后继的节点
    expanded: int = 0
    # 生成数包括起点和同一坐标以更低代价重新入队
    generated: int = 0
    # 剪枝数只统计跳过的直接子分支
    pruned: int = 0
    elapsed_ms: float = 0.0


@dataclass
class PathResult:
    path: list[Position]
    # 无解时路径为空且总代价为 -1
    cost: int
    stats: SearchStats


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def validate_grid(grid: Grid, start: Position, end: Position) -> None:
    if not grid or not grid[0]:
        raise ValueError("地图不能为空")
    if any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图必须是矩形")
    for point in (start, end):
        if len(point) != 2 or any(type(x) is not int for x in point):
            raise ValueError("坐标必须包含两个整数")
        if not (0 <= point[0] < len(grid) and 0 <= point[1] < len(grid[0])):
            raise ValueError("起点或终点超出地图范围")


def terrain_cost(cell: int | str, *, free_end: bool = False) -> int:
    # 第二题进入 E 免费，第四题的特殊格按 1 计费
    if isinstance(cell, str):
        if cell == "E" and free_end:
            return 0
        if cell in {"S", "E", "A", "B"} or cell.startswith("T"):
            return 1
    # 排除布尔值，避免把 True 当成地形代价 1
    if isinstance(cell, bool):
        raise ValueError("地形代价必须为非负整数")
    try:
        cost = int(cell)
    except (TypeError, ValueError) as error:
        raise ValueError(f"无效地形: {cell!r}") from error
    if cost < 0 or (not isinstance(cell, str) and cost != cell):
        raise ValueError("地形代价必须为非负整数")
    return cost


def make_heuristic(
    grid: Grid,
    end: Position,
    entry_cost: Callable[[int | str], int],
    mode: str,
) -> Callable[[Position], int]:
    # h=0 时按累计代价搜索，等价于 Dijkstra 算法
    if mode == "zero":
        return lambda point: 0
    if mode == "manhattan":
        return lambda point: manhattan(point, end)
    if mode != "weighted":
        raise ValueError(f"未知启发函数: {mode}")

    # 加权下界把进入终点的最后一步单独计费，避免终点免费时高估
    other_costs = [
        entry_cost(cell)
        for row, cells in enumerate(grid)
        for col, cell in enumerate(cells)
        if cell != "X" and (row, col) != end
    ]
    minimum = min(other_costs, default=0)
    final_cost = entry_cost(grid[end[0]][end[1]])

    def heuristic(point: Position) -> int:
        distance = manhattan(point, end)
        # 距终点 d 步时，前 d-1 步各按非终点格的最低代价估算
        return 0 if distance == 0 else (distance - 1) * minimum + final_cost

    return heuristic


def astar(
    grid: Grid,
    start: Position,
    end: Position,
    entry_cost: Callable[[int | str], int],
    heuristic: str = "weighted",
    *,
    tie_break: str = "small_h",
) -> PathResult:
    begin = perf_counter()
    stats = SearchStats()
    if tie_break not in {"small_h", "fifo"}:
        raise ValueError("同分规则应为 small_h 或 fifo")

    def finish(path: list[Position], cost: int) -> PathResult:
        stats.elapsed_ms = (perf_counter() - begin) * 1000
        return PathResult(path, cost, stats)

    validate_grid(grid, start, end)
    if grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return finish([], -1)
    for cells in grid:
        for cell in cells:
            if cell != "X":
                # 搜索前检查所有可通行格的代价，包括可能不会走到的格子
                entry_cost(cell)
    estimate = make_heuristic(grid, end, entry_cost, heuristic)
    serial = count()
    start_h = estimate(start)
    # 堆按 f、次排序键、入队序号排序，次排序键在 small_h 下为 h，在 fifo 下为 0
    frontier = [(start_h, start_h if tie_break == "small_h" else 0, next(serial), 0, start)]
    best_g = {start: 0}
    parent: dict[Position, Position] = {}
    stats.generated = 1

    while frontier:
        _, _, _, current_g, current = heappop(frontier)
        # 找到更便宜的走法后，旧堆记录留到出队时跳过
        if current_g != best_g[current]:
            continue
        stats.visited += 1
        # 启发函数可采纳时，终点有效出队即可按父节点回溯最优路径
        if current == end:
            path = [end]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return finish(path, current_g)

        stats.expanded += 1
        # 固定按右、下、左、上展开，确保同一输入能复现同一路径
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            neighbor = (current[0] + dr, current[1] + dc)
            row, col = neighbor
            if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
                continue
            if grid[row][col] == "X":
                continue
            tentative_g = current_g + entry_cost(grid[row][col])
            # 只有发现更低的累计代价时才更新父节点并重新入队
            if tentative_g < best_g.get(neighbor, inf):
                best_g[neighbor] = tentative_g
                parent[neighbor] = current
                h = estimate(neighbor)
                secondary = h if tie_break == "small_h" else 0
                heappush(frontier, (tentative_g + h, secondary, next(serial), tentative_g, neighbor))
                stats.generated += 1
    return finish([], -1)
