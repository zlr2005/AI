"""题目三：Alpha-beta 剪枝（Minimax 与 Alpha-beta 求博弈树最佳得分）"""

import argparse
import json
from dataclasses import dataclass, field
from math import inf
from pathlib import Path
from time import perf_counter

from search import SearchStats


@dataclass
class TreeNode:
    value: int | None = None
    children: list["TreeNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 兼容显式传入 children=None 的调用
        if self.children is None:
            self.children = []


def build_sample_tree() -> TreeNode:
    max1 = TreeNode(children=[TreeNode(3), TreeNode(5)])
    max2 = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max3 = TreeNode(children=[TreeNode(1), TreeNode(2)])
    # 两个 MIN 节点共用 max2，搜索时按它在树中的每次出现分别访问
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


def tree_from_dict(data: dict, maximizing: bool = True) -> TreeNode:
    if "tree_structure" in data:
        data = data["tree_structure"]
    # 根节点为 MAX，后续各层的 MAX 和 MIN 交替
    if "is_maximizing_player" in data and data["is_maximizing_player"] is not maximizing:
        raise ValueError("根节点必须为 MAX，后续层的 MAX 与 MIN 必须交替")
    children = data.get("children", [])
    if children:
        return TreeNode(children=[tree_from_dict(child, not maximizing) for child in children])
    if type(data.get("value")) is not int:
        raise ValueError("叶子节点必须提供整数 value")
    return TreeNode(value=data["value"])


def solve(
    root: TreeNode,
    algorithm: str = "alpha_beta",
    order: str = "original",
) -> tuple[int, SearchStats]:
    # 默认从左到右访问，反向顺序用于剪枝对照
    if algorithm not in {"minimax", "alpha_beta"}:
        raise ValueError("算法应为 minimax 或 alpha_beta")
    if order not in {"original", "reverse"}:
        raise ValueError("访问顺序应为 original 或 reverse")
    stats = SearchStats(generated=1)
    begin = perf_counter()

    def visit(node: TreeNode, maximizing: bool, alpha: float, beta: float) -> int:
        # alpha 是 MAX 已能保证的下界，beta 是 MIN 已能限制的上界
        stats.visited += 1
        if not node.children:
            if type(node.value) is not int:
                raise ValueError("叶子节点必须具有整数得分")
            return node.value
        stats.expanded += 1
        children = node.children if order == "original" else list(reversed(node.children))
        best = -inf if maximizing else inf
        for index, child in enumerate(children):
            stats.generated += 1
            value = visit(child, not maximizing, alpha, beta)
            if maximizing:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)
            # 当 alpha >= beta 时，其余子分支已无法改善祖先的选择
            if algorithm == "alpha_beta" and alpha >= beta:
                stats.pruned += len(children) - index - 1
                break
        return int(best)

    value = visit(root, True, -inf, inf)
    stats.elapsed_ms = (perf_counter() - begin) * 1000
    return value, stats


def minimax(root: TreeNode) -> int:
    return solve(root, "minimax")[0]


def alpha_beta(root: TreeNode) -> int:
    return solve(root, "alpha_beta")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="给定结构的 JSON 文件")
    args = parser.parse_args()
    root = (
        tree_from_dict(json.loads(args.input.read_text(encoding="utf-8-sig")))
        if args.input else build_sample_tree()
    )
    print(alpha_beta(root))


if __name__ == "__main__":
    main()
