"""使用独立参考算法验证正确性、状态更新与输出格式"""

import json
import random
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

import question1
import question2
import question3
from question3 import TreeNode
from question4 import GameState, TreasureGame
from search import astar, make_heuristic, terrain_cost

BASE = Path(__file__).resolve().parent


def breadth_first_distance(grid, start, end):
    if grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return -1
    # 无权地图按步数逐层搜索，用首次到达终点的距离作为对照
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        point, distance = queue.popleft()
        if point == end:
            return distance
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = point[0] + dr, point[1] + dc
            row, col = neighbor
            if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != "X" and neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return -1


def bellman_ford_distance(grid, start, end, entry_cost):
    # 独立使用 Bellman-Ford，避免用同一份 A* 实现验证自身
    if grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return -1
    vertices = [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell != "X"]
    distances = {point: float("inf") for point in vertices}
    distances[start] = 0
    for _ in range(len(vertices) - 1):
        changed = False
        for row, col in vertices:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = row + dr, col + dc
                if neighbor not in distances:
                    continue
                candidate = distances[row, col] + entry_cost(grid[neighbor[0]][neighbor[1]])
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    changed = True
        if not changed:
            break
    return -1 if distances[end] == float("inf") else int(distances[end])


def reference_tree(node, maximizing=True):
    if not node.children:
        return node.value
    values = [reference_tree(child, not maximizing) for child in node.children]
    return max(values) if maximizing else min(values)


# 朴素递归穷举合法行动，为宝藏搜索提供独立的结果对照
def reference_game(game, a, b, remaining, score_a, score_b, player, depth):
    if depth == 0 or not remaining:
        return score_a - score_b
    values = []
    source = a if player == "A" else b
    for target in remaining:
        distance = game.distances[source, target]
        if distance < 0:
            continue
        rest = tuple(item for item in remaining if item != target)
        gain = game.values[target] - distance
        if player == "A":
            values.append(reference_game(game, target, b, rest, score_a + gain, score_b, "B", depth - 1))
        else:
            values.append(reference_game(game, a, target, rest, score_a, score_b + gain, "A", depth - 1))
    if not values:
        return score_a - score_b
    return max(values) if player == "A" else min(values)


class PathTests(unittest.TestCase):
    def test_tie_break_variants_against_independent_oracles(self):
        generator = random.Random(20261022)
        for case in range(30):
            choices = [0, 1, 3, "X"] if case % 2 == 0 else [1, 1, 3, "X"]
            grid = [[generator.choice(choices) for _ in range(5)] for _ in range(5)]
            grid[0][0], grid[-1][-1] = "S", "E"
            expected = bellman_ford_distance(grid, (0, 0), (4, 4), question2.entry_cost)
            for rule in ("fifo", "small_h"):
                with self.subTest(case=case, rule=rule):
                    result = astar(grid, (0, 0), (4, 4), question2.entry_cost, tie_break=rule)
                    self.assertEqual(result.cost, expected)
                    if result.path:
                        self.assertEqual(sum(question2.entry_cost(grid[r][c]) for r, c in result.path[1:]), expected)
        with self.assertRaises(ValueError):
            astar([["S", "E"]], (0, 0), (0, 1), question2.entry_cost, tie_break="invalid")

    def test_sample_path_and_cost(self):
        path = question1.astar_path(question1.START, question1.END, question1.GRID)
        self.assertEqual(path, [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)])
        result = question2.solve(question2.START, question2.END, question2.GRID)
        self.assertEqual(result.cost, 7)
        self.assertEqual(sum(question2.entry_cost(question2.GRID[r][c]) for r, c in result.path[1:]), 7)

    def test_unreachable_and_blocked_endpoints(self):
        grid = [["S", "X", "E"]]
        self.assertEqual(question1.astar_path((0, 0), (0, 2), grid), [])
        self.assertEqual(question2.astar_cost((0, 0), (0, 2), grid), -1)
        self.assertEqual(question1.astar_path((0, 1), (0, 1), grid), [])
        self.assertEqual(question2.astar_cost((0, 0), (0, 1), grid), -1)

    def test_same_start_end(self):
        self.assertEqual(question1.astar_path((0, 0), (0, 0), [["S"]]), [(0, 0)])
        self.assertEqual(question2.astar_cost((0, 0), (0, 0), [["E"]]), 0)

    def test_zero_cost_and_string_weights(self):
        self.assertEqual(question2.astar_cost((0, 0), (0, 3), [["S", "0", "0", "E"]]), 0)
        self.assertEqual(question2.astar_cost((0, 0), (0, 1), [["S", "E"]]), 0)
        self.assertEqual(question2.astar_cost((0, 0), (0, 1), [[1, 8]]), 8)

    def test_weighted_detour(self):
        grid = [["S", 9, "E"], [1, 1, 1]]
        result = question2.solve((0, 0), (0, 2), grid)
        self.assertEqual(result.cost, 3)
        self.assertEqual(len(result.path), 5)

    def test_invalid_inputs(self):
        for grid, start, end in [([], (0, 0), (0, 0)), ([[1], [1, 2]], (0, 0), (1, 0)), ([[1]], (-1, 0), (0, 0))]:
            with self.assertRaises(ValueError):
                question2.astar_cost(start, end, grid)
        for bad_cost in (-1, "bad", True):
            with self.assertRaises(ValueError):
                question2.astar_cost((0, 0), (0, 1), [["S", bad_cost]])

    def test_random_mazes_against_bfs(self):
        generator = random.Random(1001)
        for case in range(80):
            with self.subTest(case=case):
                grid = [["X" if generator.random() < 0.3 else "." for _ in range(7)] for _ in range(6)]
                grid[0][0], grid[-1][-1] = "S", "E"
                result = question1.solve((0, 0), (5, 6), grid)
                self.assertEqual(result.cost, breadth_first_distance(grid, (0, 0), (5, 6)))
                if result.path:
                    self.assertEqual(len(result.path) - 1, result.cost)
                    for a, b in zip(result.path, result.path[1:]):
                        self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
                        self.assertNotEqual(grid[b[0]][b[1]], "X")

    def test_random_weights_against_bellman_ford(self):
        generator = random.Random(2002)
        for case in range(80):
            with self.subTest(case=case):
                choices = [0, 1, 2, 8, "X"] if case % 2 == 0 else [1, 2, 8, "X"]
                grid = [[generator.choice(choices) for _ in range(5)] for _ in range(5)]
                grid[0][0] = "S"
                grid[-1][-1] = "E" if case % 3 else generator.choice([0, 1, 8])
                result = question2.solve((0, 0), (4, 4), grid)
                self.assertEqual(result.cost, bellman_ford_distance(grid, (0, 0), (4, 4), question2.entry_cost))
                if result.path:
                    self.assertEqual(result.cost, sum(question2.entry_cost(grid[r][c]) for r, c in result.path[1:]))

    def test_weighted_heuristic_admissibility_and_consistency(self):
        grids = [question2.GRID, [["S", 0, 1], [2, 3, "E"]], [[1, 1], [2, 8]]]
        for grid in grids:
            end = len(grid) - 1, len(grid[0]) - 1
            h = make_heuristic(grid, end, question2.entry_cost, "weighted")
            self.assertEqual(h(end), 0)
            for r, row in enumerate(grid):
                for c, cell in enumerate(row):
                    if cell == "X":
                        continue
                    point = r, c
                    distance = bellman_ford_distance(grid, point, end, question2.entry_cost)
                    if distance >= 0:
                        self.assertLessEqual(h(point), distance)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < len(grid) and 0 <= nc < len(row) and grid[nr][nc] != "X":
                            self.assertLessEqual(h(point), question2.entry_cost(grid[nr][nc]) + h((nr, nc)))


class TreeTests(unittest.TestCase):
    def test_sample_and_pruning(self):
        root = question3.build_sample_tree()
        for order in ("original", "reverse"):
            minimax_value, minimax_stats = question3.solve(root, "minimax", order)
            value, stats = question3.solve(root, "alpha_beta", order)
            self.assertEqual((minimax_value, value), (5, 5))
            self.assertLessEqual(stats.visited, minimax_stats.visited)
        self.assertGreater(question3.solve(root)[1].pruned, 0)

    def test_single_leaf_and_none_children(self):
        self.assertEqual(question3.alpha_beta(TreeNode(-3, None)), -3)

    def test_teacher_json_input(self):
        data = {"tree_structure": {"is_maximizing_player": True, "children": [{"value": 2}, {"value": 5}]}}
        self.assertEqual(question3.alpha_beta(question3.tree_from_dict(data)), 5)
        with self.assertRaises(ValueError):
            question3.tree_from_dict({"is_maximizing_player": False, "value": 1})

    def test_random_trees(self):
        generator = random.Random(3003)

        def make_tree(depth):
            if depth == 0 or generator.random() < 0.2:
                return TreeNode(generator.randint(-20, 20))
            return TreeNode(children=[make_tree(depth - 1) for _ in range(generator.randint(1, 4))])

        for case in range(60):
            with self.subTest(case=case):
                root = make_tree(4)
                expected = reference_tree(root)
                for algorithm in ("minimax", "alpha_beta"):
                    for order in ("original", "reverse"):
                        self.assertEqual(question3.solve(root, algorithm, order)[0], expected)


class TreasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = TreasureGame()

    def test_sample_decisions(self):
        for algorithm in ("minimax", "alpha_beta"):
            for depth, target, value in [(2, "T2", -1), (4, "T3", 3)]:
                result, _ = self.game.solve(depth, algorithm)
                self.assertEqual(result, {"best_target": target, "best_value": value})

    def test_sorting_failure_trace_and_exact_action_values(self):
        game = TreasureGame.__new__(TreasureGame)
        game.__dict__.update(self.game.__dict__)
        game.values = {"T1": 18, "T2": 15, "T3": 3, "T4": 20}
        initial = game.initial_state()
        expected = reference_game(game, "A", "B", initial.remaining, 0, 0, "A", 4)
        action_values = {}
        for target in initial.remaining:
            child = game.move(initial, target)
            action_values[target] = reference_game(game, child.position_a, child.position_b,
                child.remaining, child.score_a, child.score_b, "B", 3)
        for order in ("original", "gain"):
            trace = []
            plain, plain_stats = game.solve(4, "alpha_beta", order)
            result, stats = game.solve(4, "alpha_beta", order, trace=trace)
            self.assertEqual(result, plain)
            self.assertEqual(stats.visited, plain_stats.visited)
            self.assertEqual(result["best_value"], expected)
            self.assertEqual(action_values[result["best_target"]], expected)
            branches = [node for node in trace if len(node["path"]) == 1]
            self.assertEqual(1 + sum(node["visited_subtree"] for node in branches), stats.visited)
            self.assertEqual(next(node for node in trace if not node["path"])["result_kind"], "exact")

    def test_distance_matrix_against_independent_oracle(self):
        for source in self.game.labels:
            for target in self.game.labels:
                expected = bellman_ford_distance(self.game.grid, self.game.points[source], self.game.points[target], terrain_cost)
                self.assertEqual(self.game.distances[source, target], expected)

    def test_state_update_and_cost_conventions(self):
        state = self.game.initial_state()
        updated = self.game.move(state, "T2")
        self.assertEqual((updated.position_a, updated.position_b, updated.score_a, updated.score_b, updated.current_player), ("T2", "B", 2, 0, "B"))
        self.assertIn("T2", state.remaining)
        self.assertNotIn("T2", updated.remaining)
        second = self.game.move(updated, "T3")
        self.assertEqual((second.position_a, second.position_b, second.score_a, second.score_b, second.current_player), ("T2", "T3", 2, 3, "A"))
        with self.assertRaises(ValueError):
            self.game.move(updated, "T2")
        self.assertEqual(terrain_cost("T2"), 1)
        self.assertEqual(question2.entry_cost("E"), 0)

    def test_passing_treasure_does_not_collect_it(self):
        # A 到 T3 的最短路经过 T2，但只能移除选中的 T3
        updated = self.game.move(self.game.initial_state(), "T3")
        self.assertEqual(updated.remaining, ("T1", "T2", "T4"))

    def test_depth_zero_and_no_treasures(self):
        self.assertEqual(self.game.solve(0)[0], {"best_target": None, "best_value": 0})
        state = GameState("T2", "T3", (), 2, -1, "A")
        self.assertEqual(self.game.solve(4, state=state)[0], {"best_target": None, "best_value": 3})

    def test_all_orders_and_depths_against_brute_force(self):
        initial = self.game.initial_state()
        for depth in range(1, 6):
            expected = reference_game(self.game, "A", "B", initial.remaining, 0, 0, "A", depth)
            for order in ("original", "reverse", "gain"):
                normal, normal_stats = self.game.solve(depth, "minimax", order)
                pruned, pruned_stats = self.game.solve(depth, "alpha_beta", order)
                self.assertEqual(normal["best_value"], expected)
                self.assertEqual(pruned, normal)
                self.assertLessEqual(pruned_stats.visited, normal_stats.visited)
                child = self.game.move(initial, pruned["best_target"])
                self.assertEqual(reference_game(self.game, child.position_a, child.position_b, child.remaining, child.score_a, child.score_b, "B", depth - 1), expected)

    def test_random_treasure_values(self):
        generator = random.Random(4004)
        for case in range(30):
            game = TreasureGame.__new__(TreasureGame)
            game.__dict__.update(self.game.__dict__)
            game.values = {target: generator.randint(-3, 15) for target in self.game.values}
            initial = game.initial_state()
            for depth in (2, 4):
                expected = reference_game(game, "A", "B", initial.remaining, 0, 0, "A", depth)
                for order in ("original", "reverse", "gain"):
                    with self.subTest(case=case, depth=depth, order=order):
                        self.assertEqual(game.solve(depth, "alpha_beta", order)[0]["best_value"], expected)

    def test_directed_distances_and_special_cells(self):
        game = TreasureGame([["A", 8, "T1", "B"]], {"A": (0, 0), "B": (0, 3), "T1": (0, 2)}, {"T1": 5})
        self.assertEqual(game.distances["A", "T1"], 9)
        self.assertEqual(game.distances["T1", "B"], 1)
        self.assertEqual(game.solve(1)[0], {"best_target": "T1", "best_value": -4})


class OutputTests(unittest.TestCase):
    def run_script(self, script, *args):
        result = subprocess.run([sys.executable, str(BASE / script), *args], capture_output=True, text=True, encoding="utf-8", check=True)
        self.assertEqual(result.stderr, "")
        return result.stdout

    def test_exact_default_outputs(self):
        expected = {
            "question1.py": "[(0,0), (0,1), (0,2), (1,2), (1,3), (1,4), (2,4), (3,4)]\n",
            "question2.py": "7\n",
            "question3.py": "5\n",
            "question4.py": 'H = 2   =>  {"best_target": "T2", "best_value": -1}\nH = 4   =>  {"best_target": "T3", "best_value":  3}\n',
        }
        for script, output in expected.items():
            with self.subTest(script=script):
                self.assertEqual(self.run_script(script), output)

    def test_matrix_output(self):
        self.assertEqual(json.loads(self.run_script("question4.py", "--matrix")), TreasureGame().distance_matrix())

    def test_custom_json_input(self):
        with tempfile.TemporaryDirectory(prefix="test_input_", dir=BASE) as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({"start": [0, 0], "end": [0, 2], "grid": [["S", ".", "E"]]}), encoding="utf-8")
            self.assertEqual(self.run_script("question1.py", "--input", str(path)), "[(0,0), (0,1), (0,2)]\n")
            path.write_text(json.dumps({"start": [0, 0], "end": [0, 2], "grid": [["S", 3, "E"]]}), encoding="utf-8")
            self.assertEqual(self.run_script("question2.py", "--input", str(path)), "3\n")
            path.write_text(json.dumps({"tree_structure": {"children": [{"value": -1}, {"value": 8}]}}), encoding="utf-8")
            self.assertEqual(self.run_script("question3.py", "--input", str(path)), "8\n")
            path.write_text(json.dumps({"grid": [["A", 1, "T1", "B"]], "points": {"A": [0, 0], "B": [0, 3], "T1": [0, 2]}, "values": {"T1": 5}}), encoding="utf-8")
            self.assertEqual(self.run_script("question4.py", "--input", str(path), "--depth", "2"), 'H = 2   =>  {"best_target": "T1", "best_value":  3}\n')


if __name__ == "__main__":
    unittest.main()
