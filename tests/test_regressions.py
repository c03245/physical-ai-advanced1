"""公開前レビューで見つかった誤判定の回帰テスト（GPU・API不要）。"""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from so101_sorter.env import COLORS, pyramid_slots
from so101_sorter.evaluation import evaluate_ground_truth
from so101_sorter.planner import plan_with_rules

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_robustness as robust  # noqa: E402


def result_fixture():
    rows = [["blue"] * 4, ["green"] * 3, ["orange"] * 2, ["red"]]
    return {
        "rows": rows,
        "planner": "rules",
        "results": [
            {"row": s.row, "index": s.index, "color": rows[s.row][s.index], "success": True} for s in pyramid_slots()
        ],
        "ground_truth": {
            "true_xy_error_mm_mean": 0,
            "true_xy_error_mm_max": 0,
            "all_blocks_at_their_level": True,
        },
    }


class PlannerTests(unittest.TestCase):
    def test_compound_colors_and_word_boundaries(self):
        counts = Counter({c: 10 for c in COLORS})
        for text, color in (("light blue", "cyan"), ("yellow-green", "lime"), ("黄緑", "lime")):
            with self.subTest(text=text):
                plan = plan_with_rules(f"{text} only", counts)
                self.assertEqual({c for row in plan.rows for c in row}, {color})
        # red を含む英単語は赤の指示にならない。
        plan = plan_with_rules("preferred", Counter({"cyan": 10, "red": 1}))
        self.assertEqual(plan.rows[0], ["cyan"] * 4)


class EvaluationTests(unittest.TestCase):
    def test_ground_truth_requires_every_slot(self):
        slots = pyramid_slots()
        points = np.array([s.xyz for s in slots])
        env = SimpleNamespace(block_positions=lambda: points, block_colors=["blue"] * len(slots))
        results = [
            {"row": s.row, "index": s.index, "color": "blue", "success": True, "measured": list(s.xyz)} for s in slots
        ]
        for records in ([], results[:-1]):
            self.assertFalse(evaluate_ground_truth(env, records)["all_blocks_at_their_level"])
        self.assertTrue(evaluate_ground_truth(env, results)["all_blocks_at_their_level"])
        points[-1, 2] -= 0.02
        self.assertFalse(evaluate_ground_truth(env, results)["all_blocks_at_their_level"])

    def test_completion_rejects_missing_duplicate_and_wrong_colors(self):
        fixture = result_fixture()
        self.assertTrue(robust.summarize_result(fixture, lambda rows: True)["complete"])
        for mode in ("empty", "missing", "duplicate", "wrong_color", "wrong_instruction", "collapsed"):
            data = copy.deepcopy(fixture)

            def check(rows):
                return mode != "wrong_instruction"

            if mode == "empty":
                data["results"] = []
            elif mode == "missing":
                data["results"].pop()
            elif mode == "duplicate":
                data["results"][-1] = data["results"][0]
            elif mode == "wrong_color":
                data["results"][0]["color"] = "red"
            elif mode == "wrong_instruction":
                pass
            else:
                data["ground_truth"]["all_blocks_at_their_level"] = False
            with self.subTest(mode=mode):
                self.assertFalse(robust.summarize_result(data, check)["complete"])

    def test_cache_invalidates_changes_force_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            context = {"source_sha256": "initial"}

            def child(*args, **kwargs):
                (out / "seed106_basic.json").write_text(json.dumps(result_fixture()))
                return SimpleNamespace(returncode=0)

            def run(force=False):
                args = ["eval_robustness.py", "--seeds", "106", "--cases", "basic"]
                with patch.object(sys, "argv", args + (["--force"] if force else [])):
                    robust.main()

            with (
                patch.object(robust, "OUT", out),
                patch.object(robust, "execution_context", side_effect=lambda: dict(context)),
                patch.object(robust.subprocess, "run", side_effect=child) as proc,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                # 旧JSONだけでは再利用しない。
                (out / "seed106_basic.json").write_text(json.dumps(result_fixture()))
                run()
                run()
                self.assertEqual(proc.call_count, 1)
                self.assertTrue(json.loads((out / "results.json").read_text())[0]["reused"])
                context["source_sha256"] = "changed"
                run()
                self.assertEqual(proc.call_count, 2)
                run(force=True)
                self.assertEqual(proc.call_count, 3)
                proc.side_effect = lambda *a, **k: SimpleNamespace(returncode=1)
                run(force=True)
                self.assertIn("error", json.loads((out / "results.json").read_text())[0])
                proc.side_effect = child
                run()
                self.assertEqual(proc.call_count, 5)


if __name__ == "__main__":
    unittest.main()
