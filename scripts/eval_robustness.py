"""実験: ブロックの初期配置（seed）と言語指示を変えて main.py を連続実行し、システム全体の成功率を測る。

指示は scripts/eval_planner.py のテスト用（プロンプトの改善に使っていない）ケースから選び、
「指示どおりの色で（計画の判定）」「10 個すべてを置けて（カメラでの実測）」「崩れていない（シミュレーションの正解値）」
の 3 つがそろったものを「指示どおりに完成」と数える。

    uv run python scripts/eval_robustness.py                 # seed 101〜110（指示は seed ごとに切り替え）
    uv run python scripts/eval_robustness.py --seeds 101 102 --viewer

結果は outputs/robust/ に、実行ごとの JSON・ログと表（results.md）として保存される（既定では動画のナレーションは省く）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_planner import CASES  # noqa: E402
from so101_sorter.env import PYRAMID_ROWS  # noqa: E402
from so101_sorter.provenance import execution_context  # noqa: E402

OUT = ROOT / "outputs" / "robust"
# 既定のデモの指示と、テスト用の指示（段の順・英語・色の分類・両端/内側・複合）
CASE_IDS = ["basic", "t_cool_warm", "t_outer_inner", "t_top_down_en", "t_mixed", "t_apex"]


def summarize_result(d: dict, check) -> dict:
    res = d["results"]
    gt = d["ground_truth"]
    expected = {(row, index) for row, size in enumerate(PYRAMID_ROWS) for index in range(size)}
    keys = [(r["row"], r["index"]) for r in res]
    shape_ok = [len(row) for row in d["rows"]] == list(PYRAMID_ROWS)
    records_ok = len(keys) == len(expected) and set(keys) == expected
    colors_ok = records_ok and shape_ok and all(r["color"] == d["rows"][r["row"]][r["index"]] for r in res)
    try:
        plan_ok = shape_ok and bool(check(d["rows"]))
    except (ValueError, IndexError):
        plan_ok = False
    n_ok = sum(r["success"] is True for r in res)
    return {
        "planner": d["planner"],
        "plan_ok": plan_ok,
        "placed": f"{n_ok}/{len(expected)}",
        "retries": sum(max(0, r.get("attempts", 1) - 1) for r in res),
        "err_mean": gt["true_xy_error_mm_mean"],
        "err_max": gt["true_xy_error_mm_max"],
        "intact": gt["all_blocks_at_their_level"] is True,
        "video_s": d.get("video_seconds"),
        "complete": plan_ok and colors_ok and n_ok == len(expected) and gt["all_blocks_at_their_level"] is True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 111)))
    ap.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES], default=CASE_IDS)
    ap.add_argument("--viewer", action="store_true", help="Genesis のビューアを開いて実行する（等倍速）")
    ap.add_argument("--force", action="store_true", help="保存済みの結果も再実行する")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {c[0]: c for c in CASES}
    context = execution_context()

    rows = []
    for seed in args.seeds:
        cid = args.cases[(seed - 100) % len(args.cases)]  # 並列に分けて実行しても、seed ごとの指示が変わらないように
        _, instruction, _, check, _, _ = cases[cid]
        out = OUT / f"seed{seed}_{cid}.mp4"
        js = out.with_suffix(".json")
        manifest = out.with_suffix(".run.json")
        cache_key = {"context": context, "seed": seed, "instruction": instruction, "viewer": args.viewer}
        try:
            reusable = json.loads(manifest.read_text()) == {"status": "complete", **cache_key}
        except (OSError, ValueError):
            reusable = False
        completed = None
        if args.force or not reusable or not js.exists():
            # 失敗した実行が残した JSON を、次回に成功結果として再利用しない。
            manifest.write_text(json.dumps({"status": "running", **cache_key}, ensure_ascii=False, indent=2))
            cmd = [
                sys.executable,
                str(ROOT / "main.py"),
                "--seed",
                str(seed),
                "--instruction",
                instruction,
                "--no-narration",
                "--out",
                str(out),
            ] + (["--viewer"] if args.viewer else [])
            print(f"=== seed {seed} / {cid}: {instruction}", flush=True)
            with open(out.with_suffix(".log"), "w") as log:
                completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if not js.exists() or (completed is not None and completed.returncode != 0):
            exit_code = completed.returncode if completed is not None else "unknown"
            rows.append(
                {"seed": seed, "case": cid, "instruction": instruction, "error": f"実行失敗 (exit {exit_code})"}
            )
            continue
        try:
            d = json.loads(js.read_text())
            result = summarize_result(d, check)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            manifest.write_text(json.dumps({"status": "invalid", **cache_key}, ensure_ascii=False, indent=2))
            rows.append({"seed": seed, "case": cid, "instruction": instruction, "error": f"結果が不正: {exc}"})
            continue
        manifest.write_text(json.dumps({"status": "complete", **cache_key}, ensure_ascii=False, indent=2))
        rows.append(
            {
                "seed": seed,
                "case": cid,
                "instruction": instruction,
                "reused": completed is None,
                **result,
            }
        )
        print(f"    -> {rows[-1]}", flush=True)

    (OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    done = [r for r in rows if "error" not in r]
    lines = [
        f"指示どおりに完成: **{sum(r['complete'] for r in done)}/{len(rows)}**",
        "",
        "| seed | 指示 | 計画 | 計画が指示どおり | 置けた数 | 置き直し | 本当の誤差 平均 / 最大 [mm] | 崩れなし | 動画 [s] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['seed']} | {r['instruction']} | {r['error']} | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {r['seed']} | {r['instruction']} | {r['planner']} | {'○' if r['plan_ok'] else '×'} | {r['placed']} | "
            f"{r['retries']} | {r['err_mean']} / {r['err_max']} | {'○' if r['intact'] else '×'} | {r['video_s']} |"
        )
    table = "\n".join(lines)
    (OUT / "results.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
