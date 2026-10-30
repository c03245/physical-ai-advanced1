"""実験: 言語指示からの計画を、LLM（Ollama / Claude）とルールベースで比べる（シミュレーション不要）。

いろいろな言い回し・条件付きの指示（暖色/寒色、両端と内側、「〜を使わずに」、上から数える、英語など）を与え、
返ってきた計画が指示どおりか・個数と矛盾しないか・何秒かかったか・自己修正が要ったかを記録する。
指示どおりかは、色の並びの正解を 1 つに決めず、条件を満たすかを調べる関数で判定する。

    uv run python scripts/eval_planner.py                       # ollama と rules
    uv run python scripts/eval_planner.py --backends ollama rules claude
    OLLAMA_MODEL=gemma4:12b uv run python scripts/eval_planner.py --tag 12b

結果は outputs/planner/ に JSON と表（results_<tag>.md）として保存される。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from so101_sorter import planner  # noqa: E402
from so101_sorter.env import COLORS, PYRAMID_ROWS  # noqa: E402
from so101_sorter.provenance import run_metadata  # noqa: E402

OUT = ROOT / "outputs" / "planner"
WARM = {"red", "orange", "pink", "brown"}
COOL = {"blue", "cyan", "green", "purple"}
# 机のブロックを指示に合わせて用意する段階と同じ「個数の制限なし」
UNLIMITED = Counter({c: sum(PYRAMID_ROWS) for c in COLORS})


def flat(rows):
    return [c for r in rows for c in r]


def rows_are(*colors):
    """各段が 1 色ずつ、下の段から colors の順になっている。"""
    return lambda rows: all(set(r) == {c} for r, c in zip(rows, colors, strict=True))


def ends_and_inner(outer, inner):
    def check(rows):
        for r in rows:
            if r[0] != outer or r[-1] != outer:
                return False
            if any(c != inner for c in r[1:-1]):
                return False
        return True

    return check


def alternating(a, b):
    """各段の中で a と b が交互に並ぶ（どちらから始めてもよい）。"""
    return lambda rows: all(set(r) <= {a, b} and all(x != y for x, y in zip(r, r[1:], strict=False)) for r in rows)


def distinct_uniform_rows(rows):
    return all(len(set(r)) == 1 for r in rows) and len({r[0] for r in rows}) == len(rows)


# (ID, 指示, 机のブロックの個数（None なら制限なし）, 判定関数, 種類)
# 開発用: プロンプトの改善を検討するときに結果を見てよいケース
DEV_CASES = [
    (
        "basic",
        "青いブロックを土台にして、緑、橙、赤の順にピラミッドを積んで",
        None,
        rows_are("blue", "green", "orange", "red"),
        "段の順",
    ),
    (
        "top_first",
        "一番上は赤、その下は橙、緑、青の順にして",
        None,
        rows_are("blue", "green", "orange", "red"),
        "段の順（上から）",
    ),
    (
        "english",
        "Build it with purple on the bottom, then cyan, lime, and brown on top.",
        None,
        rows_are("purple", "cyan", "lime", "brown"),
        "段の順（英語）",
    ),
    (
        "explicit_counts",
        "1段目は紫、2段目は黄緑、3段目と4段目はピンクにして",
        None,
        rows_are("purple", "lime", "pink", "pink"),
        "段の順",
    ),
    ("single", "全部ピンクで積んで", None, rows_are("pink", "pink", "pink", "pink"), "単色"),
    ("top_only", "頂上だけ赤で、それ以外は茶色にして", None, rows_are("brown", "brown", "brown", "red"), "部分指定"),
    ("rest_fill", "土台は青、残りはすべて緑にして", None, rows_are("blue", "green", "green", "green"), "部分指定"),
    (
        "warm_cool",
        "土台は暖色、上の段は寒色にして",
        None,
        lambda rows: set(rows[0]) <= WARM and set(flat(rows[1:])) <= COOL,
        "色の分類",
    ),
    ("cool_only", "赤を使わずに、寒色だけで積んで", None, lambda rows: set(flat(rows)) <= COOL, "否定・分類"),
    (
        "warm_gradient",
        "暖色だけを使って、段ごとに違う色にして",
        None,
        lambda rows: distinct_uniform_rows(rows) and set(flat(rows)) <= WARM,
        "色の分類・制約",
    ),
    ("distinct_rows", "段ごとに違う色にして、同じ色は2つの段で使わないで", None, distinct_uniform_rows, "制約"),
    ("outer_inner", "各段の両端を青、内側を赤にして", None, ends_and_inner("blue", "red"), "位置（両端/内側）"),
    ("alternate", "Make every row alternate between red and blue.", None, alternating("red", "blue"), "位置（交互）"),
    (
        "alt_base",
        "土台は青と水色を交互に並べて、その上は全部緑にして",
        None,
        lambda rows: alternating("blue", "cyan")(rows[:1]) and set(flat(rows[1:])) == {"green"},
        "位置（交互）",
    ),
    # 机のブロックが足りない場合: 青が 2 個しかないので、指示どおりには積めない（実行できる計画を返せるか）
    (
        "shortage",
        "青いブロックを土台にして、緑、橙、赤の順にピラミッドを積んで",
        {"blue": 2, "cyan": 4, "green": 3, "orange": 2, "red": 1, "purple": 3, "brown": 3, "pink": 3, "lime": 3},
        lambda rows: rows[0].count("blue") == 2 and set(flat(rows[1:])) >= {"green", "orange", "red"},
        "個数不足",
    ),
]
# テスト用: 改善の前に書いて固定し、改善の検討には使わない（開発用のケースへの過剰適合を確かめる）
TEST_CASES = [
    ("t_bottom_up", "下から順に、緑、緑、青、赤にして", None, rows_are("green", "green", "blue", "red"), "段の順"),
    (
        "t_top_en",
        "Put orange at the top and make everything else cyan.",
        None,
        rows_are("cyan", "cyan", "cyan", "orange"),
        "部分指定（英語）",
    ),
    (
        "t_apex",
        "ピラミッドの頂点を紫にして、土台は黄緑、中間の2段は水色",
        None,
        rows_are("lime", "cyan", "cyan", "purple"),
        "部分指定",
    ),
    (
        "t_cool_warm",
        "寒色の土台に、暖色の段を重ねて",
        None,
        lambda rows: set(rows[0]) <= COOL and set(flat(rows[1:])) <= WARM,
        "色の分類",
    ),
    (
        "t_no_brown",
        "茶色を使わずに暖色だけで積んで",
        None,
        lambda rows: set(flat(rows)) <= WARM - {"brown"},
        "否定・分類",
    ),
    ("t_outer_inner", "各段の両端はピンク、内側は緑にして", None, ends_and_inner("pink", "green"), "位置（両端/内側）"),
    (
        "t_top_down_en",
        "From top to bottom: red, pink, orange, brown.",
        None,
        rows_are("brown", "orange", "pink", "red"),
        "段の順（上から・英語）",
    ),
    (
        "t_mixed",
        "土台は赤と青を交互に、2段目は全部紫、3段目と頂上は黄緑",
        None,
        lambda rows: alternating("red", "blue")(rows[:1]) and rows_are("purple", "lime", "lime")(rows[1:]),
        "位置・段の順",
    ),
]
CASES = [(*c, "dev") for c in DEV_CASES] + [(*c, "test") for c in TEST_CASES]


def run(backend: str, instruction: str, counts: Counter) -> dict:
    """1 回計画し、結果と所要時間、LLM の呼び出し回数（2 回なら自己修正あり）を返す。"""
    calls = 0
    orig = planner._ollama_chat

    def counted(*a, **k):
        nonlocal calls
        calls += 1
        return orig(*a, **k)

    planner._ollama_chat = counted
    t0 = time.perf_counter()
    try:
        if backend == "rules":
            plan = planner.plan_with_rules(instruction, counts)
        else:
            plan = planner.BACKENDS[backend](instruction, counts)
        err = None
    except Exception as e:  # noqa: BLE001
        plan, err = None, f"{type(e).__name__}: {e}"
    finally:
        planner._ollama_chat = orig
    return {
        "rows": plan.rows if plan else None,
        "explanation": plan.explanation if plan else None,
        "error": err,
        "seconds": round(time.perf_counter() - t0, 2),
        "llm_calls": calls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["ollama", "rules"])
    ap.add_argument("--tag", default="default", help="出力ファイル名に付ける名前")
    ap.add_argument("--cases", nargs="+", default=None, help="一部のケースだけ実行する（ID）")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all", help="開発用 / テスト用のケース")
    ap.add_argument("--repeat", type=int, default=1, help="各ケースの試行回数（思考モードは温度 0 でも結果が揺れる）")
    args = ap.parse_args()
    metadata = run_metadata(vars(args))
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    for cid, instruction, avail, check, kind, split in CASES:
        if (args.cases and cid not in args.cases) or args.split not in ("all", split):
            continue
        counts = Counter(avail) if avail else UNLIMITED
        for backend, trial in [(b, t) for b in args.backends for t in range(1 if b == "rules" else args.repeat)]:
            r = run(backend, instruction, counts)
            feasible = r["rows"] is not None and planner.is_feasible(
                planner.PyramidPlan(rows=r["rows"], explanation=""), counts
            )
            try:
                correct = feasible and bool(check(r["rows"]))
            except (ValueError, IndexError):  # 段の数が違うなど
                correct = False
            records.append(
                {
                    "case": cid,
                    "split": split,
                    "trial": trial,
                    "kind": kind,
                    "instruction": instruction,
                    "backend": backend,
                    "feasible": feasible,
                    "correct": correct,
                    **r,
                }
            )
            mark = "OK " if correct else ("NG " if feasible else "ERR")
            print(
                f"[{mark}] {backend:6s} {cid:16s} {r['seconds']:5.1f}s calls={r['llm_calls']} {r['rows']}", flush=True
            )

    model = planner.OLLAMA_MODEL
    (OUT / f"results_{args.tag}.json").write_text(
        json.dumps({"metadata": metadata, "ollama_model": model, "records": records}, ensure_ascii=False, indent=1)
    )

    # 方式ごとの集計と、ケースごとの○×の表
    lines = [
        f"Ollama model: `{model}`",
        "",
        "| 方式 | 指示どおり（開発用） | 指示どおり（テスト用） | 実行可能 | 自己修正 | 平均時間 [s] |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def ratio(rs, split):
        """1 回の試行あたりの、指示どおりだったケースの数（複数回試したときは平均）。"""
        rs = [r for r in rs if r["split"] == split]
        if not rs:
            return "-"
        trials = len({r["trial"] for r in rs})
        return f"{sum(r['correct'] for r in rs) / trials:.3g}/{len(rs) // trials}"

    for b in args.backends:
        rs = [r for r in records if r["backend"] == b]
        if not rs:
            continue
        lines.append(
            f"| {b} | {ratio(rs, 'dev')} | {ratio(rs, 'test')} | {sum(r['feasible'] for r in rs)}/{len(rs)} | "
            f"{sum(r['llm_calls'] > 1 for r in rs)} | {sum(r['seconds'] for r in rs) / len(rs):.2f} |"
        )
    lines += [
        "",
        "| ID | 種類 | 指示 | " + " | ".join(args.backends) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in args.backends) + " |",
    ]
    for cid, instruction, _, _, kind, _ in CASES:
        if not any(r["case"] == cid for r in records):
            continue
        cells = []
        for b in args.backends:
            tr = [r for r in records if r["case"] == cid and r["backend"] == b]
            cells.append("".join("○" if r["correct"] else ("×" if r["feasible"] else "失") for r in tr))
        lines.append(f"| {cid} | {kind} | {instruction} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (OUT / f"results_{args.tag}.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
