"""実験: カメラ構成と較正誤差の違いで、ピラミッドの精度がどう変わるかを比べる。

設定ごとに main.py を実行し、出力された JSON（シミュレーションの正解値による誤差を含む）を表にまとめる。

    uv run python scripts/compare_cameras.py              # 既定の 3 設定 × seed 3
    uv run python scripts/compare_cameras.py --seeds 3 11

結果は outputs/compare/ に、動画・JSON・表（results.md）として保存される。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "compare"

# (名前, 説明, main.py への追加引数)
CONFIGS = [
    ("rgb_only", "俯瞰カメラのみ・較正誤差なし", ["--no-wrist"]),
    (
        "rgb_only_err",
        "俯瞰カメラのみ・較正誤差 4 mm / 1°",
        ["--no-wrist", "--calib-error-mm", "4", "--calib-error-deg", "1"],
    ),
    (
        "rgb_wrist_err",
        "俯瞰 + 手首カメラ・較正誤差 4 mm / 1°・取付誤差 1 mm / 0.5°",
        ["--calib-error-mm", "4", "--calib-error-deg", "1", "--wrist-error-mm", "1", "--wrist-error-deg", "0.5"],
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3])
    ap.add_argument("--configs", nargs="+", default=[c[0] for c in CONFIGS])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, desc, extra in CONFIGS:
        if name not in args.configs:
            continue
        for seed in args.seeds:
            out = OUT / f"{name}_seed{seed}.mp4"
            cmd = [
                sys.executable,
                str(ROOT / "main.py"),
                "--planner",
                "rules",
                "--seed",
                str(seed),
                "--out",
                str(out),
                *extra,
            ]
            print(f"=== {name} seed {seed}: {' '.join(extra)}", flush=True)
            completed = subprocess.run(cmd, cwd=ROOT, check=False)
            js = out.with_suffix(".json")
            if completed.returncode != 0 or not js.exists():
                status = f"実行失敗 (exit {completed.returncode})"
                rows.append((desc, seed, status, "-", "-", "-", "-"))
                continue
            d = json.loads(js.read_text())
            res = d["results"]
            n_ok = sum(r["success"] for r in res)
            retries = sum(r.get("attempts", 1) - 1 for r in res)
            gt = d["ground_truth"]
            rows.append(
                (
                    desc,
                    seed,
                    f"{n_ok}/{len(res)}",
                    retries,
                    gt["true_xy_error_mm_mean"],
                    gt["true_xy_error_mm_max"],
                    "○" if gt["all_blocks_at_their_level"] else "×",
                )
            )
            print(f"    -> success {n_ok}/{len(res)}, retries {retries}, ground truth {gt}", flush=True)

    lines = [
        "| 設定 | seed | 成功 | 置き直し回数 | 本当の誤差 平均 [mm] | 本当の誤差 最大 [mm] | 全段が正しい高さ |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    table = "\n".join(lines)
    (OUT / "results.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
