"""デバッグ用: AI を使わず、ブロックの正解位置を使ってピラミッドを積む（ビューア表示も可）。

uv run python scripts/debug_pick.py --viewer --video outputs/debug_pyramid.mp4
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from so101_sorter.env import ensure_wsl_cuda

ensure_wsl_cuda()

from collections import Counter

import numpy as np

from so101_sorter.controller import ArmController
from so101_sorter.env import BLOCK_SIZE, DEFAULT_BLOCKS, PyramidScene, pyramid_slots
from so101_sorter.planner import plan_with_rules

ap = argparse.ArgumentParser()
ap.add_argument("--viewer", action="store_true", default=None, help="Genesis のビューアを開く")
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--video", type=Path, default=None, help="動画用カメラの映像をこの mp4 に保存する")
args = ap.parse_args()

env = PyramidScene(DEFAULT_BLOCKS, seed=args.seed, show_viewer=args.viewer)
frames = []


def record():
    if args.video:
        record.n = getattr(record, "n", 0) + 1
        if record.n % 4 == 0:
            frames.append(env.video_cam.render()[0])


arm = ArmController(env, record)
env.step(50)
plan = plan_with_rules("", Counter(DEFAULT_BLOCKS))  # 数の多い色から順に割り当てる計画
placed: dict[tuple[int, int], int] = {}  # (段, 番号) → ブロック番号
used: set[int] = set()
for slot in pyramid_slots():
    gt = env.block_positions()
    bid = min(
        (i for i, c in enumerate(env.block_colors) if c == plan.rows[slot.row][slot.index] and i not in used),
        key=lambda i: np.linalg.norm(gt[i, :2] - np.array(slot.xyz[:2])),
    )
    if slot.row == 0:
        target = np.array(slot.xyz)
    else:  # 支えになる 2 個の実際の位置の中間に置く
        a, b = gt[placed[(slot.row - 1, slot.index)]], gt[placed[(slot.row - 1, slot.index + 1)]]
        target = np.array([*(a[:2] + b[:2]) / 2, max(a[2], b[2]) + BLOCK_SIZE])
    pitch = arm.pick_and_place(gt[bid], target, block_yaw=env.block_yaws()[bid], target_yaw=0.0)
    used.add(bid)
    placed[(slot.row, slot.index)] = bid
    arm.home(1.0)
    env.step(30)
    p = env.block_positions()[bid]
    err = (p - target) * 1000
    ok = abs(err[2]) < 5 and np.linalg.norm(err[:2]) < 8
    print(
        f"row{slot.row}#{slot.index} {env.block_colors[bid]}: pitch {math.degrees(pitch):.0f} "
        f"err_mm={np.round(err, 1)} yaw={math.degrees(env.block_yaws()[bid]):.1f}deg {'OK' if ok else 'FAIL'}",
        flush=True,
    )

if args.video:
    import imageio.v2 as imageio

    imageio.mimsave(args.video, frames, fps=25)
