"""配置結果の検証。シミュレータを起動せずにテストできる評価処理。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .env import BLOCK_SIZE, pyramid_slots

if TYPE_CHECKING:
    from .env import PyramidScene


def evaluate_ground_truth(env: PyramidScene, results: list[dict]) -> dict:
    """シミュレーションの正解値で、完成したピラミッドの「本当の」誤差を測る（カメラの較正誤差に左右されない）。

    各段のブロックは、下段は設計位置、上段は支えになる 2 個の実際の中点からのずれで評価する。
    """
    gt = env.block_positions()
    slot_of = {(s.row, s.index): s for s in pyramid_slots()}
    block_at: dict[tuple[int, int], int] = {}
    errors = []
    for r in results:
        key = (r["row"], r["index"])
        if not r["success"]:
            continue
        guess = np.array(r["measured"])
        cands = [i for i, c in enumerate(env.block_colors) if c == r["color"] and i not in block_at.values()]
        if not cands:
            continue
        i = min(cands, key=lambda i: np.linalg.norm(gt[i] - guess))
        block_at[key] = i
        if r["row"] == 0:
            ideal = np.array(slot_of[key].xyz)
        else:
            a, b = block_at.get((r["row"] - 1, r["index"])), block_at.get((r["row"] - 1, r["index"] + 1))
            if a is None or b is None:
                continue
            ideal = np.array([*(gt[a, :2] + gt[b, :2]) / 2, max(gt[a, 2], gt[b, 2]) + BLOCK_SIZE])
        err = (gt[i] - ideal) * 1000
        r["true_error_mm"] = np.round(err, 2).tolist()
        if r.get("gt_at_placement") is not None:  # 置いた後に、他のブロックの作業で押されて動いた量
            r["moved_after_placement_mm"] = np.round((gt[i] - np.array(r["gt_at_placement"])) * 1000, 2).tolist()
        errors.append(np.linalg.norm(err[:2]))
    intact = (
        len(block_at) == len(slot_of)
        and len(errors) == len(slot_of)
        and all(abs(gt[i, 2] - slot_of[k].xyz[2]) < 0.005 for k, i in block_at.items())
    )
    return {
        "true_xy_error_mm_mean": round(float(np.mean(errors)), 2) if errors else None,
        "true_xy_error_mm_max": round(float(np.max(errors)), 2) if errors else None,
        "all_blocks_at_their_level": bool(intact),
    }
