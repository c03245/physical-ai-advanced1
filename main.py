"""SO-101 × AI: Genesis 上で 10 個のブロックを 4・3・2・1 段のピラミッドに積む。

    uv run python main.py
    uv run python main.py --instruction "一番上は赤、その下は橙、緑、青の順にして" --viewer
    uv run python main.py --planner rules --seed 11 --calib-error-mm 4 --calib-error-deg 0.5

処理の流れ:
  俯瞰カメラ（普通のカメラ。--sensor rgbd なら深度カメラ） → OWLv2 による物体検出（AI）→ 位置・向きの推定
  → LLM が言語指示から段ごとの色を計画（ローカル LLM / Claude）→ 真上からの IK で把持・配置
  （掴む直前に手首カメラで位置合わせ）→ 1 個置くごとに手首カメラで実測し、ずれていれば掴み直して置き直す。
実機に移すことを想定し、カメラは普通の RGB カメラ 2 台（俯瞰 + 手首）で動く構成を既定にしている。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from so101_sorter.env import ensure_wsl_cuda

ensure_wsl_cuda()

import numpy as np
from PIL import Image

from so101_sorter.cinema import SHOT_JA, CameraOperator
from so101_sorter.controller import ArmController
from so101_sorter.evaluation import evaluate_ground_truth
from so101_sorter.env import (
    BLOCK_SIZE,
    DEFAULT_BLOCKS,
    HOLD_LOCAL,
    PITCH,
    PYRAMID_ROWS,
    PYRAMID_X0,
    PYRAMID_Y,
    PyramidScene,
    make_block_set,
    pyramid_slots,
)
from so101_sorter.narration import Narrator, duration
from so101_sorter.perception import (
    BlockDetector,
    CameraModel,
    View,
    measure_block_near,
    random_pose_error,
)
from so101_sorter.planner import PLANNER_LABELS, is_feasible, make_plan, plan_unlimited, required_colors
from so101_sorter.provenance import run_metadata
from so101_sorter.video import VideoWriter, annotate_detections

COLOR_JA = {
    "red": "赤",
    "orange": "橙",
    "green": "緑",
    "blue": "青",
    "purple": "紫",
    "lime": "黄緑",
    "pink": "ピンク",
    "cyan": "水色",
    "brown": "茶",
}
# ナレーション用の読み（音声合成が読み間違えやすい語は、かなで書く）
COLOR_READ = {
    "red": "あか",
    "orange": "だいだい",
    "green": "みどり",
    "blue": "あお",
    "purple": "むらさき",
    "lime": "きみどり",
    "pink": "ピンク",
    "cyan": "みずいろ",
    "brown": "ちゃいろ",
}
PLANNER_READ = {
    "ollama": "ローカル エルエルエム の ジェマ・フォー",
    "claude": "クロード",
    "rules": "ルールベースの計画",
}
ROW_JA = ["1段目（土台）", "2段目", "3段目", "4段目（頂上）"]
XY_TOL = 0.0025  # 置き位置の許容誤差 [m]。これを超えたら掴み直して置き直す
Z_TOL = 0.006
REFINE_MAX_SHIFT = 0.015  # 手首カメラの測り直しで、これ以上動く結果は誤認識とみなして使わない [m]
# デバッグ用: 設定すると、置いた後の確認でずれていると判定したときの手首カメラの画像をこのフォルダに保存する
DEBUG_DIR = os.environ.get("SO101_DEBUG_DIR")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruction", default="青いブロックを土台にして、緑、橙、赤の順にピラミッドを積んで")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument(
        "--aligned", action="store_true", help="ブロックをロボットの方向に揃えて置く（既定はランダムな向き）"
    )
    ap.add_argument(
        "--planner",
        choices=["ollama", "claude", "auto", "rules"],
        default="ollama",
        help="段ごとの色の計画: ローカル LLM（Ollama, 既定）/ Claude API / auto（ollama→claude）/ rules",
    )
    ap.add_argument(
        "--sensor",
        choices=["rgb", "rgbd"],
        default="rgb",
        help="俯瞰カメラ: rgb = 普通のカメラ（較正済みモデル + 既知の高さ）, rgbd = 深度カメラ",
    )
    ap.add_argument("--no-wrist", action="store_true", help="手首カメラによる位置合わせを使わない")
    ap.add_argument("--no-narration", action="store_true", help="解説ナレーション（音声）を入れない")
    ap.add_argument(
        "--fixed-blocks", action="store_true", help="指示に合わせてブロックを用意せず、決まった 9 色 × 3 個を置く"
    )
    ap.add_argument(
        "--no-camera-robot", action="store_true", help="カメラマン役の SO-101 を使わない（動画は固定カメラ 1 台で撮る）"
    )
    ap.add_argument("--calib-error-mm", type=float, default=0.0, help="俯瞰カメラの較正誤差（並進, mm）")
    ap.add_argument("--calib-error-deg", type=float, default=0.0, help="俯瞰カメラの較正誤差（回転, 度）")
    ap.add_argument("--wrist-error-mm", type=float, default=0.0, help="手首カメラの取付（ハンドアイ）誤差（並進, mm）")
    ap.add_argument("--wrist-error-deg", type=float, default=0.0, help="手首カメラの取付（ハンドアイ）誤差（回転, 度）")
    ap.add_argument(
        "--viewer", action="store_true", default=None, help="Genesis のビューアとカメラ映像を表示する（等倍速）"
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/demo.mp4"))
    ap.add_argument("--record-every", type=int, default=4, help="動画 1 フレームあたりのシミュレーションステップ数")
    ap.add_argument("--max-retries", type=int, default=2, help="1 個あたりの掴み直しの上限回数")
    ap.add_argument("--speed", type=float, default=1.0, help="アームの動作速度の倍率（動画は常に等倍速）")
    return ap.parse_args()


def in_pyramid_area(xy, margin: float = 0.0) -> bool:
    x0 = PYRAMID_X0 - BLOCK_SIZE - margin
    x1 = PYRAMID_X0 + PITCH * (PYRAMID_ROWS[0] - 1) + BLOCK_SIZE + margin
    return x0 < xy[0] < x1 and abs(xy[1] - PYRAMID_Y) < BLOCK_SIZE + margin


def main():
    args = parse_args()
    metadata = run_metadata(vars(args))
    # テーブルに置くブロック: 指示に必要な色の個数を必ず含める（足りなくて実現できない指示にならないように）
    pre_plan = None
    if args.fixed_blocks:
        block_set = list(DEFAULT_BLOCKS)
    else:
        t0 = time.time()
        pre_plan, pre_source = plan_unlimited(args.instruction, args.planner)
        pre_plan_seconds = time.time() - t0
        required = required_colors(pre_plan)
        block_set = make_block_set(required, seed=args.seed)
        print(f"Blocks for this instruction (needs {dict(required)}): {dict(Counter(block_set))}")
    env = PyramidScene(
        block_set,
        seed=args.seed,
        show_viewer=args.viewer,
        random_yaw=not args.aligned,
        camera_robot=not args.no_camera_robot,
    )
    operator = CameraOperator(env) if env.cambot is not None else None
    video = VideoWriter(args.out, fps=round(1 / (env.dt * args.record_every)))
    narrator = None if args.no_narration else Narrator()

    def say(text: str, essential: bool = False) -> None:
        """今の動画の時刻に、解説の読み上げを入れる。"""
        if narrator is not None:
            narrator.say(video.seconds, text, essential)

    def card(lines: list[str], seconds: float, background, speech: str) -> None:
        """タイトルカードを、読み上げが収まる長さで表示する。"""
        if narrator is not None:
            say(speech, essential=True)
            seconds = max(seconds, duration(speech) + 0.6)
        video.title_card(lines, seconds, background)

    use_wrist = not args.no_wrist
    tick = [0]

    monitor = {"proc": None, "path": args.out.with_name(args.out.stem + "_cameras.jpg"), "n": 0}

    def show_cameras():
        """ビューア表示中は、俯瞰カメラ（最新の AI 検出結果）と手首カメラ（ライブ）を並べたウィンドウを出す。

        Genesis のビューアと同じプロセスで OpenCV のウィンドウを使うと止まることがあるため、画像をファイルに
        書き出し、表示は別プロセス（tools/camera_monitor.py）に任せる。
        """
        if not env.realtime:
            return
        monitor["n"] += 1
        if monitor["n"] % 2:  # 約 12 fps で更新すれば十分
            return
        panes = [Image.fromarray(img) for img in video.insets.values() if img is not None]
        if not panes:
            return
        h = 360
        panes = [p.resize((p.width * h // p.height, h)) for p in panes]
        row = Image.new("RGB", (sum(p.width for p in panes), h))
        x = 0
        for p in panes:
            row.paste(p, (x, 0))
            x += p.width
        tmp = monitor["path"].with_suffix(".tmp.jpg")
        row.save(tmp, quality=85)
        os.replace(tmp, monitor["path"])  # 書きかけのファイルを読まれないよう、置き換えで更新する
        if monitor["proc"] is None:
            monitor["proc"] = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).parent / "tools" / "camera_monitor.py"),
                    str(monitor["path"]),
                    str(os.getpid()),
                ]
            )

    last_dets: list = []  # 最新の AI 検出結果（俯瞰カメラのライブ映像に重ねて表示する）

    director = {"slot": 0, "finale": False}  # カメラワークの状態（何個目を積んでいるか、完成後か）
    PICK_SHOTS = ["close_up", "low_angle", "orbit", "close_up", "low_angle"]
    PLACE_SHOTS = ["top_down", "close_up", "low_angle", "top_down", "orbit"]

    def direct_camera():
        """作業アームの手先を被写体にし、場面（掴む / 置く / 完成）に応じて演出を選んでカメラマンを動かす。"""
        tcp = env.gripper_pos(HOLD_LOCAL)
        if director["finale"]:
            operator.set_shot("hero")
        else:
            i = director["slot"]
            placing = in_pyramid_area(tcp, margin=0.03) and tcp[2] < 0.25
            operator.set_shot((PLACE_SHOTS if placing else PICK_SHOTS)[i % len(PICK_SHOTS)])
        operator.update(tcp, env.dt * args.record_every)
        video.label = f"カメラ: {SHOT_JA[operator.shot]}（カメラマン役の SO-101）"

    def on_step():
        tick[0] += 1
        if operator is not None:
            operator.servo()
        if tick[0] % args.record_every == 0:
            if use_wrist:
                video.insets["手首カメラ"] = env.render_wrist()
            if tick[0] % (args.record_every * 3) == 0 and "俯瞰カメラ（AI 検出）" in video.insets:
                # 俯瞰カメラもライブで更新する（約 8 fps）。検出枠は直近の認識結果
                video.insets["俯瞰カメラ（AI 検出）"] = annotate_detections(env.perception_cam.render()[0], last_dets)
            if operator is not None:
                direct_camera()
                video.insets["全体（固定カメラ）"] = env.video_cam.render()[0]
                video.add(operator.render())
            else:
                video.add(env.video_cam.render()[0])
            show_cameras()

    arm = ArmController(env, on_step)
    env.step(50)
    print("Loading OWLv2 ...")
    detector = BlockDetector()
    colors = list(dict.fromkeys(DEFAULT_BLOCKS))  # 認識で探す色（9 色すべて）

    # 較正済みカメラモデル（実機ではチェッカーボード等で較正した値）。誤差を指定すると、その分ずれたモデルを使う
    rng = np.random.default_rng(args.seed + 1000)
    overhead_error = random_pose_error(args.calib_error_mm / 1000, args.calib_error_deg, rng)
    wrist_error = random_pose_error(args.wrist_error_mm / 1000, args.wrist_error_deg, rng)
    overhead_model = CameraModel.from_genesis(env.perception_cam).with_error(overhead_error)

    def overhead_view() -> View:
        rgb = env.perception_cam.render()[0]
        if args.sensor == "rgbd":
            points, _ = env.perception_cam.render_pointcloud(world_frame=True)
            return View(rgb, points=points)
        return View(rgb, camera=overhead_model)

    def wrist_view() -> View:
        rgb = env.render_wrist()
        # 手首カメラの姿勢は、実機でも関節角からの順運動学とハンドアイ較正で分かる
        return View(rgb, camera=CameraModel.from_genesis(env.wrist_cam).with_error(wrist_error))

    def detect_sources():
        """ピラミッドの外にまだ置かれているブロック。"""
        view = overhead_view()
        dets = [d for d in detector.detect(view, colors) if not in_pyramid_area(d.position)]
        last_dets[:] = dets
        return dets, annotate_detections(view.rgb, dets)

    # 1. 認識
    t0 = time.time()
    sources, det_img = detect_sources()
    print(
        f"Detected {len(sources)} blocks ({time.time() - t0:.1f}s):",
        [(d.color, d.position.round(3).tolist(), round(math.degrees(d.yaw), 1)) for d in sources],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(det_img).save(args.out.with_name(args.out.stem + "_detections.png"))

    # 2. 計画
    # テーブルのブロックを決めたときの計画が、検出した個数で実行できるならそのまま使う（同じ指示を LLM に聞き直すと、
    # 違う色を選んで余分のない色が足りなくなることがある）。検出漏れなどで実行できないときだけ計画し直す
    detected = Counter(d.color for d in sources)
    if pre_plan is not None and is_feasible(pre_plan, detected):
        plan, source = pre_plan, pre_source
        print(f"Plan ({source}, {pre_plan_seconds:.1f}s, made before the scene): {plan.rows} - {plan.explanation}")
    else:
        t0 = time.time()
        plan, source = make_plan(args.instruction, [d.color for d in sources], backend=args.planner)
        print(f"Plan ({source}, {time.time() - t0:.1f}s): {plan.rows} - {plan.explanation}")

    first = operator.render() if operator is not None else env.video_cam.render()[0]
    sensor_ja = "普通のカメラ（RGB）" if args.sensor == "rgb" else "深度カメラ（RGB-D）"
    card(
        [
            "SO-101 × AI ピラミッド積み",
            "Genesis シミュレーション / 等倍速 / ブロック 10 個",
            f"俯瞰カメラ: {sensor_ja}" + (" + 手首カメラ" if use_wrist else ""),
        ],
        4.0,
        first,
        "エスオー・ワンオーワン と エーアイ で、10個のブロックを、ピラミッドに積みます。",
    )
    card(
        ["言語指示", f"「{args.instruction}」"],
        3.5,
        first,
        f"指示は、{args.instruction.replace('橙', 'だいだい')}、です。",
    )
    counts = {c: sum(d.color == c for d in sources) for c in colors}
    found = [f"{COLOR_JA[c]}×{n}" for c, n in counts.items() if n]
    n_colors = len(found)
    card(
        [f"AI 認識 (OWLv2): {n_colors} 色・{len(sources)} 個検出", "  ".join(found[:5]), "  ".join(found[5:])],
        3.5,
        det_img,
        f"オウル・ブイツー が、{n_colors}色、{len(sources)}個のブロックを見つけました。この中から選んで積みます。",
    )
    card(
        [f"タスク計画 ({PLANNER_LABELS[source]})"]
        + [f"{ROW_JA[r]}: {'・'.join(COLOR_JA[c] for c in row)}" for r, row in enumerate(plan.rows)]
        + [plan.explanation],
        4.0,
        first,
        f"{PLANNER_READ[source]}が、10個それぞれの色を決めました。土台は、"
        f"{'、'.join(COLOR_READ[c] for c in plan.rows[0])}。頂上は{COLOR_READ[plan.rows[-1][0]]}です。",
    )
    video.insets["俯瞰カメラ（AI 検出）"] = det_img

    def slot_narration(n_slot: int, color: str) -> str:
        """各ブロックを置き始めるときの解説（技術の説明を少しずつ挟む）。"""
        c = COLOR_READ[color]
        errs = [np.linalg.norm(r["error_mm"][:2]) for r in results if r.get("error_mm")]
        lines = {
            0: f"まず土台です。{c}のブロックを、手首カメラで位置を確かめてから掴みます。",
            1: "置いたあとは、手首カメラで位置を測り、ずれていれば置き直します。",
            2: (
                "撮影しているのは、カメラマン役の、もう1台の エスオー・ワンオーワン です。"
                if operator is not None
                else "土台の3個目です。"
            ),
            3: "土台の最後の1個です。",
            4: f"2段目に入ります。{c}のブロックを、下の2個のちょうど中間に置きます。",
            5: "指には平らなパッドを付けて、ブロックが滑らないようにしています。",
            6: "2段目の最後の1個です。",
            7: f"3段目に入ります。{c}のブロックです。",
            8: (
                f"ここまでの置き位置の誤差は、平均およそ{np.mean(errs):.1f}ミリメートルです。"
                if errs
                else f"{c}のブロックを置きます。"
            ),
            9: f"最後に、頂上の{c}を置きます。",
        }
        return lines.get(n_slot, f"{c}のブロックを置きます。")

    # 3. 積む（1 個置くごとに確認）
    placed: dict[tuple[int, int], np.ndarray] = {}  # (段, 番号) → 実測したブロック中心
    results = []
    refine_log = {"grasp": 0, "grasp_tried": 0}
    slots = pyramid_slots()
    for n_slot, slot in enumerate(slots):
        director["slot"] = n_slot
        color = plan.rows[slot.row][slot.index]
        if slot.row == 0:
            target = np.array(slot.xyz)
        else:  # 支えになる 2 個の実測位置の中間
            a, b = placed.get((slot.row - 1, slot.index)), placed.get((slot.row - 1, slot.index + 1))
            if a is None or b is None:
                print(f"row{slot.row}#{slot.index}: missing support, skipping")
                results.append({"row": slot.row, "index": slot.index, "color": color, "success": False})
                continue
            target = np.array([*(a[:2] + b[:2]) / 2, max(a[2], b[2]) + BLOCK_SIZE])

        def refine_grasp(xyz, yaw, color=color):
            """掴む直前: 手首カメラで掴むブロックを測り直す。"""
            refine_log["grasp_tried"] += 1
            pose = measure_block_near(wrist_view(), color, xyz)
            if pose is None or np.linalg.norm(pose[0][:2] - xyz[:2]) > REFINE_MAX_SHIFT:
                return None
            refine_log["grasp"] += 1
            return pose

        measured, attempts, block = None, 0, None
        for attempt in range(args.max_retries + 1):
            attempts = attempt + 1
            if block is None:  # テーブルから新しいブロックを取る
                cands = [d for d in sources if d.color == color]
                if not cands:
                    print(f"row{slot.row}#{slot.index} {color}: no {color} block left on the table")
                    break
                d = min(cands, key=lambda d: np.linalg.norm(d.position[:2] - target[:2]))
                block = (d.position, d.yaw)
                taken = d  # 認識結果を使い回すので、取ったブロックは候補から外す
            else:
                taken = None
            what = "掴み直して置き直し" if attempt else "配置"
            if attempt:
                say("ずれていたので、掴み直して置き直します。")
            else:
                say(slot_narration(n_slot, color))
            video.caption = (
                f"[{n_slot + 1}/{len(slots)}] {ROW_JA[slot.row]} {slot.index + 1}個目: "
                f"{COLOR_JA[color]}のブロックを{what}"
            )
            video.subcaption = (
                f"目標 x={target[0]:.3f} y={target[1]:.3f} z={target[2]:.3f} m / "
                f"AI 推定位置 x={block[0][0]:.3f} y={block[0][1]:.3f} 向き {math.degrees(block[1]):+.1f}°"
            )
            wrist_check = []

            def check_with_wrist(color=color, target=target, out=wrist_check):
                """離した直後、真上から手首カメラで置いた結果を測る（持っているブロックに隠されない）。"""
                view = wrist_view()
                out.append(measure_block_near(view, color, target))
                out.append(view.rgb)

            arm.pick_and_place(
                block[0],
                target,
                block_yaw=block[1],
                target_yaw=0.0,
                speed=args.speed,
                refine_grasp=refine_grasp if use_wrist else None,
                after_release=check_with_wrist if use_wrist else None,
            )
            if taken is not None:
                sources = [d for d in sources if d is not taken]
                last_dets[:] = sources

            # 確認には、近くから見られる手首カメラを優先する（俯瞰カメラの較正誤差に左右されない）。
            # 手首カメラで確認できれば、毎回ホームに戻らずに次のブロックへ向かう。テーブル上のブロックは
            # 最初の認識結果を使い回し、掴む直前に手首カメラで測り直す
            pose = wrist_check[0] if wrist_check else None
            if pose is None:
                # 手首カメラで確認できない・目標の位置にない: ホームに戻り、俯瞰カメラで確認して認識し直す
                arm.home(1.2 / args.speed)
                arm.hold(0.2)
                pose = measure_block_near(overhead_view(), color, target)
                sources, det_img = detect_sources()
                video.insets["俯瞰カメラ（AI 検出）"] = det_img
            if pose is None:  # 目標の位置にない: 落ちたか掴めていない → テーブルから取り直す
                print(f"row{slot.row}#{slot.index} {color}: not found at target (attempt {attempts})")
                block = None
                continue
            err = pose[0] - target
            print(
                f"row{slot.row}#{slot.index} {color}: err_mm={np.round(err * 1000, 1)} "
                f"yaw={math.degrees(pose[1]):+.1f} (attempt {attempts})"
            )
            if DEBUG_DIR and wrist_check and np.linalg.norm(err[:2]) > XY_TOL:
                Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
                name = f"row{slot.row}_{slot.index}_{color}_try{attempts}.png"
                Image.fromarray(wrist_check[1]).save(Path(DEBUG_DIR) / name)
            if np.linalg.norm(err[:2]) <= XY_TOL and abs(err[2]) <= Z_TOL:
                measured = pose[0]
                break
            block = pose  # 近いがずれている: その場所から掴み直して置き直す
        ok = measured is not None
        gt_now = None
        if ok:
            placed[(slot.row, slot.index)] = measured
            # 評価専用: 置いた直後の正解位置を記録しておき、後の作業で押されて動いた量を最後に調べる
            gt = env.block_positions()
            same = [i for i, c in enumerate(env.block_colors) if c == color]
            gt_now = np.round(gt[min(same, key=lambda i: np.linalg.norm(gt[i] - measured))], 4).tolist()
        results.append(
            {
                "row": slot.row,
                "index": slot.index,
                "color": color,
                "success": ok,
                "attempts": attempts,
                "measured": None if not ok else np.round(measured, 4).tolist(),
                "gt_at_placement": gt_now,
                "error_mm": None if not ok else np.round((measured - target) * 1000, 2).tolist(),
            }
        )

    n_ok = sum(r["success"] for r in results)
    video.caption = ""
    video.insets = {}
    director["finale"] = True
    say("配置作業を終えました。", essential=True)
    arm.home(1.5 / args.speed)
    arm.hold(1.0)
    if operator is not None:
        arm.hold(6.0)  # 完成したピラミッドの周りをカメラが回り込む
    last = operator.render() if operator is not None else env.video_cam.render()[0]
    gt_eval = evaluate_ground_truth(env, results)
    complete = n_ok == len(slots) and gt_eval["all_blocks_at_their_level"]
    video.label = ""
    card(
        [
            "配置作業終了",
            f"計画に沿った配置: {n_ok} / {len(slots)} 個",
            "最終確認: 全段の高さは正常" if complete else "最終確認: 未配置または高さの異常あり",
        ],
        4.0,
        last,
        f"計画に沿って{n_ok}個を配置しました。"
        + ("全段の高さを確認しました。" if complete else "最終確認は未達です。"),
    )
    video.close()

    summary = {
        "metadata": metadata,
        "placement_complete": complete,
        "instruction": args.instruction,
        "planner": source,
        "rows": plan.rows,
        "explanation": plan.explanation,
        "sensor": args.sensor,
        "wrist_camera": use_wrist,
        "calib_error": {
            "overhead_mm": args.calib_error_mm,
            "overhead_deg": args.calib_error_deg,
            "wrist_mm": args.wrist_error_mm,
            "wrist_deg": args.wrist_error_deg,
        },
        "wrist_refinements": refine_log,
        "ground_truth": gt_eval,
        "camera_robot_contacts": operator.contacts if operator is not None else None,
        "results": results,
        "video_seconds": round(video.seconds, 1),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if narrator is not None:
        narrator.mux(args.out, video.seconds)
    print(f"Wrist refinements: {refine_log}")
    print(f"Ground truth: {gt_eval}")
    print(f"Success {n_ok}/{len(results)} - video: {args.out} ({video.seconds:.1f} s)")


if __name__ == "__main__":
    main()
