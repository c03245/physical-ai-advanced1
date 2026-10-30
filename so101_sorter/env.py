"""Genesis のシーン: 台に載せた SO-101、色付きブロック（既定 27 個）、4・3・2・1 段のピラミッドを積む台座、カメラ。"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# 手首カメラ付きの純正 SO-101 MJCF に平らな指パッドを追加したもの（tools/make_pad_mjcf.py で生成）
SO101_MJCF = ROOT / "assets" / "so101" / "so101_pads.xml"

BLOCK_SIZE = 0.025

# "gripper" リンク座標系での点（ローカル x = 指の閉じる方向、可動指の側が正）。
# 可動指はブロックを固定指のパッド（内面 x = -7.4 mm）に平らに押し付けるので、保持したブロックの中心は決まった点
# になる: その面からブロック半個分、"gripperframe" サイトの高さ。
HOLD_LOCAL = np.array([-0.0074 + BLOCK_SIZE / 2, 0.0, -0.0981274])
# 掴むときはその点から可動指の側へ 3 mm ずらして下ろし、固定指がブロックの角に乗らないようにする。
# 指を閉じると、ブロックはその 3 mm だけ滑って HOLD_LOCAL に収まる。
GRASP_LOCAL = HOLD_LOCAL + np.array([0.003, 0.0, 0.0])

# 手首カメラ: SO-ARM100 の so101_new_calib_camera.xml にある wrist_camera ボディの姿勢（gripper 座標系）。
# MJCF にはレンズの向きの定義がないので、モジュールの前面方向（ボディの +z）を光軸とし、レンズ面に視点を置く。
WRIST_CAM_RES = (640, 480)
WRIST_CAM_FOV = 60.0  # 垂直画角（度）
WRIST_CAM_POS = np.array([0.0025, 0.0734, 0.0066])  # wrist_camera ボディ原点
WRIST_CAM_AXES = np.array([[1.0, 0.0, 0.0], [0.0, -0.906, -0.424], [0.0, 0.424, -0.906]])  # 列 = ボディの x, y, z 軸
WRIST_CAM_LENS = 0.026  # ボディ原点からレンズ面までの距離


def wrist_cam_offset() -> np.ndarray:
    """gripper リンク座標系 → 手首カメラ（Genesis / OpenGL 流: -z が視線、y が上）の同次変換。"""
    forward = WRIST_CAM_AXES[:, 2]
    up = WRIST_CAM_AXES[:, 1]  # 画像の上 = 指先の側（ロボットの根元側から見た向きになる）
    z = -forward
    x = np.cross(up, z)
    T = np.eye(4)
    T[:3, :3] = np.stack([x / np.linalg.norm(x), up, z], axis=1)
    T[:3, 3] = WRIST_CAM_POS + WRIST_CAM_LENS * forward
    return T


GRIP_TORQUE = float(os.environ.get("SO101_GRIP_TORQUE", "0.8"))  # グリッパー関節のトルク上限 [N·m]

# ブロックの 9 色。照明の当たる上面の色が互いに十分離れるように選んだ（黄色は SO-101 自体の色、白は机の色なので
# 使わない。黒は、手首カメラに写る指パッドや影と紛らわしく、置いた後の確認がずれるので使わない）
COLORS = {
    "red": (0.85, 0.10, 0.10),
    "orange": (0.98, 0.52, 0.05),
    "green": (0.10, 0.62, 0.18),
    "blue": (0.12, 0.25, 0.85),
    "purple": (0.50, 0.18, 0.72),
    "lime": (0.55, 0.85, 0.12),
    "pink": (0.98, 0.48, 0.75),
    "cyan": (0.35, 0.82, 0.95),
    "brown": (0.42, 0.24, 0.10),
}
BLOCKS_PER_COLOR = 3
# 固定のブロック（--fixed-blocks）: 9 色 × 3 個 = 27 個
DEFAULT_BLOCKS = [c for c in COLORS for _ in range(BLOCKS_PER_COLOR)]
# テーブルに置くブロックの数（前方の半円の帯に、間隔 4.2 cm で確実に置けるのは 34 個ほど。指が入る余裕を残す）
TOTAL_BLOCKS = 27


def make_block_set(required: dict[str, int], total: int = TOTAL_BLOCKS, seed: int = 0) -> list[str]:
    """指示に必要な色をちょうどその個数だけ含む、合計 total 個のブロックの色のリスト。

    積む色は余らせない（余ると、どれを使っても同じになり選ぶ意味がない）。残りは使わない色だけで、少ない色から
    順に均等に埋める（紛らわしい色が混ざった中から選ぶ構図にするため）。使わない色がなければ必要な分だけにする。
    """
    rng = np.random.default_rng(seed)
    blocks = [c for c, n in required.items() for _ in range(n)]
    others = [str(c) for c in rng.permutation([c for c in COLORS if not required.get(c)])]
    while others and len(blocks) < total:
        blocks.append(min(others, key=lambda c: (blocks.count(c), others.index(c))))
    return blocks


TABLE_COLOR = (0.95, 0.95, 0.95)  # 白い机
AMBIENT_LIGHT = 0.25  # 環境光（既定の 0.1 では影が濃すぎる）

# アームは高さ 10 cm の台に載せる（机の端に、作業面より高くクランプで固定するのと同じ）。テーブルがアームの根元より
# 低いと、グリッパーを真下に向けたままピラミッドの全段に届く。床に置くと、5 自由度の SO-101 は高さ約 8 cm より上で
# グリッパーを垂直に保てない。
PEDESTAL_H = float(os.environ.get("SO101_PEDESTAL_H", "0.10"))
PEDESTAL_XY = (0.13, 0.11)  # アームを載せる台の大きさ（中心はロボットの根元）

# ピラミッド: 列はロボットの x 軸方向（放射方向）に並べる。指はそれと直交する向きに水平に閉じるので、
# 同じ列の隣のブロックに入り込まない。
PYRAMID_ROWS = (4, 3, 2, 1)
PYRAMID_X0 = 0.165  # 1 段目の最初のブロックの中心
PYRAMID_Y = 0.0
GAP = 0.006  # 1 段目の隣り合うブロックの隙間
PITCH = BLOCK_SIZE + GAP
PAD_H = 0.002

# カメラマン役の SO-101（任意）: 作業エリアの斜め向かいに置き、手先にジンバル付きカメラを持たせて撮影する
CAMBOT_XY = (0.55, -0.22)  # 作業エリアから離し、腕を伸ばした自然な姿勢で撮れるようにする
CAMBOT_YAW = math.atan2(0.0 - CAMBOT_XY[1], 0.20 - CAMBOT_XY[0])  # 作業エリアの中心 (0.20, 0) を向く
CAMBOT_REST_Q = np.array([0.0, -0.7, 0.79, 1.6, 0.0, 0.3])  # コンパクトな待機姿勢（作業エリアにはみ出さない）

# ブロックを最初に置く範囲（ピラミッドの両側）
# ロボットの根元からの距離（前方の半円の帯）。近すぎると、下ろすときに腕が自分の台座の角に当たる
SPAWN_RADIUS = (0.14, 0.268)
SPAWN_KEEP_OUT = ((0.13, 0.29), 0.05)  # ピラミッドの周り（x の範囲, |y| の上限）には置かない
MIN_SEPARATION = 0.042


@dataclass(frozen=True)
class Slot:
    row: int  # 0 = 1 段目
    index: int  # 段の中での位置（ロボット側から数える）
    xyz: tuple[float, float, float]  # 設計上のブロック中心


def pyramid_slots() -> list[Slot]:
    """設計上のブロック中心を、積む順（下の段から、ロボット側から）に返す。"""
    slots = []
    for row, n in enumerate(PYRAMID_ROWS):
        for i in range(n):
            x = PYRAMID_X0 + PITCH * (i + row / 2)
            slots.append(Slot(row, i, (x, PYRAMID_Y, PAD_H + BLOCK_SIZE * (row + 0.5))))
    return slots


def ensure_wsl_cuda() -> None:
    """WSL2 では、Genesis の CUDA バックエンドに /usr/lib/wsl/lib を LD_LIBRARY_PATH へ入れる必要がある。

    動的リンカは起動時にしか LD_LIBRARY_PATH を読まないので、設定してからプロセスを 1 回だけ起動し直す。
    """
    wsl_lib = "/usr/lib/wsl/lib"
    if not os.path.isdir(wsl_lib):
        return
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if wsl_lib in paths:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join([wsl_lib, *filter(None, paths)])
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])


class PyramidScene:
    def __init__(
        self,
        block_colors: list[str],
        seed: int = 0,
        show_viewer: bool | None = None,
        random_yaw: bool = True,
        camera_robot: bool = False,
        video_res: tuple[int, int] = (1280, 720),
    ):
        import genesis as gs

        if show_viewer is None:  # SO101_VIEWER=1 なら毎回ビューアを開く
            show_viewer = os.environ.get("SO101_VIEWER", "0") == "1"
        gs.init(backend=gs.gpu, logging_level="warning", seed=seed)
        self.rng = np.random.default_rng(seed)
        self.random_yaw = random_yaw

        self.dt = 0.01
        # ビューア表示中は、シミュレーションを実時間（等倍速）に合わせて進める
        self.realtime = show_viewer
        self._wall_t0: float | None = None
        self._sim_steps = 0
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            rigid_options=gs.options.RigidOptions(
                # 自己衝突を有効にする（隣り合うリンクどうしは除く）。無効だと、カメラマン役のロボットが自分の腕に
                # めり込んでも止まらない
                enable_self_collision=True,
                enable_adjacent_collision=False,
                # 接触を硬くする（既定 0.01 s）。柔らかいと、握ったパッドがブロックにめり込み、離したときに
                # ブロックが横へ滑って下の段を引きずる。サブステップ (0.01/4 s) の 2 倍が下限
                constraint_timeconst=float(os.environ.get("SO101_CONTACT_TIMECONST", "0.005")),
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(AMBIENT_LIGHT,) * 3),
            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.55, -0.45, 0.42),
                camera_lookat=(0.17, 0.0, 0.05),
                camera_fov=45,
            ),
        )
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=TABLE_COLOR))
        # 凸分解して、2 本の指の間の隙間を残す（1 つの凸包にすると隙間が埋まる）
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(SO101_MJCF),
                pos=(0, 0, PEDESTAL_H),
                decompose_robot_error_threshold=float(os.environ.get("SO101_DECOMP", "0.05")),
            )
        )
        if PEDESTAL_H > 0:
            self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, PEDESTAL_H / 2), size=(*PEDESTAL_XY, PEDESTAL_H), fixed=True),
                surface=gs.surfaces.Default(color=(0.25, 0.25, 0.28)),
            )
        self.cambot = None
        if camera_robot:
            self.cambot = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=str(SO101_MJCF),
                    pos=(*CAMBOT_XY, PEDESTAL_H),
                    euler=(0, 0, math.degrees(CAMBOT_YAW)),
                    decompose_robot_error_threshold=float(os.environ.get("SO101_DECOMP", "0.05")),
                )
            )
            self.cambot_pedestal = self.scene.add_entity(
                gs.morphs.Box(
                    pos=(*CAMBOT_XY, PEDESTAL_H / 2),
                    size=(*PEDESTAL_XY, PEDESTAL_H),
                    fixed=True,
                    euler=(0, 0, math.degrees(CAMBOT_YAW)),
                ),
                surface=gs.surfaces.Default(color=(0.25, 0.25, 0.28)),
            )
        pad_len = PITCH * PYRAMID_ROWS[0] + 0.012
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(PYRAMID_X0 + PITCH * (PYRAMID_ROWS[0] - 1) / 2, PYRAMID_Y, PAD_H / 2),
                size=(pad_len, BLOCK_SIZE + 0.012, PAD_H),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.35, 0.35, 0.38)),
        )

        self.block_colors = list(block_colors)
        self.blocks = []
        for color, (pos, yaw) in zip(block_colors, self._sample_block_poses(len(block_colors)), strict=True):
            self.blocks.append(
                self.scene.add_entity(
                    gs.morphs.Box(size=(BLOCK_SIZE,) * 3, pos=(*pos, BLOCK_SIZE / 2), euler=(0, 0, math.degrees(yaw))),
                    material=gs.materials.Rigid(friction=1.5),
                    surface=gs.surfaces.Default(color=COLORS[color]),
                )
            )

        # AI の認識に使う、ほぼ真上からの俯瞰カメラ。すべての上面が見え、畳んだアームが隠すのは根元の周りだけで、
        # ピラミッドの列のブロックどうしも隠し合わない
        self.perception_cam = self.scene.add_camera(
            res=(960, 720),
            pos=(0.21, -0.10, 0.65),
            lookat=(0.19, 0.0, 0.0),
            up=(1.0, 0.0, 0.0),
            fov=48,
        )
        # デモ動画用の斜め上からのカメラ
        # 全体を撮る固定カメラ。カメラマン役のロボットがいるときは、それが手前に来ない反対側（+y 側）から撮る
        self.video_cam = self.scene.add_camera(
            res=video_res,
            fov=45,
            pos=(0.42, 0.50, 0.45) if camera_robot else (0.55, -0.45, 0.42),
            lookat=(0.28, -0.06, 0.06) if camera_robot else (0.17, 0.0, 0.05),
        )
        # 手首カメラ（公式の手首カメラ取付具に付く 32x32 mm の UVC モジュール相当）。build 後に gripper リンクへ取り付ける
        # near は既定の 5 cm では大きすぎる: 指や折りたたんだ腕がそれより近くにあり、切り取られて透明に見える
        self.wrist_cam = self.scene.add_camera(res=WRIST_CAM_RES, fov=WRIST_CAM_FOV, near=0.005)
        # カメラマン役のロボットが持つカメラ（姿勢は cinema.py が毎フレーム決める）
        self.cine_cam = self.scene.add_camera(res=video_res, fov=55, near=0.01) if camera_robot else None

        self.scene.build()
        self.gripper_link = self.robot.get_link("gripper")
        self.wrist_cam.attach(self.gripper_link, wrist_cam_offset())
        # 待機姿勢。後ろへ 105° 回して、俯瞰カメラからブロックを隠さないようにする。自己衝突を有効にした探索で、
        # 腕・手首カメラが自分の体や台にめり込まない姿勢のうち、最もコンパクトなものを選んだ
        self.home_q = np.array([1.83, -0.7, 0.79, 1.6, 1.57, 0.31])
        # 握る力を制限する: サーボの最大トルク（パッドで約 55 N）では、シミュレーションの接触がわずかに柔らかいため、
        # パッドが 15 g のブロックにめり込んでしまう。約 12 N で十分に保持でき、減衰の大きい指も素早く閉じる
        lo, hi = (t.cpu().numpy() for t in self.robot.get_dofs_force_range())
        lo[5], hi[5] = -GRIP_TORQUE, GRIP_TORQUE
        self.robot.set_dofs_force_range(lo, hi)
        self.robot.set_dofs_position(self.home_q)
        self.robot.control_dofs_position(self.home_q)
        if self.cambot is not None:
            self.cambot.set_dofs_position(CAMBOT_REST_Q)
            self.cambot.control_dofs_position(CAMBOT_REST_Q)

    def _sample_block_poses(self, n: int):
        """ロボットの前方の半円の帯（ピラミッドの周りを除く）に、重ならないようにランダムにブロックを置く位置。

        向きは既定でランダム（認識で推定し、グリッパーを合わせる）。random_yaw=False ならロボットの方を向ける。
        """
        (kx0, kx1), ky = SPAWN_KEEP_OUT
        # 収まらなければ、間隔を少しずつ詰めて置き直す
        for separation in (MIN_SEPARATION, MIN_SEPARATION - 0.002, MIN_SEPARATION - 0.004):
            poses: list[np.ndarray] = []
            for _ in range(40000):
                if len(poses) == n:
                    break
                p = np.array(
                    [self.rng.uniform(0.0, SPAWN_RADIUS[1]), self.rng.uniform(-SPAWN_RADIUS[1], SPAWN_RADIUS[1])]
                )
                if not SPAWN_RADIUS[0] < np.linalg.norm(p) < SPAWN_RADIUS[1]:
                    continue
                if kx0 < p[0] < kx1 and abs(p[1]) < ky:
                    continue
                if all(np.linalg.norm(p - q) > separation for q in poses):
                    poses.append(p)
            if len(poses) == n:
                break
        else:
            raise RuntimeError(f"could not place {n} blocks without overlap")
        if self.random_yaw:
            return [(p, self.rng.uniform(-math.pi / 4, math.pi / 4)) for p in poses]
        return [(p, math.atan2(p[1], p[0])) for p in poses]

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.scene.step()
            if self.realtime:
                self._sim_steps += 1
                if self._wall_t0 is None:
                    self._wall_t0 = time.perf_counter()
                ahead = self._sim_steps * self.dt - (time.perf_counter() - self._wall_t0)
                if ahead > 0:
                    time.sleep(ahead)

    @property
    def cambot_pedestal_link(self) -> int:
        return self.cambot_pedestal.links[0].idx if self.cambot is not None else -1

    def link_owner(self, link_idx: int) -> str:
        """リンク番号から、それが属する物の名前（"作業アーム" / "ブロック (色)" / "床" など）を返す。ログ用。"""
        if any(link.idx == link_idx for link in self.robot.links):
            return "作業アーム"
        for color, block in zip(self.block_colors, self.blocks):
            if block.links[0].idx == link_idx:
                return f"ブロック ({color})"
        return "床・台座など"

    def gripper_pos(self, local_point: np.ndarray) -> np.ndarray:
        """作業アームの gripper 座標系の点 `local_point` のワールド座標（順運動学）。"""
        from genesis.utils.geom import quat_to_R

        R = quat_to_R(self.gripper_link.get_quat().cpu().numpy())
        return self.gripper_link.get_pos().cpu().numpy() + R @ local_point

    def render_wrist(self) -> np.ndarray:
        """手首カメラの現在の画像 (H, W, 3)。"""
        self.wrist_cam.move_to_attach()
        return self.wrist_cam.render()[0]

    def block_yaws(self) -> np.ndarray:
        """ブロックの向きの正解値（rad）, (N,)。デバッグと評価専用。"""
        q = np.stack([b.get_quat().cpu().numpy() for b in self.blocks])  # wxyz
        w, x, y, z = q.T
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def block_positions(self) -> np.ndarray:
        """ブロック中心の正解値, (N, 3)。デバッグと評価専用で、制御には使わない。"""
        return np.stack([b.get_pos().cpu().numpy() for b in self.blocks])
