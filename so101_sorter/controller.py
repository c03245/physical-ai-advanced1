"""SO-101 のピック&プレース動作（Genesis の IK + 関節空間での補間）。

* 手首を 90° 回し、指がアームの鉛直面と直交する向き（接線方向）に閉じるようにする。ピラミッドの列はロボットから
  放射方向に並ぶので、指が同じ列の隣のブロックに入り込まない。
* 届く限り真上から掴む（そのためにアームを台に載せている）。傾けた接近が必要な場合は、掴むときと置くときを
  *同じ*傾きにして、ブロックを直立したまま離す。手首を回してあるので、傾けても挟んでいる面は傾かない。
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .env import GRASP_LOCAL, HOLD_LOCAL, PyramidScene

# 可動指側の余裕は約 5 mm（固定指側は 3 mm）。広げすぎると、可動指が隣のブロックやピラミッドに当たる
GRIPPER_OPEN = 0.31
GRIPPER_RELEASE = 0.28  # 離すときは少しだけ開く（約 6 mm。指でピラミッドをこすらないように）
GRIPPER_CLOSED = 0.12  # パッドは約 0.2 rad でブロックに触れる。少し先を指令して握り続けさせる
APPROACH_DZ = 0.045  # 掴む・置く位置の上で一旦止まる高さ
RELEASE_DZ = 0.001  # 置く高さのほぼ真上（ほぼ接触）で離し、ブロックを落とさない
SLOW_DZ = 0.012  # 掴む・置く直前の最後の 12 mm はゆっくり動かす

# 試す接近の傾き（rad）。0 = 真下向き。負 = グリッパーがロボット側へ傾く
PITCH_CANDIDATES = np.radians([0, -15, -30, -45])
IK_POS_TOL = 0.0025
PAN_AXIS_XY = np.array([0.0388, 0.0])  # shoulder_pan 関節の軸の位置（MJCF の "shoulder" ボディ）


def arm_heading(pos) -> float:
    """把持点を `pos` に持っていくための、アームの鉛直面の向き（方位角）。

    手首を 90° 回しているので、保持したブロックはその面から HOLD_LOCAL[0]（数 mm）横にずれている。
    """
    v = np.asarray(pos[:2], dtype=float) - PAN_AXIS_XY
    r = max(np.linalg.norm(v), abs(HOLD_LOCAL[0]) + 1e-6)
    return math.atan2(v[1], v[0]) - math.asin(HOLD_LOCAL[0] / r)


def snap_yaw(face_yaw: float, pos) -> float:
    """向き `face_yaw`（90° ごとに同じ）の面を、指がまっすぐ挟むためのグリッパーの向き。

    真上からの把持でのみ有効（手首の回転で指を任意の向きにできる）。等価な 4 つの向きのうち、`pos` での
    アームの自然な向きに最も近いものを選ぶ。
    """
    return yaw_options(face_yaw, pos)[0]


def yaw_options(face_yaw: float, pos) -> list[float]:
    """`snap_yaw` の候補を、アームの自然な向きに近い順に並べたもの（±180° 回すものは除く）。"""
    natural = arm_heading(pos)
    k = round((natural - face_yaw) / (math.pi / 2))
    best = face_yaw + k * math.pi / 2
    side = 1 if natural >= best else -1  # 次に近いのは、自然な向きの側に 90° 回したもの
    return [best, best + side * math.pi / 2, best - side * math.pi / 2]


def grasp_quat(yaw: float, pitch: float) -> np.ndarray:
    """グリッパーの姿勢（wxyz）。ローカル -z が接近方向、ローカル x が指の閉じる方向。

    Rz(yaw) でアームの鉛直面の向きを決め、Ry(pitch) でその面内で接近方向を傾け、最後の 90° の回転で
    指が水平に閉じるようにする。
    """
    from genesis.utils.geom import R_to_quat

    cz, sz, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    roll = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return R_to_quat(rz @ ry @ roll)


class ArmController:
    def __init__(self, env: PyramidScene, on_step: Callable[[], None] | None = None):
        self.env = env
        self.robot = env.robot
        self.on_step = on_step or (lambda: None)
        self.q = env.home_q.copy()
        self.gripper = GRIPPER_OPEN
        self.q[5] = self.gripper

    # ---- 低レベルの動作 ------------------------------------------------------------------------------
    def _tick(self, q: np.ndarray) -> None:
        self.robot.control_dofs_position(q)
        self.env.step()
        self.on_step()

    def move_joints(self, q_target: np.ndarray, duration: float, eased: bool = True) -> None:
        """関節空間での移動。直線経路の 1 区間でなければ、コサインで加減速する。"""
        q0 = self.q.copy()
        n = max(1, int(duration / self.env.dt))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n) if eased else i / n
            self._tick(q0 + (q_target - q0) * s)
        self.q = q_target.copy()

    def solve_ik(
        self, pos, pitch: float, yaw: float | None = None, local_point: np.ndarray = HOLD_LOCAL
    ) -> tuple[np.ndarray, float]:
        """gripper 座標系の点 `local_point` を、指定の傾き（真上なら指の向きも）で `pos` に置く IK。

        戻り値は (関節角, 誤差 [m])。現在の関節角と既定値の 2 通りの初期値で解き、局所解を避ける。
        """
        pos = np.asarray(pos, dtype=np.float64)
        if yaw is None:
            yaw = arm_heading(pos)
        best, best_err = None, np.inf
        for seed in (self.q, None):
            q, err = self.robot.inverse_kinematics(
                link=self.env.gripper_link,
                pos=pos,
                quat=grasp_quat(yaw, pitch),
                local_point=local_point,
                init_qpos=seed,
                max_samples=30,
                return_error=True,
            )
            q, err = q.cpu().numpy(), err.cpu().numpy().ravel()
            e = np.linalg.norm(err[:3]) + 0.02 * np.linalg.norm(err[3:])
            if e < best_err:
                best, best_err = q, e
            if e < IK_POS_TOL:
                break
        best[5] = self.gripper
        return best, best_err

    def move_tcp(
        self, pos, pitch: float, yaw: float | None, duration: float, local_point: np.ndarray = HOLD_LOCAL
    ) -> None:
        q, err = self.solve_ik(pos, pitch, yaw, local_point)
        if err > IK_POS_TOL:
            print(f"[controller] warning: IK target {np.round(pos, 3)} off by {err * 1000:.1f} mm")
        self.move_joints(q, duration)

    def move_linear(
        self,
        start,
        end,
        pitch: float,
        yaw: float | None,
        duration: float,
        n_waypoints: int = 6,
        local_point: np.ndarray = HOLD_LOCAL,
    ) -> None:
        """把持点をほぼ直線で動かす（真上からの接近・離脱用）。"""
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        for i in range(1, n_waypoints + 1):
            q, _ = self.solve_ik(start + (end - start) * i / n_waypoints, pitch, yaw, local_point)
            self.move_joints(q, duration / n_waypoints, eased=False)

    def set_gripper(self, value: float, duration: float = 0.6) -> None:
        self.gripper = value
        q = self.q.copy()
        q[5] = value
        self.move_joints(q, duration)

    def hold(self, duration: float) -> None:
        """現在の指令をしばらく保つ。"""
        for _ in range(int(duration / self.env.dt)):
            self._tick(self.q)

    def home(self, duration: float = 1.5) -> None:
        q = self.env.home_q.copy()
        q[5] = self.gripper
        self.move_joints(q, duration)

    # ---- タスクレベルの動作 ---------------------------------------------------------------------------
    def path_collides(self, q_target: np.ndarray, n: int = 8) -> bool:
        """今の関節角から `q_target` まで関節空間で動かす途中で、アームが何かに当たるか（自己衝突を含む）。

        IK は衝突を考えないので別に調べる。根元に近いブロックを真上から掴むときは腕を深く折りたたむので、手首の
        向きによっては手首カメラ（と取付具）が自分の肩や上腕に当たり、それ以上進めなくなる（seed 106 の橙で、
        6 回続けて掴めなかった）。シミュレーションを進めずに、途中の姿勢を置いて衝突だけを調べ、元に戻す。
        """
        robot = self.robot
        q_now = robot.get_dofs_position()
        v_now = robot.get_dofs_velocity()
        try:
            for t in np.linspace(0.0, 1.0, n + 1)[1:]:
                robot.set_dofs_position(self.q + (q_target - self.q) * t)
                if len(robot.detect_collision()):
                    return True
            return False
        finally:
            robot.set_dofs_position(q_now)
            robot.set_dofs_velocity(v_now)

    def grasp_yaw(self, face_yaw: float, grasp_pts) -> float:
        """真上から挟む向きの候補のうち、掴む位置の真上まで動かす途中でアームがどこにも当たらないもの
        （自然な向きに近いものを優先。どれも当たるなら自然な向き）。"""
        options = yaw_options(face_yaw, grasp_pts[0])
        for yaw in options:
            q, err = self.solve_ik(grasp_pts[1], 0.0, yaw, GRASP_LOCAL)
            if err < IK_POS_TOL and not self.path_collides(q):
                if yaw != options[0]:
                    print(f"[controller] grasp yaw {math.degrees(yaw):.0f} deg (avoids a collision)")
                return yaw
        return options[0]

    def plan_grasp(self, grasp_pts, release_pts, block_yaw, target_yaw):
        """すべての経由点に届く (傾き, 掴むときの向き, 置くときの向き) を選ぶ。

        真上からを優先する。その場合、掴むときは指をブロックの面に（移動中にアームが当たる向きは避ける）、置くときは
        `target_yaw` に合わせる。傾けた接近になる場合はアームの自然な向きを使う（ブロックはアームの鉛直面に対する
        向きを保ったまま運ばれる）。
        """
        errs = {}
        for pitch in PITCH_CANDIDATES:
            if pitch == 0 and block_yaw is not None and target_yaw is not None:
                gy, ry = self.grasp_yaw(block_yaw, grasp_pts), snap_yaw(target_yaw, release_pts[0])
            else:
                gy = ry = None
            err = max(
                [self.solve_ik(p, pitch, gy, GRASP_LOCAL)[1] for p in grasp_pts]
                + [self.solve_ik(p, pitch, ry)[1] for p in release_pts]
            )
            errs[pitch] = (err, gy, ry)
            if err < IK_POS_TOL:
                return float(pitch), gy, ry
        best = min(errs, key=lambda k: errs[k][0])
        print(
            f"[controller] warning: no tilt reaches all points; best {math.degrees(best):.0f} deg "
            f"(err {errs[best][0] * 1000:.1f} mm)"
        )
        return float(best), errs[best][1], errs[best][2]

    def pick_and_place(
        self,
        block_xyz,
        target_xyz,
        block_yaw: float | None = None,
        target_yaw: float | None = 0.0,
        speed: float = 1.0,
        refine_grasp: Callable | None = None,
        after_release: Callable | None = None,
    ) -> float:
        """block_xyz を中心とするブロックを掴み、中心が target_xyz に来るように置く。

        block_yaw / target_yaw (rad): ブロックの面の現在の向きと、置いたときの向き（None = 気にしない）。
        refine_grasp(xyz, yaw): 掴む位置の真上で一旦止まったときに呼ばれ、手首カメラで測り直した (xyz, yaw) を
        返す（測れなければ None）。真上からの把持のときだけ使う。
        after_release(): 離して真上に戻ったところで呼ばれる（手首カメラで置いた結果を確かめる）。
        戻り値は使った傾き (rad)。
        """
        block_xyz = np.asarray(block_xyz, dtype=float)
        target_xyz = np.asarray(target_xyz, dtype=float)

        def waypoints():
            grasp = np.array([block_xyz[0], block_xyz[1], block_xyz[2] - 0.0005])
            release = target_xyz + [0, 0, RELEASE_DZ]
            carry_z = max(grasp[2], release[2]) + APPROACH_DZ
            return grasp, np.array([grasp[0], grasp[1], carry_z]), release, release + [0, 0, APPROACH_DZ]

        grasp, above_grasp, release, above_release = waypoints()
        pitch, gy, ry = self.plan_grasp([grasp, above_grasp], [release, above_release], block_yaw, target_yaw)

        self.set_gripper(GRIPPER_OPEN, 0.4 / speed)
        self.move_tcp(above_grasp, pitch, gy, 1.5 / speed, GRASP_LOCAL)
        if refine_grasp is not None and pitch == 0:
            self.hold(0.2)  # 揺れが収まってから撮る
            r = refine_grasp(block_xyz, block_yaw)
            if r is not None:
                block_xyz, block_yaw = np.asarray(r[0], dtype=float), r[1]
                grasp, above_grasp, release, above_release = waypoints()
                # 測り直した向きでも、選んだ候補と同じ側の向きを使う
                gy = min(yaw_options(block_yaw, grasp), key=lambda y: abs(y - gy))
                self.move_tcp(above_grasp, pitch, gy, 0.5 / speed, GRASP_LOCAL)
        near_grasp = grasp + [0, 0, SLOW_DZ]
        self.move_linear(above_grasp, near_grasp, pitch, gy, 0.8 / speed, 4, GRASP_LOCAL)
        self.move_linear(near_grasp, grasp, pitch, gy, 0.6 / speed, 3, GRASP_LOCAL)
        self.set_gripper(GRIPPER_CLOSED, 0.8 / speed)
        self.hold(0.4)  # 減衰の大きい指が閉じきるまで待つ
        # ここからブロックは HOLD_LOCAL で保持されている。その点を基準に真上へ持ち上げる
        self.move_linear(grasp, above_grasp, pitch, gy, 1.0 / speed)
        self.move_tcp(above_release, pitch, ry, 1.8 / speed)
        near_release = release + [0, 0, SLOW_DZ]
        self.move_linear(above_release, near_release, pitch, ry, 0.8 / speed, n_waypoints=4)
        self.move_linear(near_release, release, pitch, ry, 1.0 / speed, n_waypoints=4)
        self.hold(0.2)  # 指を開く前に、ブロックを支えの上で落ち着かせる
        self.set_gripper(GRIPPER_RELEASE, 1.0 / speed)
        self.move_linear(release, above_release, pitch, ry, 0.8 / speed, n_waypoints=4)
        self.gripper = GRIPPER_OPEN
        if after_release is not None:
            self.hold(0.2)
            after_release()
        return pitch
