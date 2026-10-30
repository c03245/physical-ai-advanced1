"""カメラワーク: カメラマン役の SO-101 にジンバル付きカメラを持たせ、作業アームを追いかけて撮影する。

* カメラマンのロボットは、カメラを置く「位置」だけを IK で決めて関節の位置制御で動く（5 自由度の SO-101 では、
  レンズの向きを鉛直面の外へ向けられないため）。カメラの向き（パン・チルト）と水平はジンバルが受け持つ想定で、
  注視点を向き、常に水平を保つ。
* 演出（寄り・仰ぎ見・俯瞰・回り込み）ごとに、被写体（作業アームの手先など）に対するカメラ位置を決める。
  目標は急に変えず、カメラ位置・注視点とも滑らかに追従させる。
* カメラが入ってよい範囲を、作業エリア（ブロックとピラミッドのある範囲）の外に限り、作業アームやブロックと
  ぶつからないようにする。
"""

from __future__ import annotations

import math

import numpy as np

from .env import CAMBOT_REST_Q, CAMBOT_XY, HOLD_LOCAL, PEDESTAL_H, PyramidScene

# ジンバル（カメラ）を付ける位置: 指先より約 3 cm 先（gripper 座標系）。自分の指が写り込まないようにする
GIMBAL_LOCAL = np.array([HOLD_LOCAL[0], 0.0, -0.13])

# カメラが入ってよい範囲: 作業エリアの奥（x >= 0.33、ピラミッドの端から約 6 cm）か、ブロックが置かれる範囲より
# 手前（y <= -0.25）。0.29 にしていたときは、低い位置に寄った手先がピラミッドの端を押していた
SAFE_X = 0.33
SAFE_Y = -0.25
MIN_Z = 0.06  # テーブルに近づきすぎない（指先が床や台座に触れないように）
MAX_REACH = 0.23  # カメラマンの肩からカメラまでの最大距離。腕を伸ばしきると前腕が肩に当たるので、余裕を持たせる
SHOULDER = np.array([*CAMBOT_XY, PEDESTAL_H + 0.062])  # カメラマンの肩のおおよその位置
MAX_JOINT_STEP = 0.03  # 目標の関節角の 1 フレーム（40 ms）あたりの変化の上限 [rad]
SERVO_SMOOTHING = 0.08  # 指令値を目標へ近づける割合（10 ms ごと。時定数は約 0.12 s）
STABILIZER = 0.35  # 画面に使うカメラ位置をならす割合（フレームごと）
ELBOW_MAX = 1.40  # 肘（elbow_flex）をこれ以上曲げると、前腕が肩に当たる（可動限界は 1.69 rad）
MIN_RADIAL = 0.20  # カメラマンの根元からカメラまでの水平距離の下限。近すぎると腕を後ろへ折りたたみ、自分の肩に当たる

PYRAMID_CENTER = np.array([0.21, 0.0, 0.04])

# 手先（ジンバル）の向き: 根元から外向きの方向を、この角度だけ下へ傾ける（度）。順に試す
APPROACH_PITCHES = np.radians([25.0, 45.0, 5.0, 65.0, -15.0])


def _outward_quat(pos: np.ndarray, pitch: float) -> np.ndarray:
    """手先の接近方向（gripper のローカル -z）を、カメラマンの根元から外向き・pitch だけ下向きにする姿勢。"""
    from genesis.utils.geom import R_to_quat

    radial = np.asarray(pos[:2], dtype=float) - np.array(CAMBOT_XY)
    radial /= max(np.linalg.norm(radial), 1e-6)
    approach = np.array([radial[0] * math.cos(pitch), radial[1] * math.cos(pitch), -math.sin(pitch)])
    z = -approach
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= max(np.linalg.norm(x), 1e-6)
    return R_to_quat(np.stack([x, np.cross(z, x), z], axis=1))


def _allowed(p: np.ndarray) -> bool:
    return (p[0] >= SAFE_X or p[1] <= SAFE_Y) and p[2] >= MIN_Z


def _make_safe(p: np.ndarray, focus: np.ndarray) -> np.ndarray:
    """禁止範囲に入った位置を被写体から遠ざかる向きに押し出し、カメラマンが無理なく届く範囲（根元に近すぎず、
    腕を伸ばしきらない）に収める。"""
    p = p.copy()
    d = p - focus
    d[2] = 0.0
    d = d / max(np.linalg.norm(d), 1e-6)
    for _ in range(40):
        if _allowed(p):
            break
        p[:2] += 0.01 * d[:2]
    p[2] = max(p[2], MIN_Z)
    r = p[:2] - np.array(CAMBOT_XY)
    if np.linalg.norm(r) < MIN_RADIAL:
        p[:2] = np.array(CAMBOT_XY) + r * MIN_RADIAL / max(np.linalg.norm(r), 1e-6)
    v = p - SHOULDER
    if np.linalg.norm(v) > MAX_REACH:
        p = SHOULDER + v * MAX_REACH / np.linalg.norm(v)
    return p


# ---- 演出: (被写体, 経過時間 [s]) → (カメラ位置, 注視点) -----------------------------------------------
def _spherical(center, azim_deg, elev_deg, dist):
    a, e = math.radians(azim_deg), math.radians(elev_deg)
    return center + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def shot_close_up(focus, t):
    """寄り: 手先のすぐ近くから、少し上から見る。"""
    return _spherical(focus, -25, 20, 0.16), focus


def shot_low_angle(focus, t):
    """仰ぎ見: テーブルすれすれの低い位置から、手先を見上げる。"""
    pos = _spherical(focus, -25, 0, 0.17)
    pos[2] = MIN_Z + 0.01  # 低すぎると、カメラマンが腕を限界まで低く伸ばして自分の腕に触れる
    return pos, focus + [0, 0, 0.02]


def shot_top_down(focus, t):
    """俯瞰: 高い位置から、ピラミッドと手先を見下ろす。"""
    target = 0.5 * (focus + PYRAMID_CENTER)
    return np.array([0.34, -0.08, 0.34]), target


def shot_orbit(focus, t):
    """回り込み: 被写体の周りを、ゆっくり横へ回る（約 0.1 rad/s）。"""
    azim = -75 + 12 * math.sin(0.35 * t)
    return _spherical(focus, azim, 25, 0.30), focus


def shot_hero(focus, t):
    """完成形: 低い位置から、ピラミッドの周りを回り込んで見上げる。"""
    azim = -80 + 50 * (0.5 - 0.5 * math.cos(min(t / 8.0, 1.0) * math.pi))
    pos = _spherical(PYRAMID_CENTER, azim, 8, 0.28)
    return pos, PYRAMID_CENTER + [0, 0, 0.03]


SHOTS = {
    "close_up": shot_close_up,
    "low_angle": shot_low_angle,
    "top_down": shot_top_down,
    "orbit": shot_orbit,
    "hero": shot_hero,
}
SHOT_JA = {"close_up": "寄り", "low_angle": "仰ぎ見", "top_down": "俯瞰", "orbit": "回り込み", "hero": "完成"}


class CameraOperator:
    """カメラマン役の SO-101 を動かし、ジンバル付きカメラで撮影する。"""

    def __init__(self, env: PyramidScene, pos_smoothing: float = 0.06, look_smoothing: float = 0.12):
        assert env.cambot is not None and env.cine_cam is not None, "PyramidScene(camera_robot=True) が必要"
        self.env = env
        self.bot = env.cambot
        self.link = self.bot.get_link("gripper")
        self.q = CAMBOT_REST_Q.copy()
        self.shot = "orbit"
        self.shot_t = 0.0
        self.pos_smoothing, self.look_smoothing = pos_smoothing, look_smoothing
        self.pitch_idx = 0
        # 撮影開始時から自然な姿勢で構える（待機姿勢からの急な動き出しで、自分の肩に触れないように）
        self.goal_pos = _make_safe(shot_orbit(PYRAMID_CENTER, 0.0)[0], PYRAMID_CENTER)
        self.q = self._solve(self.goal_pos)
        self.q[5] = CAMBOT_REST_Q[5]
        self.bot.set_dofs_position(self.q)
        self.bot.control_dofs_position(self.q)
        self.q_goal, self.q_cmd = self.q.copy(), self.q.copy()
        self.look = PYRAMID_CENTER.copy()
        self.cam_pos = self._gimbal_pos()
        self.contacts: dict[str, int] = {}  # カメラマンが触れた相手（自分の台以外）→ 回数。安全の確認用

    def _gimbal_pos(self) -> np.ndarray:
        from genesis.utils.geom import quat_to_R

        R = quat_to_R(self.link.get_quat().cpu().numpy())
        return self.link.get_pos().cpu().numpy() + R @ GIMBAL_LOCAL

    def set_shot(self, name: str) -> None:
        if name != self.shot:
            self.shot, self.shot_t = name, 0.0

    def update(self, focus: np.ndarray, dt: float) -> None:
        """被写体に合わせてカメラの目標位置を決め、カメラマンの関節に指令を送る（動画の 1 フレームごとに呼ぶ）。"""
        self.shot_t += dt
        pos, look = SHOTS[self.shot](np.asarray(focus, dtype=float), self.shot_t)
        pos = _make_safe(pos, np.asarray(focus, dtype=float))
        self.goal_pos += self.pos_smoothing * (pos - self.goal_pos)
        self.look += self.look_smoothing * (np.asarray(look, dtype=float) - self.look)
        q = self._solve(self.goal_pos)
        q[5] = CAMBOT_REST_Q[5]
        self.q = q
        # 目標の関節角は 1 フレームで大きく跳ばさない（解が切り替わったときの急な動きを抑える）
        self.q_goal += np.clip(q - self.q_goal, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        self._log_contacts()

    def servo(self) -> None:
        """シミュレーションの毎ステップ呼ぶ: 指令値を目標の関節角へ滑らかに近づけて送る（一次遅れ）。

        動画のフレームごと（40 ms ごと）に階段状に指令を変えると、サーボの位置制御が行き過ぎ・戻りを繰り返して
        カメラが振動するため、10 ms ごとに少しずつ動かす。
        """
        self.q_cmd += SERVO_SMOOTHING * (self.q_goal - self.q_cmd)
        self.bot.control_dofs_position(self.q_cmd)

    def _solve(self, pos: np.ndarray) -> np.ndarray:
        """カメラの位置に届く関節角。手先の向きも「根元から外向き・少し下向き」に指定して、自然な姿勢にする。

        位置だけを指定すると、手首の向きが自由に残り、手首を前腕側へ折り込む（自分の腕にめり込む）解が出ることが
        ある。向きも指定すると 5 自由度がすべて決まる。届かなければ傾きを変えて試し、最も誤差の小さい解を使う。
        """
        best, best_err = None, np.inf
        # 前回使った傾きを最初に試し、届く限りそれを使い続ける（解の切り替わりによる揺れを防ぐ）
        order = [self.pitch_idx] + [i for i in range(len(APPROACH_PITCHES)) if i != self.pitch_idx]
        best_idx = self.pitch_idx
        for i in order:
            pitch = APPROACH_PITCHES[i]
            q, err = self.bot.inverse_kinematics(
                link=self.link,
                pos=pos,
                quat=_outward_quat(pos, pitch),
                local_point=GIMBAL_LOCAL,
                rot_mask=[False, False, True],
                init_qpos=self.q,
                max_samples=5,
                return_error=True,
            )
            q, err = q.cpu().numpy(), err.cpu().numpy().ravel()
            e = np.linalg.norm(err[:3]) + 0.01 * np.linalg.norm(err[3:])
            if q[2] > ELBOW_MAX:  # 肘を曲げすぎた姿勢は前腕が肩に当たるので、他に解があればそちらを使う
                e += 0.05
            if e < best_err:
                best, best_err, best_idx = q, e, i
            if e < 0.003:
                break
        self.pitch_idx = best_idx
        return best

    def _log_contacts(self) -> None:
        """カメラマンが、自分の台以外（作業アーム・ブロック・床）に触れていないかを記録する。"""
        ct = self.bot.get_contacts()
        own = {link.idx for link in self.bot.links}
        for a, b in zip(ct["link_a"].tolist(), ct["link_b"].tolist()):
            other = b if a in own else a
            if other == self.env.cambot_pedestal_link:
                continue
            name = "自分の腕" if (a in own and b in own) else self.env.link_owner(other)
            if name not in self.contacts:
                print(f"[cinema] warning: camera robot touched {name}")
            self.contacts[name] = self.contacts.get(name, 0) + 1

    def render(self) -> np.ndarray:
        """ジンバル付きカメラの画像: 位置はロボットが実際に運んだ所、向きは注視点を向いて水平を保つ。"""
        # スタビライザー: ロボットが運んだ位置の細かな揺れを、画面に使う位置では軽くならす
        self.cam_pos += STABILIZER * (self._gimbal_pos() - self.cam_pos)
        pos = self.cam_pos
        look = self.look
        up = (0.0, 0.0, 1.0)
        if np.linalg.norm(np.cross(look - pos, up)) < 1e-3:  # 真下を向くときは上方向を変える
            up = (1.0, 0.0, 0.0)
        self.env.cine_cam.set_pose(pos=pos, lookat=look, up=up)
        return self.env.cine_cam.render()[0]
