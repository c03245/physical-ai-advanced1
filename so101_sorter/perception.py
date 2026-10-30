"""AI による認識: オープン語彙の物体検出（OWLv2）と、画素から 3D のブロックの位置・向きへの変換。

画素を 3D に持ち上げる方法は 2 通り（`View` を参照）:
  * RGB-D: 深度カメラの点群をそのまま使う。
  * RGB のみ（普通のカメラ）: 較正済みのピンホールカメラモデルを使い、画素の光線を、高さが分かっている
    ブロック上面の水平面（テーブルやピラミッドの各段）と交わらせる。深度センサは不要。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .env import BLOCK_SIZE, COLORS, TABLE_COLOR

MODEL_ID = "google/owlv2-base-patch16-ensemble"
# OWLv2 に渡す色の名前（英語で自然な言い方）
QUERY_NAMES = {"cyan": "light blue", "lime": "lime green"}


def lit_top_color(rgb) -> np.ndarray:
    """物体の色が、照明の当たった上面でどう写るか（シミュレーションの画像で測った関係を直線で近似したもの）。

    実機なら、色見本を撮影してこの関係（カメラの色の較正）を求める。
    """
    return 0.856 * np.asarray(rgb, dtype=np.float32) + 0.092


# 画素ごとの色分類に使う参照色。机・机の影・台座・ロボットの色も入れて、それらがブロックと判定されないようにする
_DISTRACTORS = {
    "_table": lit_top_color(TABLE_COLOR),
    "_table_shadow": np.array(TABLE_COLOR, dtype=np.float32) * 0.43,
    "_robot": lit_top_color((0.95, 0.8, 0.15)),
    "_pedestal": lit_top_color((0.25, 0.25, 0.28)),
}
_REF = np.array([lit_top_color(c) for c in COLORS.values()] + list(_DISTRACTORS.values()), dtype=np.float32)
_NAMES = list(COLORS) + list(_DISTRACTORS)
COLOR_MAX_DIST = 0.22  # どの参照色からもこれ以上離れた画素（暗い側面など）は、どの色にも分類しない
WORKSPACE_R = (0.09, 0.33)  # ブロックがありうるロボットの根元からの距離 [m]


@dataclass
class CameraModel:
    """較正済みのピンホールカメラ: 内部パラメータ K と、カメラ → ワールドの姿勢 T_wc（OpenCV の軸: x 右, y 下, z 前）。

    実機ではチェッカーボードや ArUco マーカーによる較正値を使う。シミュレーションでは Genesis のカメラから作り、
    必要なら較正誤差を加える（with_error）。
    """

    K: np.ndarray
    T_wc: np.ndarray

    @classmethod
    def from_genesis(cls, cam) -> CameraModel:
        # Genesis のカメラ姿勢は OpenGL の軸（y が上、-z 方向を見る）なので、y と z を反転する
        return cls(
            np.asarray(cam.intrinsics, dtype=float), np.asarray(cam.transform, dtype=float) @ np.diag([1, -1, -1, 1])
        )

    def with_error(self, delta_T: np.ndarray) -> CameraModel:
        """較正誤差を含んだモデル: 真の姿勢にカメラ座標系での一定のずれ delta_T（4x4）を掛けたもの。

        固定カメラなら外部パラメータの較正誤差、手首カメラならハンドアイ較正（取付位置）の誤差にあたる。
        """
        return CameraModel(self.K.copy(), self.T_wc @ delta_T)

    def plane_points(self, shape: tuple[int, int], z: float) -> np.ndarray:
        """各画素の光線が高さ z の水平面と交わるワールド座標, (H, W, 3)。交わらなければ NaN。"""
        h, w = shape
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(self.K).T @ self.T_wc[:3, :3].T
        o = self.T_wc[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - o[2]) / rays[..., 2]
        pts = o + t[..., None] * rays
        pts[~(t > 0)] = np.nan
        return pts


def random_pose_error(trans_m: float, rot_deg: float, rng: np.random.Generator) -> np.ndarray:
    """大きさが trans_m [m] / rot_deg [度] で、向きがランダムな剛体変換（較正誤差の模擬用）。"""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = math.radians(rot_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k
    t = rng.normal(size=3)
    T[:3, 3] = t * trans_m / np.linalg.norm(t)
    return T


class View:
    """カメラ画像 1 枚と、その画素を 3D に持ち上げる手段（点群 = RGB-D、またはカメラモデル = RGB のみ）の組。"""

    def __init__(self, rgb: np.ndarray, points: np.ndarray | None = None, camera: CameraModel | None = None):
        assert (points is None) != (camera is None), "give either a point cloud or a camera model"
        self.rgb, self.points, self.camera = rgb, points, camera
        self._planes: dict[float, np.ndarray] = {}

    @property
    def camera_xy(self) -> np.ndarray | None:
        """RGB のみの場合のカメラの水平位置（側面のはみ出しの向きを判定するのに使う）。"""
        return None if self.camera is None else self.camera.T_wc[:2, 3]

    def lift(self, mask: np.ndarray, top_z: float | None, region=np.s_[:, :]) -> np.ndarray:
        """`mask`（`region` 部分の真偽画像）のうち上面の画素の 3D 点, (N, 3)。

        top_z: 上面の高さの見込み。RGB-D なら None でもよい（最も高い点を上面とみなす）。RGB のみでは必須で、
        上面の明るさ（真上から照らされて最も明るい面）の画素だけを残し、側面の大部分を除く。
        """
        if self.points is not None:
            pts = self.points[region][mask]
            pts = pts[np.isfinite(pts).all(1)]
            return pts if top_z is None else pts[np.abs(pts[:, 2] - top_z) < 0.004]
        return self.lift_layers(mask, top_z, region)[0]

    def lift_layers(self, mask: np.ndarray, top_z: float | None, region=np.s_[:, :]) -> list[np.ndarray]:
        """RGB のみの場合に、`mask` を明るさの山ごとに分けて、明るいものから順に 3D に持ち上げた点のリスト。

        ふつうは最も明るい面が上面だが、照明とカメラの向きによっては側面のほうが明るく写る。呼び出し側で、
        上面 1 枚分の形になっているものを順に探す。RGB-D なら `lift` と同じ 1 つだけを返す。
        """
        if self.points is not None:
            return [self.lift(mask, top_z, region)]
        if top_z is None:
            raise ValueError("RGB-only lifting needs the top-face height")
        key = round(top_z, 5)
        if key not in self._planes:
            self._planes[key] = self.camera.plane_points(self.rgb.shape[:2], top_z)
        layers = []
        for layer in _brightness_layers(self.rgb[region], mask):
            pts = self._planes[key][region][mask & layer]
            layers.append(pts[np.isfinite(pts).all(1)])
        return layers or [np.zeros((0, 3))]


def _brightness_layers(rgb: np.ndarray, mask: np.ndarray, tol: float = 6.0) -> list[np.ndarray]:
    """`mask` の明るさの分布で十分な量のある山ごとに、その山から `tol` 以内の画素の真偽画像（明るい順）。

    1 つの面は照明の当たり方が一様なので、ほぼ同じ明るさで写る。最も多い明るさだけを使うと、黒いブロックでは
    暗い側面や影、指のパッドの画素のほうが多く、上面を取り逃がしていた。
    """
    if mask.sum() == 0:
        return []
    v = rgb.max(-1).astype(float)
    hist = np.convolve(np.bincount(v[mask].astype(int), minlength=256), np.ones(5), "same")
    peaks = [
        i for i in range(1, 255) if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] >= 0.3 * hist.max()
    ]
    levels = sorted(set(peaks), reverse=True) or [int(np.argmax(hist))]
    return [np.abs(v - level) <= tol for level in levels]


@dataclass
class Detection:
    color: str
    score: float
    box: tuple[int, int, int, int]  # 画素座標の x0, y0, x1, y1
    position: np.ndarray  # ワールド座標でのブロック中心 [m]
    yaw: float  # 面の向き（rad、-45〜45 度の範囲）


class BlockDetector:
    """テキストで指示する検出器で「<色> の立方体」を見つけ、それぞれの 3D の位置・向きを求める。"""

    def __init__(self, device: str | None = None, score_threshold: float = 0.12):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(MODEL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(self.device).eval()
        self.score_threshold = score_threshold

    @torch.no_grad()
    def detect(self, view: View, colors: list[str], top_z: float | None = None) -> list[Detection]:
        """指定した色のブロックをすべて返す。top_z: 上面の高さ（RGB のみの場合、省略時はテーブル上とみなす）。"""
        rgb = view.rgb
        if view.points is None and top_z is None:
            top_z = BLOCK_SIZE
        h, w = rgb.shape[:2]
        queries = [f"a photo of a small {QUERY_NAMES.get(c, c)} cube" for c in colors]
        inputs = self.processor(text=[queries], images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        side = max(h, w)  # OWLv2 は画像を正方形にパディングして処理する
        result = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=[(side, side)], text_labels=[queries]
        )[0]

        # 役割分担: OWLv2（AI）は「どこにブロックがあるか」、較正した色の判定は「何色か」を受け持つ。
        # 9 色の名前で同時に問い合わせると、OWLv2 は枠ごとに 1 つの色名しか付けず、似た色（赤と茶など）を取り違える
        labels = classify_pixels(rgb)
        cands = []
        for score, box in zip(result["scores"].tolist(), result["boxes"].tolist()):
            x0, y0, x1, y1 = (round(v) for v in box)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            # 枠の中で画素の多い色から順に試す。3D の推定にはその色の画素だけを使うので、枠が隣のブロックにかかって
            # いても正確さを保てる
            region = labels[y0:y1, x0:x1]
            counts = {c: int((region == _NAMES.index(c)).sum()) for c in colors}
            for color in sorted(counts, key=counts.get, reverse=True)[:2]:
                if counts[color] < 15:
                    break
                mask = region == _NAMES.index(color)
                pose = None
                for pts in view.lift_layers(mask, top_z, np.s_[y0:y1, x0:x1]):
                    pose = _top_face_pose(pts, view.camera_xy)
                    if pose is not None:
                        break
                if (
                    pose is None
                    or not (WORKSPACE_R[0] < np.linalg.norm(pose[0][:2]) < WORKSPACE_R[1])
                    or pose[0][2] > 0.15
                ):
                    continue
                cands.append(Detection(color, score, (x0, y0, x1, y1), *pose))

        # 実物 1 個につき検出 1 つ: 3D の中心がよりスコアの高い検出と重なる候補は捨てる
        cands.sort(key=lambda d: -d.score)
        kept: list[Detection] = []
        for d in cands:
            if all(np.linalg.norm(d.position[:2] - k.position[:2]) > BLOCK_SIZE for k in kept):
                kept.append(d)
        return kept


def classify_pixels(rgb: np.ndarray) -> np.ndarray:
    """各画素を、照明の当たった上面の色が最も近い参照色に分類する。(H, W) の参照色番号、どれにも近くなければ -1。

    色を「色合い（明るさで割った比）」と「明るさ」に分けて比べる。白と黒は色合いが同じで明るさが違い、黒い
    ブロックと机の影は明るさが近くても色合いが違う（影は木の色を帯びる）ので、両方を使うと見分けられる。
    """
    px = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    v = px.max(1, keepdims=True)
    ref_v = _REF.max(1, keepdims=True)
    chroma = np.linalg.norm(px[:, None, :] / np.maximum(v, 1e-3)[:, None, :] - (_REF / ref_v)[None], axis=-1)
    dist = chroma + 1.5 * np.abs(v - ref_v.T)
    nearest = np.argmin(dist, axis=1)
    nearest[dist[np.arange(len(px)), nearest] >= COLOR_MAX_DIST] = -1
    return nearest.reshape(rgb.shape[:2])


def color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    """照明の当たった上面の色が `color` に最も近い画素。"""
    return classify_pixels(rgb) == _NAMES.index(color)


def _top_face_pose(pts: np.ndarray, camera_xy: np.ndarray | None = None):
    """1 個のブロックの 3D 点から、上面だけを使ってブロック中心と向き (yaw) を求める。

    camera_xy: RGB のみの場合のカメラの水平位置。上面の高さに投影した側面の画素はカメラ側へはみ出すので、
    幅がブロック 1 個分を超えた軸では、カメラから遠い側の辺を基準にブロック 1 個分だけを残す。
    """
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 10:
        return None
    z_top = np.percentile(pts[:, 2], 95)
    top = pts[pts[:, 2] > z_top - 0.003]
    if len(top) < 8 or z_top < BLOCK_SIZE * 0.6:
        return None
    yaw = _square_yaw(top[:, :2])
    c, s = math.cos(-yaw), math.sin(-yaw)
    local = top[:, :2] @ np.array([[c, -s], [s, c]]).T
    lo, hi = local.min(0), local.max(0)
    extent = hi - lo
    # 上面 1 枚分の大きさであること: 小さい = 隠れている・一部だけ、大きい = 同色のブロック 2 個が合わさっている
    if extent.min() < BLOCK_SIZE * 0.7 or extent.max() > BLOCK_SIZE * 1.4:
        return None
    centre_local = (lo + hi) / 2  # 外接矩形の中心は、点の粗密に左右されにくい
    if camera_xy is not None:
        cam_local = np.asarray(camera_xy, dtype=float) @ np.array([[c, -s], [s, c]]).T
        for axis in (0, 1):
            if extent[axis] > BLOCK_SIZE * 1.04:  # 側面がはみ出している
                if cam_local[axis] < centre_local[axis]:  # カメラは小さい側 → 大きい側の辺が正しい
                    centre_local[axis] = hi[axis] - BLOCK_SIZE / 2
                else:
                    centre_local[axis] = lo[axis] + BLOCK_SIZE / 2
    c, s = math.cos(yaw), math.sin(yaw)
    centre = np.array([[c, -s], [s, c]]) @ centre_local
    return np.array([centre[0], centre[1], z_top - BLOCK_SIZE / 2]), yaw


def _square_yaw(xy: np.ndarray) -> float:
    """点の集まりから正方形の向きを求める: 外接矩形の面積が最小になる回転角。"""
    angles = np.radians(np.arange(-45.0, 45.0, 1.0))
    best, best_area = 0.0, np.inf
    for a in angles:
        c, s = math.cos(-a), math.sin(-a)
        r = xy @ np.array([[c, -s], [s, c]]).T
        area = np.prod(r.max(0) - r.min(0))
        if area < best_area:
            best, best_area = a, area
    return float(best)


def measure_block_near(view: View, color: str, xyz, half_window: float = 0.025):
    """`xyz` にある（軸に沿って置かれた）はずの `color` のブロックの位置・向き。見つからなければ None。

    置いた結果の確認と、手首カメラでの位置合わせに使う。2D の検出器では隣のブロックとまとまってしまう密集した
    ピラミッドの中でも使えるよう、目標の周りの、予想される高さにあるその色の上面の点を集める。ある軸で幅が
    ブロック 1 個分を大きく超えたら、同じ色の別のブロックが混ざっているので、ブロック 1 個分の幅の窓で点が最も
    多く入る位置を探し、その窓の点だけを残す（ほぼ同数なら目標に近いほう）。

    以前は目標に近い側の辺を基準にしていたが、両側に同じ色が少しずつ混ざると（ピンクの上にピンクを積み、手前に
    下の段のピンクの上面がのぞく場合など）、混ざった側の端を辺と取り違えて 12 mm ずれた値を返していた。
    """
    xyz = np.asarray(xyz, dtype=float)
    top_z = xyz[2] + BLOCK_SIZE / 2
    # 先に目標の周りの画素だけに絞る（遠くにある同じ色のもの、例えば黒い指パッドを、上面の明るさの判定に混ぜない）
    if view.points is not None:
        near = view.points
    else:
        near = view.camera.plane_points(view.rgb.shape[:2], top_z)
    window = (np.abs(near[..., 0] - xyz[0]) < half_window) & (np.abs(near[..., 1] - xyz[1]) < half_window)
    pts = view.lift(color_mask(view.rgb, color) & window, top_z)
    if len(pts) < 20:
        return None
    keep = np.ones(len(pts), bool)
    for axis in (0, 1):
        v = pts[:, axis]
        lo, hi = v.min(), v.max()
        if hi - lo > BLOCK_SIZE * 1.15:
            half = BLOCK_SIZE / 2 + 0.001
            centres = np.arange(lo + BLOCK_SIZE / 2, hi - BLOCK_SIZE / 2 + 1e-9, 0.0005)
            counts = np.array([np.count_nonzero(np.abs(v - c) <= half) for c in centres])
            good = centres[counts >= 0.95 * counts.max()]
            c = good[np.argmin(np.abs(good - xyz[axis]))]
            keep &= np.abs(v - c) <= half
    return _top_face_pose(pts[keep], view.camera_xy)
