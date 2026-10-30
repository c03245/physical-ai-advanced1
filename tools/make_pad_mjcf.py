"""assets/so101/so101_pads.xml を生成する: 手首カメラ付き SO-101 の MJCF に、平らな指パッドを追加したもの。

純正の指は内側が約 8° の V 字で、しかも可動指は回転するため、立方体は指先の線でしか挟めず、運ぶ途中で滑ったり
回転したりする。各指に薄い箱形のパッドを付けると、BLOCK 幅の物体を挟んだときに平行になる 2 枚の平面ができる
（実機のグリッパーに滑り止めのゴムパッドを貼るのと同じ考え方）。

元データは TheRobotStudio/SO-ARM100 の so101_new_calib_camera.xml（公式の手首カメラ取付具とカメラ付き）。

    uv run python tools/make_pad_mjcf.py
"""

import math
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "assets" / "so101" / "so101_new_calib_camera.xml"
DST = SRC.with_name("so101_pads.xml")

BLOCK = 0.025
FIXED_FACE_X = -0.0074  # 固定側パッドの内面の位置（gripper 座標系。指先より 0.5 mm 内側に出す）
PAD_Z = (-0.104, -0.080)  # パッドが指に沿って覆う範囲（gripper 座標系）
PAD_HALF_Y = 0.006
FIXED_THICK = 0.003
MOVING_THICK = 0.005
Q_PARALLEL = 0.20  # 可動側パッドが固定側と平行になる（間隔がちょうど BLOCK になる）グリッパー角

# moving_jaw_so101_v1 ボディの gripper 座標系での姿勢（MJCF より）
JAW_POS = np.array([0.0202, 0.0188, -0.0234])
JAW_QUAT = (0.707107, 0.707107, 0.0, 0.0)

# 元データの wrist_camera_mount ボディは重心が仮の値 (0, 0, 0) のままで、形状の外にある（Genesis が警告を出す）。
# wrist_camera_mount_so101_v1.stl から求めた体積の重心（ボディ座標系、m）に置き換える
MOUNT_INERTIAL_SRC = '<inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>'
MOUNT_COM = (-0.02554, -0.03062, 0.01708)


def quat_to_R(w, x, y, z):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def fmt(v):
    return " ".join(f"{float(a):.6g}" for a in v)


def pad_geoms(name, pos, quat, half):
    """衝突用の箱と、同じ形の見た目用の箱（黒いゴム風）を返す。"""
    common = f'type="box" pos="{fmt(pos)}" quat="{fmt(quat)}" size="{fmt(half)}"'
    return (
        f'<geom name="{name}" {common} friction="1.5 0.01 0.001" group="3"/>\n'
        f'<geom name="{name}_visual" {common} contype="0" conaffinity="0" group="2" rgba="0.12 0.12 0.12 1"/>\n'
    )


def main():
    zc, hz = (PAD_Z[0] + PAD_Z[1]) / 2, (PAD_Z[1] - PAD_Z[0]) / 2
    fixed = pad_geoms(
        "fixed_pad", [FIXED_FACE_X - FIXED_THICK / 2, 0, zc], [1, 0, 0, 0], [FIXED_THICK / 2, PAD_HALF_Y, hz]
    )

    # 可動側パッド: グリッパー角が Q_PARALLEL のときの位置を gripper 座標系で決め、可動指ボディの座標系に直す
    centre_g = np.array([FIXED_FACE_X + BLOCK + MOVING_THICK / 2, 0, zc])
    Rb = quat_to_R(*JAW_QUAT)
    c, s = math.cos(-Q_PARALLEL), math.sin(-Q_PARALLEL)
    Rq_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_local = Rq_inv @ Rb.T
    moving = pad_geoms(
        "moving_pad", R_local @ (centre_g - JAW_POS), R_to_quat(R_local), [MOVING_THICK / 2, PAD_HALF_Y, hz]
    )

    xml = SRC.read_text()
    site = '<site group="3" name="gripperframe"'
    jaw_joint_end = 'class="sts3215"/>'
    assert site in xml
    xml = xml.replace(site, fixed + site, 1)
    jaw_start = xml.index('<body name="moving_jaw_so101_v1"')
    k = xml.index(jaw_joint_end, jaw_start) + len(jaw_joint_end)
    xml = xml[:k] + "\n" + moving + xml[k:]
    mount_start = xml.index('<body name="wrist_camera_mount"')
    m = xml.index(MOUNT_INERTIAL_SRC, mount_start)
    assert m < xml.index("<body", mount_start + 1), "wrist_camera_mount の inertial が見つからない"
    xml = (
        xml[:m]
        + MOUNT_INERTIAL_SRC.replace('pos="0 0 0"', f'pos="{fmt(MOUNT_COM)}"')
        + xml[m + len(MOUNT_INERTIAL_SRC) :]
    )
    xml = xml.replace("<mujoco ", f"<!-- tools/make_pad_mjcf.py で生成: {SRC.name} + 指パッド -->\n<mujoco ", 1)
    DST.write_text(xml)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
