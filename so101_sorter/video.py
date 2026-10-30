"""デモ動画の書き出し: 字幕、カメラ映像のワイプ、タイトルカードを PIL で描く。"""

from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@cache
def font(size: int) -> ImageFont.ImageFont:
    """英数字も含む日本語フォント（標準の太さ）。見つからなければ PIL の既定フォント（英数字のみ）。"""
    try:
        out = subprocess.run(
            ["fc-list", ":lang=ja:charset=30 41 61", "--format", "%{file}|%{index}|%{style}\n"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        for line in sorted(out.splitlines()):
            path, index, style = line.split("|", 2)
            if "Regular" in style and "Italic" not in style:
                return ImageFont.truetype(path, size, index=int(index))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return ImageFont.load_default(size)


def annotate_detections(rgb: np.ndarray, detections) -> np.ndarray:
    """検出結果（枠・色名・スコア）を画像に描き込む。"""
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for det in detections:
        d.rectangle(det.box, outline=(255, 230, 0), width=3)
        d.text(
            (det.box[0], max(0, det.box[1] - 22)),
            f"{det.color} {det.score:.2f}",
            font=font(18),
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return np.asarray(im)


class VideoWriter:
    def __init__(self, path: Path, fps: int = 25, res: tuple[int, int] = (1280, 720)):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps, self.res = fps, res
        self.writer = imageio.get_writer(self.path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
        self.caption = ""
        self.subcaption = ""
        # 右上に縦に並べるカメラ映像: ラベル → 画像（None なら表示しない）
        self.insets: dict[str, np.ndarray | None] = {}
        self.label = ""  # 左上に出す短い表示（今のカメラの演出名など）
        self.n_frames = 0

    def add(self, frame: np.ndarray) -> None:
        im = Image.fromarray(frame).convert("RGB")
        if im.size != self.res:
            im = im.resize(self.res)
        d = ImageDraw.Draw(im, "RGBA")
        w, h = self.res
        y = 12
        for label, img in self.insets.items():
            if img is None:
                continue
            iw = w // 5
            inset = Image.fromarray(img).resize((iw, iw * img.shape[0] // img.shape[1]))
            x = w - iw - 12
            im.paste(inset, (x, y))
            d.rectangle((x, y, x + iw, y + inset.height), outline=(255, 255, 255), width=2)
            d.text((x + 6, y + 4), label, font=font(16), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            y += inset.height + 10
        if self.label:
            d.text((16, 12), self.label, font=font(20), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        if self.caption:
            d.rectangle((0, h - 92, w, h), fill=(0, 0, 0, 150))
            d.text((24, h - 84), self.caption, font=font(30), fill=(255, 255, 255))
            d.text((24, h - 40), self.subcaption, font=font(22), fill=(210, 210, 210))
        self.writer.append_data(np.asarray(im))
        self.n_frames += 1

    def title_card(self, lines: list[str], seconds: float, background: np.ndarray | None = None) -> None:
        """背景を暗くして、中央に文字を並べたカードを `seconds` 秒表示する（1 行目は大きく）。"""
        w, h = self.res
        if background is None:
            im = Image.new("RGB", self.res, (24, 28, 36))
        else:
            im = Image.fromarray(background).convert("RGB").resize(self.res)
            im = Image.blend(im, Image.new("RGB", self.res, (0, 0, 0)), 0.55)
        d = ImageDraw.Draw(im)
        y = h // 2 - 30 * len(lines)
        for i, line in enumerate(lines):
            size = 44 if i == 0 else 28
            tw = d.textlength(line, font=font(size))
            d.text(((w - tw) / 2, y), line, font=font(size), fill=(255, 255, 255))
            y += size + 22
        frame = np.asarray(im)
        for _ in range(int(seconds * self.fps)):
            self.writer.append_data(frame)
            self.n_frames += 1

    @property
    def seconds(self) -> float:
        return self.n_frames / self.fps

    def close(self) -> None:
        self.writer.close()
