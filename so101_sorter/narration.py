"""解説ナレーション: 日本語の音声合成（pyopenjtalk、ローカルで動き API 不要）で読み上げ、動画に音声として重ねる。

使い方:
    narrator = Narrator()
    narrator.say(video.seconds, "読み上げる文")   # 動画内の時刻と文を記録する
    ...
    narrator.mux(video_path, video.seconds)       # 動画を書き出した後、音声トラックを合成する
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
from functools import cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48000
SPEED = 1.15  # 少し速めに読む
GAP = 0.25  # 前の読み上げとの最小の間隔 [s]
MAX_DELAY = 2.5  # 予定時刻よりこれ以上遅れる読み上げは省く [s]（essential=True のものは省かない）


@cache
def synthesize(text: str) -> np.ndarray:
    """text の読み上げ音声（float32, -1〜1, 48 kHz）。同じ文は 1 回だけ合成する。"""
    # pyopenjtalk-plus は ONNX Runtime がないと、読み込み時に「何」の読み分け（なに/なん）が無効だと print する。
    # 台本に「何」は出てこないので、その表示だけを捨てる
    with contextlib.redirect_stdout(io.StringIO()):
        import pyopenjtalk

    x, sr = pyopenjtalk.tts(text, speed=SPEED)
    assert sr == SAMPLE_RATE, sr
    x = x.astype(np.float32)
    return 0.8 * x / max(float(np.abs(x).max()), 1e-6)


def duration(text: str) -> float:
    """読み上げにかかる時間 [s]。"""
    return len(synthesize(text)) / SAMPLE_RATE


class Narrator:
    def __init__(self):
        self.events: list[tuple[float, str, bool]] = []  # (動画内の時刻 [s], 文, 省略不可か)

    def say(self, t: float, text: str, essential: bool = False) -> None:
        self.events.append((t, text, essential))

    def build_track(self, total_seconds: float) -> np.ndarray:
        """読み上げを時刻どおりに並べた音声。前の読み上げと重なるときは後ろにずらし、遅れすぎるものは省く。"""
        track = np.zeros(int((total_seconds + 1.0) * SAMPLE_RATE), dtype=np.float32)
        busy_until = 0.0
        for t, text, essential in sorted(self.events, key=lambda e: e[0]):
            start = max(t, busy_until + GAP)
            if start - t > MAX_DELAY and not essential:
                continue
            x = synthesize(text)
            i = int(start * SAMPLE_RATE)
            if i >= len(track):
                break
            n = min(len(x), len(track) - i)
            track[i : i + n] += x[:n]
            busy_until = start + len(x) / SAMPLE_RATE
        return np.clip(track, -1.0, 1.0)

    def mux(self, video_path: Path, total_seconds: float) -> None:
        """動画に音声トラックを合成して、同じファイル名で置き換える。

        ffmpeg は imageio-ffmpeg に同梱のものを使う（システムに ffmpeg が入っていなくても動く）。
        """
        import imageio_ffmpeg
        import scipy.io.wavfile

        video_path = Path(video_path)
        wav = video_path.with_suffix(".narration.wav")
        tmp = video_path.with_suffix(".tmp.mp4")
        scipy.io.wavfile.write(wav, SAMPLE_RATE, (self.build_track(total_seconds) * 32767).astype(np.int16))
        try:
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-i",
                    str(wav),
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    "-shortest",
                    str(tmp),
                ],
                check=True,
            )
            os.replace(tmp, video_path)
        finally:
            # 途中で失敗しても、作業用のファイルを残さない
            wav.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
