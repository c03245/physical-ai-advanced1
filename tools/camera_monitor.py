"""カメラ映像の表示ウィンドウ（main.py が別プロセスで起動する）。

main.py が書き出す画像ファイルを監視し、更新されたら OpenCV のウィンドウに表示し直す。Genesis のビューアと同じ
プロセスで OpenCV のウィンドウを使うと、WSL ではスレッドどうしが待ち合って止まることがあるため、プロセスを分けている。

    uv run python tools/camera_monitor.py <画像ファイル> [<親プロセスの PID>]
"""

import os
import sys
import time

import cv2

# opencv-python 同梱の Qt はフォントを持たないのに、cv2 は読み込み時に QT_QPA_FONTDIR を存在しない cv2/qt/fonts に
# 設定するので、ウィンドウを開くたびに警告が出る。Qt がこの変数を読むのは最初のウィンドウを作るときなので、
# ここでシステムのフォントのフォルダに設定し直す（ウィンドウの表示は英数字だけなので、どのフォントでもよい）
for _font_dir in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF", "/usr/share/fonts/dejavu"):
    if os.path.isdir(_font_dir):
        os.environ["QT_QPA_FONTDIR"] = _font_dir
        break


def main():
    path = sys.argv[1]
    parent = int(sys.argv[2]) if len(sys.argv) > 2 else None
    title = "cameras: overhead (AI) | wrist"
    last = 0.0
    while True:
        if parent is not None:
            try:
                os.kill(parent, 0)  # 親プロセスが終わったら閉じる
            except OSError:
                break
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = last
        if mtime != last:
            img = cv2.imread(path)
            if img is not None:
                cv2.imshow(title, img)
                last = mtime
        if cv2.waitKey(30) == 27:  # Esc で閉じる
            break
        time.sleep(0.01)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
