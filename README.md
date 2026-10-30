# SO-101 × AI ピラミッド積み（Genesis）

Physical AI 応用1講座の最終課題として制作した作品です。
シミュレータ [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 上のロボットアーム **SO-101** が、**AI による物体認識**と**ローカル LLM によるタスク計画**を使い、机に散らばった 9 色・27 個のブロックから言語指示に合う 10 個を選んで、4・3・2・1 段のピラミッドに積みます。
デモ動画は、もう 1 台の SO-101 を「カメラマン」にして、ジンバル付きカメラで作業の様子を追いかけて撮影します。固定カメラだけでは得られない、寄り・仰ぎ見・俯瞰・回り込みのダイナミックな視点をロボット自身が作ります。

```
            ■          ← 4段目（頂上）
           ■■         ← 3段目
          ■■■        ← 2段目
         ■■■■       ← 1段目（土台）

「青いブロックを土台にして、緑、橙、赤の順にピラミッドを積んで」
        │
        ▼
┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ 俯瞰カメラ    │──▶│ OWLv2 物体検出    │──▶│ LLM で段ごとの色  │──▶│ 真上から把持して配置 │
│ (普通のカメラ) │   │ + 位置・向きの推定 │   │ を計画 (Gemma 4)  │   │ 直前に手首カメラで   │
└──────────────┘   └──────────────────┘   └──────────────────┘   │ 位置合わせ (SO-101) │
        ▲                                                         └─────────┬──────────┘
        └──── 1 個置くごとに手首カメラで実測し、2.5 mm を超えてずれていれば掴み直して置き直す ◀─┘
```

## デモ動画

[デモ動画を見る](https://youtu.be/8T1swwySxgQ)（等倍速・日本語の解説ナレーション付き）。
このリポジトリには、動画を生成するコードとシミュレーション用のモデルを含めています。

## 結果のまとめ

| 項目 | 結果 |
| --- | --- |
| デモ（seed 3・Gemma 4 による計画） | **10/10 成功**、置き直し 0 回、本当の配置誤差 平均 **0.89 mm**・最大 **1.38 mm**、動画 160.2 秒 |
| 言語指示からの計画（23 種類の指示） | Gemma 4（改善後）が**開発用 13/15・テスト用 8/8** で指示どおり（3 回の平均。改善前 12.3/15・6.7/8、ルールベース 8/15・1/8） |
| 頑健性（初期配置10通りに、5種類の指示と既定の指示を割り当て） | **10/10 が指示どおりに完成**、置き直し 0 回、本当の誤差は最大 1.72 mm（連続実行で見つけた 3 種類の不具合を直した後） |
| 較正誤差への強さ | 俯瞰カメラに 4 mm / 1° の較正誤差があっても、手首カメラで **10/10**（俯瞰カメラのみでは 2/10） |

詳細な評価条件と結果は[実装・評価の記録](docs/experiments.md)を参照してください。上記は変更前に記録した実験結果で、今回の修正版では再測定していません。

## 動作環境

- Ubuntu 22.04 / 24.04（WSL2 でも動作確認済み）
- NVIDIA GPU（RTX 3090で確認。現在の起動設定はGPU向けです）
- [uv](https://docs.astral.sh/uv/)
- Python 3.10〜3.12（uv が自動で用意します）

## セットアップ

```bash
git clone <このリポジトリ>
cd <リポジトリ>
uv sync          # Genesis, PyTorch (CUDA 12.8), transformers, ollama などを導入
```

### pre-commit（Ruff など）

コミット前にRuffの検査・自動修正・フォーマットと、末尾空白や設定ファイルの検証を実行します。

```bash
uv sync --dev
uv run pre-commit install
uv run pre-commit run --all-files
```

Markdownの文法も同じフックで検査します。

初回実行時に、OWLv2 の重み（約 600 MB）が Hugging Face から自動でダウンロードされます。さらに、SO-101 の衝突形状の凸分解に 1 分ほどかかります（結果はキャッシュされます）。

### ローカル LLM（Ollama + Gemma 4）

タスク計画には、既定で手元の GPU で動くローカル LLM を使います。API キーは不要で、オフラインで動きます。

Ollamaは[公式インストール手順](https://ollama.com/download)から導入してください。スクリプトを使う場合は、内容を確認してから実行します。

```bash
curl -fsSLo ollama-install.sh https://ollama.com/install.sh
less ollama-install.sh
sh ollama-install.sh
rm ollama-install.sh
ollama pull gemma4:e4b  # 既定のモデル。精度を上げたい場合は gemma4:12b など
```

**WSL2 での注意:** WSL2 の Ollama は、同じ GPU を CUDA と Vulkan の両方で検出し、Vulkan（WSL の dzn は試験的な実装）を選ぶことがあります。その場合、Gemma 4 の読み込みで `llama-server process has terminated: signal: aborted` となって失敗します。Vulkan を無効にして CUDA を使わせてください。

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_VULKAN=0"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

RTX 3090 では、`gemma4:e4b` による計画 1 回にかかる時間は約 5 秒です。

別のモデルを使う場合は、環境変数 `OLLAMA_MODEL=gemma4:12b` で指定します。Ollama のサーバーが動いていない場合、モデルを取得していない場合、ブロックの個数と矛盾する計画が返った場合は、自動的にルールベースの計画に切り替わります。どの方式で計画したかは、ログと動画に表示されます。

### Claude API（任意）

`--planner claude` を指定するとClaude APIを使います。APIキーが必要です（`export ANTHROPIC_API_KEY=...`）。
`--planner auto`も、Ollamaが失敗した場合にClaude APIを使います。
Claude利用時は、指示文と色・個数（配置生成前の仮の個数を含む）がAnthropicへ送信され、API利用料が発生します。
既定の`ollama`はローカルサーバーを使います。ただし、`OLLAMA_HOST`を変更している場合は、その接続先へ入力を送信します。

## 使い方

```bash
# デモ動画を生成（outputs/demo.mp4, demo_detections.png, demo.json）
uv run python main.py

# 指示や配置を変える
uv run python main.py --instruction "一番上は赤、その下は橙、緑、青の順にして" --seed 11

# Genesis のビューアとカメラ映像のウィンドウを開き、等倍速で動きを見ながら実行（WSL2 では WSLg のウィンドウ）
uv run python main.py --viewer
SO101_VIEWER=1 uv run python main.py      # 環境変数で常にビューアを開く

# 計画方式を切り替える（ollama / claude / auto / rules）
uv run python main.py --planner rules

# カメラの構成を変える: 深度カメラを使う / 手首カメラを使わない / 較正誤差を模擬する
uv run python main.py --sensor rgbd
uv run python main.py --no-wrist --calib-error-mm 4 --calib-error-deg 1

# 評価実験 1: カメラ構成と較正誤差の比較（結果は outputs/compare/results.md）
uv run python scripts/compare_cameras.py

# 評価実験 2: 言語指示からの計画を LLM とルールベースで比べる（結果は outputs/planner/、数分）
uv run python scripts/eval_planner.py --repeat 3

# 評価実験 3: 初期配置と指示を変えて通しで実行する（結果は outputs/robust/results.md、1 回 約 17 分）
uv run python scripts/eval_robustness.py
uv run python scripts/eval_robustness.py --force  # 保存済みの結果も取り直す

# AI を使わず、正解位置でピラミッドを積む（制御のデバッグ用）
uv run python scripts/debug_pick.py --viewer --video outputs/debug_pyramid.mp4
```

ビューアを開いているときは、Genesis のビューアに加えて「cameras: overhead (AI) | wrist」というウィンドウが開きます。俯瞰カメラ（直近の AI 検出結果付き）、手首カメラ、全体の固定カメラのライブ映像が並びます。

動画のメインは、カメラマン役の SO-101 が撮った映像です。右側に、全体の固定カメラ（2 台のロボットが映る）、俯瞰カメラ（AI 検出）、手首カメラを小さく並べ、左上に今のカメラの演出名を表示します。

主なオプション:

| オプション | 説明 | 既定値 |
| --- | --- | --- |
| `--instruction` | 自然言語の指示（日本語・英語） | 青を土台に、緑・橙・赤の順 |
| `--seed` | ブロック初期配置の乱数シード | 3 |
| `--aligned` | ブロックをロボットの方向に揃えて置く（既定はランダムな向き） | off |
| `--fixed-blocks` | 指示によらず 9 色 × 3 個の固定のブロックを置く | off |
| `--planner` | 計画方式：`ollama`（ローカル LLM）/ `claude` / `auto`（ollama → claude の順に試す）/ `rules` | `ollama` |
| `--sensor` | 俯瞰カメラ：`rgb`（普通のカメラ）/ `rgbd`（深度カメラ） | `rgb` |
| `--no-wrist` | 手首カメラによる位置合わせ・確認を使わない | off |
| `--no-camera-robot` | カメラマン役の SO-101 を使わず、固定カメラ 1 台で動画を撮る | off |
| `--no-narration` | 解説ナレーション（音声）を入れない | off |
| `--calib-error-mm` / `--calib-error-deg` | 俯瞰カメラの較正誤差（並進 / 回転）を模擬する | 0 |
| `--wrist-error-mm` / `--wrist-error-deg` | 手首カメラの取付（ハンドアイ）誤差を模擬する | 0 |
| `--max-retries` | 1 個あたりの掴み直しの上限回数 | 2 |
| `--speed` | アームの動作速度の倍率（動画は常に等倍速で記録） | 1.0 |
| `--viewer` | ビューアとカメラ映像のウィンドウを開く（実時間で進行） | off |
| `--out` | 出力動画のパス | `outputs/demo.mp4` |

## 構成

```
main.py                    パイプライン全体（認識 → 計画 → 配置 → 実測・置き直し → 動画出力）
so101_sorter/
  env.py                   Genesis シーン（台に載せた SO-101・ブロック 27 個・ピラミッドの台座・俯瞰/手首カメラ）
  perception.py            OWLv2 によるブロック検出、画像からの位置・向きの推定（RGB / RGB-D）、配置後の実測
  planner.py               ローカル LLM（Ollama）/ Claude / ルールベースによる 1 個ずつの色の計画
  controller.py            IK による把持・配置（真上からの把持、指の向き合わせ、そっと置く動作）
  narration.py             解説ナレーション（ローカルの日本語音声合成で読み上げ、動画に音声として合成）
  cinema.py                カメラマン役の SO-101 によるカメラワーク（ジンバル付きカメラ、演出の切り替え）
  video.py                 字幕・カメラ映像のワイプ・タイトルカード付きの動画出力
scripts/debug_pick.py      正解位置を使ったピラミッド積みのデバッグ
scripts/compare_cameras.py カメラ構成と較正誤差の比較実験（評価実験 1）
scripts/eval_planner.py    言語指示からの計画の評価（23 種類の指示、LLM とルールベースの比較。評価実験 2）
scripts/eval_robustness.py 初期配置と指示を変えた通しの連続実行（評価実験 3）
tools/make_pad_mjcf.py     手首カメラ付き SO-101 の MJCF に指パッドを追加した so101_pads.xml を生成
tools/camera_monitor.py    カメラ映像のウィンドウ（main.py が別プロセスで起動）
assets/so101/              SO-101 の MJCF / URDF / STL（TheRobotStudio/SO-ARM100, Apache-2.0）
```

## 実装・評価の詳細

認識・制御の工夫、実験条件、結果表、失敗の原因と対策は[実装・評価の記録](docs/experiments.md)にまとめています。

## 限界と今後の課題

- **実機での検証がまだ:** 結果はすべてシミュレーションです。台への固定・指パッド・普通の USB カメラ 2 台（俯瞰 + 手首）という構成は実機でも再現できるように選びましたが、照明の変化・カメラのノイズ・サーボのバックラッシュは模擬していません。実機の SO-101 を LeRobot 経由で動かして確かめるのが次の課題です。
- **曖昧な指示:** 「土台は暖色、上の段は寒色」の「上の段」を一番上の段だけと解釈するなど、LLM が人の意図と違う読み方をすることがあります。計画を実行前に言葉と図で示して確認を取る、Gemma 4 に画像を見せて完成形を確かめさせる、などで補えると考えています。
- **形は固定:** 積む形は 4・3・2・1 段のピラミッドに限っています。LLM に積む位置の座標列を出力させれば、壁・階段・アーチなど形そのものを言葉で指定できます。
- **把持は学習していない:** 把持は IK による決まった動作です。予備実験として、手首カメラの画像から最後の把持だけを模倣学習（LeRobot の ACT、IK によるデモで学習）させました。単独の把持テストでは 24/30 回成功しましたが、ピラミッドを積む本番では 10 回中 4 回しか成功せず（残りは IK に切り替えて完成）、動画も 194 秒に延びたため、提出版には入れていません。作業途中の場面（積みかけのピラミッドが写る画像）をデモに含めていなかったことが原因と考えています。

## 動画について

- 長さは `outputs/demo.json` の `video_seconds` で確認できます（既定設定で約 160 秒。掴み直しが多いと長くなります）。
- 出力は等倍速です（倍速にはしていません）。

## 開発での AI の利用

コードの作成・デバッグ・文章の推敲に、Claude Code（Anthropic）を使いました。タスクの設計、実験の方針、結果の確認と判断は自分で行っています。

## クレジット・ライセンス

- このリポジトリのコード: [MIT License](LICENSE)（`assets/so101/` 以下は除く）
- SO-101 モデル: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)（Apache License 2.0、[assets/so101/LICENSE](assets/so101/LICENSE)）。`so101_pads.xml` は、これに指パッドを追加した改変版です。
- OWLv2: [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble)（Apache License 2.0）
- Gemma 4: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)（[Ollama](https://ollama.com/library/gemma4) 経由で利用）
- シミュレータ: [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)（Apache License 2.0）
- 音声合成: [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)（Open JTalk ベース、ローカルで動作）
