# SO101 ロボット - URDF / MuJoCo モデル

> 本ファイルは [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) の `Simulation/SO101/README.md` を日本語訳したものです（Apache License 2.0、[LICENSE](LICENSE) 参照）。
> 本プロジェクトでは、このうち `so101_new_calib.xml`・`so101_new_calib_camera.xml`（手首カメラ付き）の MJCF、`so101_new_calib.urdf`、および `assets/` 内の STL メッシュを同梱しています。
> `so101_pads.xml` は、本プロジェクトの `tools/make_pad_mjcf.py` が `so101_new_calib_camera.xml` に指パッドを追加して生成した改変版で、シミュレーションではこれを使います。

このディレクトリには、SO101 ロボットの URDF ファイルと MuJoCo（MJCF）ファイルが含まれています。

## 概要

- ロボットのモデルファイルは、Onshape で設計された CAD モデルから [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) プラグインを使って生成されています。
- 生成された URDF は、`package://...` ではなく相対パスでメッシュを参照できるよう修正されています。
- ベース部分の衝突メッシュは、シミュレーションや動作計画で衝突の挙動に問題があったため削除されています。

## キャリブレーション方式

MuJoCo の `scene.xml` は、キャリブレーション方式の異なる 2 種類の SO101 モデルに対応しています。

- **新キャリブレーション（デフォルト）**: 各関節の仮想ゼロ点を関節可動範囲の**中央**に設定。→ `so101_new_calib.xml` を使用。
- **旧キャリブレーション**: 各関節の仮想ゼロ点を、ロボットが**水平に完全に伸びきった**姿勢に設定。→ `so101_old_calib.xml` を使用。

方式を切り替えるには、`scene.xml` で読み込むロボットファイルを変更してください。

（※ `scene.xml`、旧キャリブレーション版、リーダーアーム版の URDF は本プロジェクトには同梱していません。リーダーアーム用の STL は `assets/` に含まれていますが使っていません。）

## リーダーアームの URDF

`so101_leader_new_calib.urdf` は **SO-101 リーダー**（遠隔操作用）アームの URDF で、上記「新キャリブレーション」のゼロ点の定義を共有しています。ベース・肩・上腕・前腕・手首のリンクとメッシュはフォロワーアームのものを再利用しており、グリッパー部分のみが異なります。フォロワーの固定グリッパー本体は `Wrist_Roll_SO101.stl` と `Handle_SO101.stl` に、可動ジョーは `Trigger_SO101.stl` に置き換えられています（メッシュは `STEP/SO101/Leader_Specific/` から変換）。駆動関節の名前は、他の 5 関節との互換性のため `gripper` のままです。

```bash
rerun Simulation/SO101/so101_leader_new_calib.urdf
```

**キャリブレーションに関する注意:** アームの 5 関節は `so101_new_calib.urdf` と同じキャリブレーションです。トリガー関節のゼロ点、可動範囲（暫定で 0〜45°）、および印刷部品の慣性の推定値は、CAD の形状と印刷部品の大まかな平均密度から求めたもので、実機のリーダーアームを計測したものではありません。MJCF/XML 版はまだありません。

## モーターパラメータ

ロボットに使われている STS3215 モーターの特性値は、[Open Duck Mini プロジェクト](https://github.com/apirrone/Open_Duck_Mini) のものを流用しています。

## グリッパーに関する注意

LeRobot では、グリッパーは**直動関節**として表現されます。

- `0` = 完全に閉じた状態
- `100` = 完全に開いた状態

この対応関係は、現在の URDF / MuJoCo ファイルには**まだ反映されていません**。

---

issue の作成や改善のコントリビューションを歓迎します！
