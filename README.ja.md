![logo](./docs/images/logo.svg)

# PAMIQ VRChat

[English](./README.md) | **日本語**

PAMIQとVRChatを連携させるためのインターフェースライブラリ

## ✨ 機能

- **視覚入力**: `ImageSensor`でOBS仮想カメラからVRChatの画面をキャプチャ
- **マウス制御**: `MouseActuator`・`SmoothMouseActuator`で自然なマウス操作を実現
- **アバター操作**: `OscActuator`・`SmoothOscActuator`でOSC経由のアバター制御
- **スムーズな動作**: 加速度操作で滑らかな操作感を実現
- **PAMIQ連携**: [PAMIQ-Core](https://mlshukai.github.io/pamiq-core/)と組み合わせてAIエージェントを開発

## 📦 インストール

> \[!NOTE\]
> **Linux 🐧** をお使いの場合は、事前に依存ライブラリの[**inputtino**](https://github.com/games-on-whales/inputtino/tree/stable/bindings/python#installation)をインストールしておいてください。

```sh
# pipでインストール
pip install pamiq-vrchat

# ソースからインストール
git clone https://github.com/MLShukai/pamiq-vrchat.git
cd pamiq-vrchat
pip install .
```

## 🛠️ VRChat環境のセットアップ

### 前提条件

- デスクトップ環境を備えたLinuxまたはWindowsPC
- VRChatが動作するスペックのPC

### Steamのインストール

[Steam公式サイト](https://store.steampowered.com/about/)からダウンロードしてインストールしてください。

### **（🐧 Linuxユーザーのみ）** Protonを有効化

Steam → 設定 → 互換性から、`他のすべてのタイトルでSteam Playを有効化`をオンにしてください。

![steam_compatibility](./docs/images/steam_compatibility.png)

#### （オプション）Proton GEのインストール

LinuxでVRChat内のビデオプレイヤーを使いたい場合は、[Proton GE](https://github.com/GloriousEggroll/proton-ge-custom?tab=readme-ov-file#installation)をインストールすることをお勧めします。

インストール後は、Steam → 設定 → 互換性の`他のタイトルを実行する際に使用するツール:`から`GE-Proton`を選択してください。

### VRChatのインストール

[Steamストア](https://store.steampowered.com/app/438100/VRChat/)から**VRChat**をライブラリに追加してインストールしてください。

インストール完了後、VRChatを起動してアカウントにログインしてください。

### OBSのセットアップ

OBSのインストールと仮想カメラの設定方法は、[pamiq-ioのドキュメント](https://github.com/MLShukai/pamiq-io?tab=readme-ov-file#obs-virtual-camera)をご覧ください。

> \[!NOTE\]
> OBSのビデオ設定にある`出力（スケーリング）解像度`と`FPS値`は、`ImageSensor`クラスの出力に影響します。
> ![obs-video-setting](./docs/images/obs_video_setting.png)

OBSでVRChatのウィンドウをキャプチャして、仮想カメラを有効にしてください。

あらかじめ用意された[シーンコレクション](./obs_settings/)も使用できます。OBSの`シーンコレクション`タブ → `インポート`から読み込んで、チェックボックスがオンになっていることをご確認ください。

### OSCを有効化

1. 「ランチパッド」を開く（`Esc`キーを押す）
2. メインメニューに移動
3. 設定を開く（⚙️アイコンをクリック）
4. `すべての設定を検索`をクリックして「OSC」と入力し、Enterキーを押す
5. `OSC`ボタンをオンにする

![enable_osc](./docs/images/osc_enable.png)

## 🎮 サンプルプロジェクトの実行

VRChat環境をセットアップしたら、サンプルプロジェクトを実行できます。

PAMIQを実行してみましょう！

- **Linuxユーザーの場合**

  ```bash
  # サンプル実行
  ./run-sample.linux.sh
  ```

- **Windowsユーザーの場合**

  ```powershell
  # サンプル実行（PowerShellで）
  .\Run-Sample.Windows.ps1
  # 実行ポリシーエラーが出る場合
  powershell -noexit -ExecutionPolicy Bypass -File .\Run-Sample.Windows.ps1
  ```

これらのスクリプトは以下の処理を自動実行します：

- 依存関係の確認と自動インストール
- CUDA環境の確認
- VRChatとOBSの起動状態をチェック
- キーボード制御インターフェース（[**`pamiq-kbctl`**](https://mlshukai.github.io/pamiq-core/user-guide/console/#keyboard-shortcut-controller)）を起動
- 自律AIエージェントを起動

> \[!IMPORTANT\]
> **マウス制御について：** エージェントが起動すると、VRChat操作のためにマウスが制御されます。システムを一時停止したい場合は **`Alt+Shift+P`** を押してください。マウス制御を取り戻すための重要なショートカットです。

実装の詳細（アーキテクチャ、ハイパーパラメータ、学習手順など）は[`src/run_sample.py`](src/run_sample.py)をご覧ください。

## 🚀 使用例

### 画像取得

```python
from pamiq_vrchat.sensors import ImageSensor

# OBS仮想カメラに自動接続
sensor = ImageSensor()
# カメラインデックスを指定する場合
# sensor = ImageSensor(camera_index=0)
# （Windows限定）解像度を指定する場合
# sensor = ImageSensor(width=1920, height=1080)

# 画面をキャプチャ
frame = sensor.read()
# frameは(height, width, channels)のnumpy配列
```

### マウス制御

> \[!NOTE\]
> マウス制御を使用する際は、VRChatのウィンドウをアクティブにしておいてください。

```python
from pamiq_vrchat.actuators import MouseActuator, MouseButton, SmoothMouseActuator

# 基本的なマウス制御
mouse = MouseActuator()
# 水平方向100px/秒、垂直方向50px/秒でマウスを移動
mouse.operate({"move_velocity": (100.0, 50.0)})
# 左ボタンを押す
mouse.operate({"button_press": {MouseButton.LEFT: True}})
# 左ボタンを離す
mouse.operate({"button_press": {MouseButton.LEFT: False}})

# 滑らかなマウス制御（段階的な加速と自然なクリック感）
smooth_mouse = SmoothMouseActuator(
    delta_time=0.05,     # 更新間隔
    time_constant=0.2,   # 移動の平滑化
    press_delay=0.05,    # ボタン押下の遅延
    release_delay=0.1    # ボタン離しの遅延
)
smooth_mouse.operate({"move_velocity": (100.0, 50.0)})
```

### アバター制御（OSC）

```python
from pamiq_vrchat.actuators import OscActuator, OscAxes, OscButtons, SmoothOscActuator

# 基本的なOSC制御
osc = OscActuator()
# 前進する
osc.operate({"axes": {OscAxes.Vertical: 1.0}})
# ジャンプする
osc.operate({"buttons": {OscButtons.Jump: True}})
# 走って前進する
osc.operate({
    "axes": {OscAxes.Vertical: 1.0},
    "buttons": {OscButtons.Run: True}
})

# 滑らかなOSC制御（より自然な動き）
smooth_osc = SmoothOscActuator(
    delta_time=0.05,     # 更新間隔
    time_constant=0.2,   # 軸値の平滑化
    press_delay=0.05,    # ボタン押下の遅延
    release_delay=0.1    # ボタン離しの遅延
)
smooth_osc.operate({"axes": {OscAxes.Vertical: 0.5}})
```

## 🤝 貢献

開発環境のセットアップやプロジェクトへのコントリビューションについては、[CONTRIBUTING.md](CONTRIBUTING.md)をご覧ください。

## 📄 ライセンス

このプロジェクトはMITライセンスで公開されています。詳細は[LICENSE](LICENSE)ファイルをご確認ください。
