# 🚗 CIFAR-10 乗り物画像分類器 (Vehicle Image Classifier)

CIFAR-10データセットに含まれる乗り物クラスだけを使い、画像分類モデルを作成するプロジェクトです。

## 目的

CIFAR-10の乗り物4クラスを分類するシンプルなCNNモデルを学習・評価・推論できるようにすることです。初心者でも理解しやすく、確実に動くことを優先しています。

## 使用するデータセット

**CIFAR-10** — 10クラス・60,000枚のカラー画像（32x32ピクセル）のデータセットです。  
このプロジェクトでは、以下の **乗り物4クラスだけ** を抽出して使用します。

| 元のラベル | クラス名     | 新しいラベル |
|:----------:|:------------:|:------------:|
| 0          | airplane     | 0            |
| 1          | automobile   | 1            |
| 8          | ship         | 2            |
| 9          | truck        | 3            |

## ディレクトリ構成

```
vehicle-image-classifier/
├── README.md              # このファイル
├── requirements.txt       # 必要なライブラリ一覧
├── train.py               # 学習スクリプト
├── evaluate.py            # 評価スクリプト
├── predict.py             # 推論スクリプト（画像1枚を分類）
├── check_dataset.py       # データセット確認スクリプト
├── .gitignore             # Git管理から除外するファイル
├── models/
│   ├── __init__.py
│   └── simple_cnn.py      # CNNモデルの定義
├── saved_models/          # 学習済みモデルの保存先（自動作成）
└── sample_images/         # 推論用のサンプル画像
```

## セットアップ方法

### 1. Python環境を用意する

Python 3.8以上が必要です。

### 2. 仮想環境を作成する（推奨）

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. 必要なライブラリをインストールする

```bash
pip install -r requirements.txt
```

## 使い方

### データセットの確認

CIFAR-10から4クラスが正しく抽出されるか確認します。

```bash
python check_dataset.py
```

### モデルの学習

CIFAR-10をダウンロードし、CNNモデルを学習します。学習済みモデルは `saved_models/vehicle_classifier.pth` に保存されます。

```bash
python train.py
```

学習中は、各エポックの損失 (Loss) と精度 (Accuracy) が表示されます。

### モデルの評価

学習済みモデルをテストデータで評価します。

```bash
python evaluate.py
```

以下の情報が表示されます：
- 全体の精度 (Overall Accuracy)
- クラス別の精度
- 混同行列（`confusion_matrix.png` として保存）

### 画像1枚で推論する

任意の画像ファイルを指定して、乗り物の種類を予測します。

```bash
python predict.py --image sample_images/example.jpg
```

予測されたクラス名と確信度が表示されます。

## 注意点

- 初回実行時にCIFAR-10データセット（約170MB）が自動ダウンロードされます
- GPUがなくてもCPUで学習できます（ただし時間がかかります）
- 学習済みモデル (`saved_models/`) やデータ (`data/`) は `.gitignore` で除外されています
- `predict.py` に渡す画像は、乗り物（飛行機・車・船・トラック）の画像を推奨します
- モデルはシンプルなCNNのため、精度は高くありません。学習の流れを理解するための教材としてお使いください
