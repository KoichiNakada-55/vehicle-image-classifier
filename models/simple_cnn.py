"""
シンプルなCNNモデルの定義
CIFAR-10の乗り物画像（32x32ピクセル、カラー）を4クラスに分類します。
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    シンプルな畳み込みニューラルネットワーク (CNN)

    構成:
    - 畳み込み層 x 3（特徴を抽出する層）
    - プーリング層 x 3（画像サイズを縮小する層）
    - 全結合層 x 2（分類を行う層）

    入力: 32x32のカラー画像 (3チャンネル)
    出力: 4クラス (airplane, automobile, ship, truck)
    """

    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        # 畳み込み層1: 入力3チャンネル(RGB) → 32個のフィルタ
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 → 16x16

        # 畳み込み層2: 32チャンネル → 64個のフィルタ
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 → 8x8

        # 畳み込み層3: 64チャンネル → 128個のフィルタ
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 → 4x4

        # 全結合層: 分類を行う
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 過学習を防ぐ
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """順伝播: 画像を入力して、クラスの予測値を出力する"""
        # 畳み込み + 活性化 + プーリング を3回繰り返す
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # 2次元の特徴マップを1次元に変換
        x = x.view(x.size(0), -1)

        # 全結合層で分類
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
