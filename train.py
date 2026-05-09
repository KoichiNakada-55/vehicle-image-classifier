"""
train.py - CIFAR-10 乗り物画像分類モデルの学習スクリプト

CIFAR-10から乗り物4クラス (airplane, automobile, ship, truck) を抽出し、
シンプルなCNNモデルを学習させます。
学習済みモデルは saved_models/vehicle_classifier.pth に保存されます。
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.simple_cnn import SimpleCNN

# ===== 設定 =====
# 対象クラス（CIFAR-10のクラス番号）
# CIFAR-10: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
#           5=dog, 6=frog, 7=horse, 8=ship, 9=truck
TARGET_CLASSES = {0: 0, 1: 1, 8: 2, 9: 3}  # {元のラベル: 新しいラベル}
CLASS_NAMES = ["airplane", "automobile", "ship", "truck"]

# ハイパーパラメータ（学習の設定）
BATCH_SIZE = 64       # 一度に処理する画像の数
NUM_EPOCHS = 15       # データ全体を何回繰り返し学習するか
LEARNING_RATE = 0.001 # 学習率（モデルの更新の大きさ）
SAVE_PATH = "saved_models/vehicle_classifier.pth"


def get_vehicle_indices(dataset):
    """データセットから乗り物クラスのインデックスを取得する"""
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in TARGET_CLASSES:
            indices.append(i)
    return indices


def remap_labels(dataset, indices):
    """元のラベルを新しいラベル（0〜3）に変換する"""
    remapped = []
    for idx in indices:
        _, original_label = dataset[idx]
        new_label = TARGET_CLASSES[original_label]
        remapped.append(new_label)
    return remapped


class VehicleDataset(torch.utils.data.Dataset):
    """乗り物クラスだけを抽出したデータセット"""

    def __init__(self, base_dataset, indices, new_labels):
        self.base_dataset = base_dataset
        self.indices = indices
        self.new_labels = new_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[self.indices[idx]]
        label = self.new_labels[idx]
        return image, label


def main():
    # デバイスの設定（GPUが使えればGPU、なければCPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ===== データの前処理 =====
    # 画像をテンソルに変換し、正規化する
    transform = transforms.Compose([
        transforms.ToTensor(),  # 画像をPyTorchのテンソルに変換 (0〜1の範囲)
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10の平均値
            std=(0.2470, 0.2435, 0.2616)    # CIFAR-10の標準偏差
        )
    ])

    # ===== CIFAR-10データセットのダウンロードと読み込み =====
    print("CIFAR-10データセットを準備中...")
    full_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # ===== 乗り物クラスだけを抽出 =====
    print("乗り物クラスを抽出中...")
    train_indices = get_vehicle_indices(full_train)
    train_labels = remap_labels(full_train, train_indices)
    train_dataset = VehicleDataset(full_train, train_indices, train_labels)

    print(f"学習データ数: {len(train_dataset)}")

    # データローダーの作成（バッチ処理用）
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # ===== モデル・損失関数・最適化関数の準備 =====
    model = SimpleCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()  # 分類問題でよく使われる損失関数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam最適化

    print(f"\n学習を開始します（エポック数: {NUM_EPOCHS}）")
    print("=" * 60)

    # ===== 学習ループ =====
    for epoch in range(NUM_EPOCHS):
        model.train()  # 学習モードに設定
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # データをデバイスに転送
            images, labels = images.to(device), labels.to(device)

            # 勾配をリセット
            optimizer.zero_grad()

            # 順伝播: モデルで予測
            outputs = model(images)

            # 損失を計算
            loss = criterion(outputs, labels)

            # 逆伝播: 勾配を計算
            loss.backward()

            # パラメータを更新
            optimizer.step()

            # 統計を更新
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # エポックごとの結果を表示
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        print(
            f"Epoch [{epoch + 1:2d}/{NUM_EPOCHS}] "
            f"Loss: {epoch_loss:.4f}  "
            f"Accuracy: {epoch_acc:.2f}%"
        )

    print("=" * 60)
    print("学習が完了しました！")

    # ===== モデルの保存 =====
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"モデルを保存しました: {SAVE_PATH}")


if __name__ == "__main__":
    main()
