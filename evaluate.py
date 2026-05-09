"""
evaluate.py - 学習済みモデルの評価スクリプト

保存されたモデルを読み込み、CIFAR-10のテストデータで評価します。
全体の精度、クラス別の精度、混同行列を表示・保存します。
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUIなし環境でも画像保存できるようにする
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.simple_cnn import SimpleCNN

# ===== 設定 =====
TARGET_CLASSES = {0: 0, 1: 1, 8: 2, 9: 3}
CLASS_NAMES = ["airplane", "automobile", "ship", "truck"]
MODEL_PATH = "saved_models/vehicle_classifier.pth"


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


def main():
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ===== テストデータの準備 =====
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    print("テストデータを準備中...")
    full_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # 乗り物クラスだけを抽出
    test_indices = get_vehicle_indices(full_test)
    test_labels = remap_labels(full_test, test_indices)
    test_dataset = VehicleDataset(full_test, test_indices, test_labels)

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2
    )
    print(f"テストデータ数: {len(test_dataset)}")

    # ===== モデルの読み込み =====
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # 評価モードに設定
    print(f"モデルを読み込みました: {MODEL_PATH}")

    # ===== 評価 =====
    print("\n評価中...")
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 勾配計算を無効化（メモリ節約）
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ===== 全体の精度 =====
    overall_accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"\n{'=' * 50}")
    print(f"全体の精度 (Overall Accuracy): {overall_accuracy:.2f}%")
    print(f"{'=' * 50}")

    # ===== クラス別の精度 =====
    print("\nクラス別の精度:")
    print(f"{'クラス名':<15} {'正解数':>6} {'合計':>6} {'精度':>8}")
    print("-" * 40)
    for i, class_name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        class_correct = np.sum(all_preds[mask] == i)
        class_total = np.sum(mask)
        class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0
        print(f"{class_name:<15} {class_correct:>6} {class_total:>6} {class_acc:>7.2f}%")

    # ===== 混同行列の作成と保存 =====
    print("\n混同行列を作成中...")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix - Vehicle Classifier")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("混同行列を confusion_matrix.png に保存しました")


if __name__ == "__main__":
    main()
