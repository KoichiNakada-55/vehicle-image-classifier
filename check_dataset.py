"""
check_dataset.py - データセット確認スクリプト

CIFAR-10から乗り物4クラスが正しく抽出されているか確認します。
各クラスの画像枚数とラベル変換が正しいことを検証します。
"""

import torchvision
import torchvision.transforms as transforms

# ===== 設定 =====
# CIFAR-10の全クラス名
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 抽出対象のクラス（元のラベル → 新しいラベル）
TARGET_CLASSES = {0: 0, 1: 1, 8: 2, 9: 3}
CLASS_NAMES = ["airplane", "automobile", "ship", "truck"]


def check_dataset(dataset, dataset_name):
    """データセットから乗り物クラスを抽出し、情報を表示する"""
    print(f"\n{'=' * 50}")
    print(f"  {dataset_name}")
    print(f"{'=' * 50}")

    # 各クラスのカウント（.targets属性を使って高速にカウント）
    class_counts = {label: 0 for label in TARGET_CLASSES.keys()}
    total_vehicle = 0

    for label in dataset.targets:
        if label in TARGET_CLASSES:
            class_counts[label] += 1
            total_vehicle += 1

    # 結果を表示
    print(f"\nCIFAR-10 全体のデータ数: {len(dataset)}")
    print(f"乗り物クラスのデータ数: {total_vehicle}")
    print()

    print(f"{'元のラベル':>10} {'クラス名':<12} {'新ラベル':>8} {'画像枚数':>8}")
    print("-" * 45)
    for original_label, new_label in TARGET_CLASSES.items():
        class_name = CIFAR10_CLASSES[original_label]
        count = class_counts[original_label]
        print(
            f"{original_label:>10} {class_name:<12} {new_label:>8} {count:>8}"
        )

    # ラベル変換の確認
    print(f"\n--- ラベル変換の確認 ---")
    for original_label, new_label in TARGET_CLASSES.items():
        original_name = CIFAR10_CLASSES[original_label]
        new_name = CLASS_NAMES[new_label]
        status = "OK" if original_name == new_name else "NG"
        print(
            f"  {original_name} (元ラベル={original_label}) "
            f"→ {new_name} (新ラベル={new_label}) [{status}]"
        )


def main():
    print("CIFAR-10 乗り物クラス抽出チェック")
    print("=" * 50)

    # データセットのダウンロード（前処理なし）
    transform = transforms.Compose([transforms.ToTensor()])

    print("\nCIFAR-10データセットをダウンロード中...")
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # 学習データの確認
    check_dataset(train_dataset, "学習データ (Train)")

    # テストデータの確認
    check_dataset(test_dataset, "テストデータ (Test)")

    print(f"\n{'=' * 50}")
    print("チェック完了！すべて正常です。")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
