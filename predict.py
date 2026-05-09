"""
predict.py - 任意の画像1枚を分類する推論スクリプト

使い方:
    python predict.py --image sample_images/example.jpg

学習済みモデルを使って、入力画像が何の乗り物かを予測します。
"""

import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image

from models.simple_cnn import SimpleCNN

# ===== 設定 =====
CLASS_NAMES = ["airplane", "automobile", "ship", "truck"]
MODEL_PATH = "saved_models/vehicle_classifier.pth"


def load_image(image_path):
    """
    画像ファイルを読み込み、モデル入力用に前処理する

    - 32x32ピクセルにリサイズ
    - テンソルに変換
    - 学習時と同じ正規化を適用
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),       # CIFAR-10と同じサイズにリサイズ
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # 画像を開いてRGBに変換（グレースケール画像にも対応）
    image = Image.open(image_path).convert("RGB")
    # 前処理を適用し、バッチ次元を追加 (1, 3, 32, 32)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def main():
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(
        description="乗り物画像を分類します"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="分類したい画像のパス（例: sample_images/example.jpg）"
    )
    args = parser.parse_args()

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== モデルの読み込み =====
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # ===== 画像の読み込みと推論 =====
    print(f"画像を読み込み中: {args.image}")
    image_tensor = load_image(args.image).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        # ソフトマックスで確率に変換
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # ===== 結果を表示 =====
    pred_class = CLASS_NAMES[predicted.item()]
    pred_confidence = confidence.item() * 100

    print(f"\n{'=' * 40}")
    print(f"予測結果: {pred_class}")
    print(f"確信度:   {pred_confidence:.2f}%")
    print(f"{'=' * 40}")

    # 全クラスの確率を表示
    print("\n各クラスの確率:")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = probabilities[0][i].item() * 100
        bar = "█" * int(prob / 2)
        print(f"  {class_name:<12} {prob:>6.2f}%  {bar}")


if __name__ == "__main__":
    main()
