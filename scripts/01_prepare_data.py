# scripts/01_prepare_data.py
"""
Prepare MNIST for AnomalyMap.

Downloads MNIST into ./data, normalizes to [0,1] float32,
splits into:
  - train_normal.pt   (digits in NORMAL_DIGITS from train split)
  - test_normal.pt    (digits in NORMAL_DIGITS from test split)
  - test_anomaly.pt   (digits in ANOMALY_DIGITS from test split)

Each .pt file is a dict with:
  {
    "images": FloatTensor [N, 1, 28, 28],
    "labels": LongTensor  [N]
  }

Usage (from project root):
  python scripts/01_prepare_data.py
  python scripts/01_prepare_data.py --normal 0 1 2 3 4 5 6 7 8 --anomaly 9
  python scripts/01_prepare_data.py --data-dir data --seed 42
"""

import argparse
import os
from typing import List, Tuple

import torch
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MNIST splits for AnomalyMap.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store raw and processed data.",
    )
    parser.add_argument(
        "--normal",
        type=int,
        nargs="+",
        default=list(range(0, 9)),  # 0–8 as normal
        help="Digit classes treated as NORMAL.",
    )
    parser.add_argument(
        "--anomaly",
        type=int,
        nargs="+",
        default=[9],  # 9 as anomaly
        help="Digit classes treated as ANOMALY.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed (for any potential shuffling/extensions).",
    )
    parser.add_argument(
        "--normalize-mean",
        type=float,
        default=0.0,
        help="Mean for normalization (applied after ToTensor scaling to [0,1]).",
    )
    parser.add_argument(
        "--normalize-std",
        type=float,
        default=1.0,
        help="Std for normalization (applied after ToTensor). Use 1.0 to keep [0,1].",
    )
    return parser.parse_args()


def load_mnist(data_dir: str, mean: float, std: float) -> Tuple[datasets.MNIST, datasets.MNIST]:
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),             # scales to [0,1], float32
        transforms.Normalize((mean,), (std,)),  # optional normalization
    ])
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_ds, test_ds


def split_by_labels(
    images: torch.Tensor, labels: torch.Tensor, keep_labels: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for k in keep_labels:
        mask |= (labels == k)
    return images[mask], labels[mask]


def as_tensor_batch(ds: datasets.MNIST) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an entire torchvision dataset to batched tensors:
      images: [N, 1, 28, 28] float32
      labels: [N] long
    """
    # Pre-allocate and fill for efficiency
    N = len(ds)
    imgs = torch.empty((N, 1, 28, 28), dtype=torch.float32)
    lbls = torch.empty((N,), dtype=torch.long)
    for i in range(N):
        x, y = ds[i]                   # x: [1,28,28] float32, y: int
        imgs[i] = x
        lbls[i] = y
    return imgs, lbls


def save_split(path: str, images: torch.Tensor, labels: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"images": images, "labels": labels}, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[AnomalyMap] Preparing MNIST in '{args.data_dir}'")
    print(f"  NORMAL digits : {args.normal}")
    print(f"  ANOMALY digits: {args.anomaly}")

    train_ds, test_ds = load_mnist(args.data_dir, args.normalize_mean, args.normalize_std)

    # Convert to batched tensors
    train_imgs, train_lbls = as_tensor_batch(train_ds)
    test_imgs, test_lbls = as_tensor_batch(test_ds)

    # Splits
    train_normal_imgs, train_normal_lbls = split_by_labels(train_imgs, train_lbls, args.normal)
    test_normal_imgs, test_normal_lbls = split_by_labels(test_imgs, test_lbls, args.normal)
    test_anom_imgs,  test_anom_lbls  = split_by_labels(test_imgs, test_lbls, args.anomaly)

    # Save
    train_path = os.path.join(args.data_dir, "train_normal.pt")
    test_norm_path = os.path.join(args.data_dir, "test_normal.pt")
    test_anom_path = os.path.join(args.data_dir, "test_anomaly.pt")

    save_split(train_path, train_normal_imgs, train_normal_lbls)
    save_split(test_norm_path, test_normal_imgs, test_normal_lbls)
    save_split(test_anom_path, test_anom_imgs, test_anom_lbls)

    # Report
    print("\n[AnomalyMap] Done.")
    print(f"  Saved: {train_path}       -> {len(train_normal_lbls)} samples")
    print(f"  Saved: {test_norm_path}    -> {len(test_normal_lbls)} samples")
    print(f"  Saved: {test_anom_path}    -> {len(test_anom_lbls)} samples")
    # Sanity check: shapes
    print(f"\nShapes:")
    print(f"  train_normal: {tuple(train_normal_imgs.shape)}")
    print(f"  test_normal : {tuple(test_normal_imgs.shape)}")
    print(f"  test_anomaly: {tuple(test_anom_imgs.shape)}")


if __name__ == "__main__":
    main()
