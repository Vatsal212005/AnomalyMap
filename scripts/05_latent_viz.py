# scripts/05_latent_viz.py
"""
Latent space visualization with t-SNE for AnomalyMap.

Extracts encoder features for normal and anomaly test samples,
projects them to 2D using sklearn.manifold.TSNE, and plots a scatter.

Outputs:
  - results/latent_space_tsne.png
  - results/latent_embeddings.csv

Usage:
  python scripts/05_latent_viz.py --device cuda --max-per-split 2000 --perplexity 35 --tsne-lr auto --seed 42
"""

import argparse
import os
import sys
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# project imports
sys.path.append(os.path.abspath("."))
from models.autoencoder import Autoencoder  # noqa: E402

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.manifold import TSNE
import csv
import inspect


# ----------------------------- Args ----------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="t-SNE latent space visualization (encoder features).")
    p.add_argument("--data-dir", type=str, default="data", help="Where test_normal.pt / test_anomaly.pt live.")
    p.add_argument("--weights", type=str, default="models/autoencoder.pth", help="Trained AE weights.")
    p.add_argument("--latent-dim", type=int, default=64, help="Latent size (must match training).")
    p.add_argument("--batch-size", type=int, default=1024, help="Batch size for encoding.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device selection.")
    p.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP on CUDA.")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    p.set_defaults(amp=True)

    p.add_argument("--max-per-split", type=int, default=2000,
                   help="Subsample up to this many samples from each split (to keep TSNE fast).")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    # parse tsne-lr as string to allow 'auto'
    p.add_argument("--tsne-lr", type=str, default="auto", help="t-SNE learning rate ('auto' or float).")
    p.add_argument("--seed", type=int, default=123, help="Random seed.")
    return p.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------- Data ----------------------------- #
def load_split(path: str) -> TensorDataset:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run scripts/01_prepare_data.py first.")
    blob = torch.load(path, weights_only=True, map_location="cpu")
    imgs = blob["images"].float()
    labels = blob["labels"].long()
    return TensorDataset(imgs, labels)


def make_loader(ds: TensorDataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def subsample_indices(n: int, k: int, seed: int) -> List[int]:
    if k <= 0 or k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k, replace=False).tolist()


# ----------------------------- Encode ----------------------------- #
@torch.no_grad()
def encode_dataset(model: Autoencoder, loader: DataLoader, device: torch.device, use_amp: bool) -> torch.Tensor:
    model.eval()
    feats = []

    try:
        autocast_ctx = torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda"))
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))

    with autocast_ctx:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            z = model.encode(x)
            feats.append(z.detach().cpu())

    return torch.cat(feats, dim=0)


# ----------------------------- Main ----------------------------- #
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # convert tsne-lr to correct type
    tsne_lr: Union[str, float]
    tsne_lr = "auto" if args.tsne_lr.lower() == "auto" else float(args.tsne_lr)

    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        print(f"[AnomalyMap] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[AnomalyMap] Using CPU")

    # Load datasets
    norm_ds = load_split(os.path.join(args.data_dir, "test_normal.pt"))
    anom_ds = load_split(os.path.join(args.data_dir, "test_anomaly.pt"))

    norm_idx = subsample_indices(len(norm_ds), args.max_per_split, args.seed + 1)
    anom_idx = subsample_indices(len(anom_ds), args.max_per_split, args.seed + 2)

    norm_subset = torch.utils.data.Subset(norm_ds, norm_idx)
    anom_subset = torch.utils.data.Subset(anom_ds, anom_idx)

    norm_loader = make_loader(norm_subset, args.batch_size, args.num_workers, device)
    anom_loader = make_loader(anom_subset, args.batch_size, args.num_workers, device)
    print(f"[AnomalyMap] Encoding latents | normal={len(norm_subset)} | anomaly={len(anom_subset)}")

    # Model
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Missing weights: {args.weights}. Train with scripts/02_train_autoencoder.py first.")
    state = torch.load(args.weights, weights_only=True, map_location=device)
    model.load_state_dict(state)

    # Encode
    z_norm = encode_dataset(model, norm_loader, device, args.amp)
    z_anom = encode_dataset(model, anom_loader, device, args.amp)

    X = torch.cat([z_norm, z_anom], dim=0).numpy()
    y_split = np.array([0] * len(z_norm) + [1] * len(z_anom), dtype=np.int64)

    def collect_labels(subset: torch.utils.data.Subset) -> np.ndarray:
        return np.array([int(lbl) for _, lbl in subset], dtype=np.int64)

    y_digit = np.concatenate([collect_labels(norm_subset), collect_labels(anom_subset)], axis=0)

    # Detect correct sklearn TSNE parameter name
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": args.perplexity,
        "learning_rate": tsne_lr,
        "init": "pca",
        "random_state": args.seed,
        "verbose": 1,
    }

    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000

    print("[AnomalyMap] Running t-SNE (this may take a bit)...")
    tsne = TSNE(**tsne_kwargs)
    X_2d = tsne.fit_transform(X)

    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "latent_embeddings.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "label", "tsne_x", "tsne_y"])
        for i, (xy, s, lab) in enumerate(zip(X_2d, y_split, y_digit)):
            w.writerow([i, "normal" if s == 0 else "anomaly", int(lab), float(xy[0]), float(xy[1])])
    print(f"[AnomalyMap] Saved embeddings -> {csv_path}")

    # Plot
    plt.figure(figsize=(7.5, 6.2))
    mask_norm = (y_split == 0)
    mask_anom = (y_split == 1)
    plt.scatter(X_2d[mask_norm, 0], X_2d[mask_norm, 1], s=8, alpha=0.65, label="Normal")
    plt.scatter(X_2d[mask_anom, 0], X_2d[mask_anom, 1], s=8, alpha=0.65, label="Anomaly")
    plt.legend()
    plt.title("t-SNE of Encoder Latents (Normal vs Anomaly)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.25)
    out_plot = os.path.join("results", "latent_space_tsne.png")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=170)
    print(f"[AnomalyMap] Saved plot -> {out_plot}")
    print("[AnomalyMap] Done.")


if __name__ == "__main__":
    main()
