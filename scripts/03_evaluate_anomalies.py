# scripts/03_evaluate_anomalies.py
"""
Evaluate reconstruction error on normal vs anomaly images.

Loads:
  - data/test_normal.pt   (dict: {"images": FloatTensor [N,1,28,28], "labels": LongTensor})
  - data/test_anomaly.pt  (same structure)
  - models/autoencoder.pth

Computes per-image MSE reconstruction error and saves:
  - results/reconstruction_errors.csv  (columns: split,index,label,mse)
  - results/error_histogram.png        (overlayed histograms)

Usage (from project root):
  python scripts/03_evaluate_anomalies.py
  # GPU & AMP:
  python scripts/03_evaluate_anomalies.py --device cuda
"""

import argparse
import csv
import os
import sys
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

# import model from project root
sys.path.append(os.path.abspath("."))
from models.autoencoder import Autoencoder  # noqa: E402

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate anomaly reconstruction errors.")
    p.add_argument("--data-dir", type=str, default="data", help="Directory with test .pt files.")
    p.add_argument("--weights", type=str, default="models/autoencoder.pth", help="Path to trained AE weights.")
    p.add_argument("--batch-size", type=int, default=512, help="Eval batch size.")
    p.add_argument("--latent-dim", type=int, default=64, help="Latent dim (must match training).")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Select device.")
    p.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP (CUDA only).")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    p.set_defaults(amp=True)
    return p.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(path: str) -> TensorDataset:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}. Run scripts/01_prepare_data.py first.")
    blob = torch.load(path)  # dict: images [N,1,28,28], labels [N]
    imgs = blob["images"].float()
    labels = blob["labels"].long()
    return TensorDataset(imgs, labels)


@torch.no_grad()
def batched_mse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """
    Returns per-sample MSE tensor of shape [N].
    """
    model.eval()
    all_mse = []
    # Prefer new torch.amp API, fallback to cuda.amp for older versions
    try:
        autocast_ctx = torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda"))
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))

    with autocast_ctx:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            x_hat = model(x)
            # per-image MSE over all pixels
            mse = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3))  # [B]
            all_mse.append(mse.detach().cpu())

    return torch.cat(all_mse, dim=0)  # [N]


def to_loader(ds: TensorDataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def save_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "index", "label", "mse"])
        writer.writerows(rows)


def plot_hist(normal_mse: torch.Tensor, anomaly_mse: torch.Tensor, out_path: str):
    plt.figure(figsize=(7.2, 4.2))
    plt.hist(normal_mse.numpy(), bins=50, alpha=0.6, density=True, label="Normal")
    plt.hist(anomaly_mse.numpy(), bins=50, alpha=0.6, density=True, label="Anomaly")
    plt.xlabel("Per-image reconstruction MSE")
    plt.ylabel("Density")
    plt.title("Reconstruction Error Distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def main():
    args = parse_args()

    # Device + perf knobs
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

    # Load data
    test_norm_ds = load_split(os.path.join(args.data_dir, "test_normal.pt"))
    test_anom_ds = load_split(os.path.join(args.data_dir, "test_anomaly.pt"))
    print(f"[AnomalyMap] test_normal: {len(test_norm_ds)} | test_anomaly: {len(test_anom_ds)}")

    test_norm_loader = to_loader(test_norm_ds, args.batch_size, args.num_workers, device)
    test_anom_loader = to_loader(test_anom_ds, args.batch_size, args.num_workers, device)

    # Model
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Missing weights: {args.weights}. Train first with scripts/02_train_autoencoder.py")
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(sd)
    print(f"[AnomalyMap] Loaded weights from {args.weights}")

    # Evaluate per-sample MSE
    normal_mse = batched_mse(model, test_norm_loader, device, args.amp)      # [N_norm]
    anomaly_mse = batched_mse(model, test_anom_loader, device, args.amp)     # [N_anom]

    # Prepare CSV rows
    rows = []
    # normals
    for idx, (_, lbl) in enumerate(test_norm_ds):
        rows.append(["normal", idx, int(lbl), float(normal_mse[idx].item())])
    # anomalies
    for idx, (_, lbl) in enumerate(test_anom_ds):
        rows.append(["anomaly", idx, int(lbl), float(anomaly_mse[idx].item())])

    # Save CSV
    csv_path = os.path.join("results", "reconstruction_errors.csv")
    save_csv(csv_path, rows)
    print(f"[AnomalyMap] Saved per-image errors -> {csv_path}")

    # Plot histogram
    hist_path = os.path.join("results", "error_histogram.png")
    plot_hist(normal_mse, anomaly_mse, hist_path)
    print(f"[AnomalyMap] Saved error histogram -> {hist_path}")

    # Quick summary
    print("[AnomalyMap] Summary (mean MSE):")
    print(f"  normal  : {normal_mse.mean().item():.6f}")
    print(f"  anomaly : {anomaly_mse.mean().item():.6f}")


if __name__ == "__main__":
    main()
