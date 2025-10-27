# scripts/02_train_autoencoder.py
"""
Train a convolutional autoencoder on MNIST normal digits (GPU-optimized).

Inputs:
  - data/train_normal.pt
Outputs:
  - models/autoencoder.pth
  - results/training_loss_plot.png

Examples:
  # auto-select CUDA if available (default)
  python scripts/02_train_autoencoder.py

  # force CUDA (error if not available)
  python scripts/02_train_autoencoder.py --device cuda

  # CPU only
  python scripts/02_train_autoencoder.py --device cpu

  # disable AMP
  python scripts/02_train_autoencoder.py --no-amp
"""

import argparse
import os
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# make project root importable
sys.path.append(os.path.abspath("."))

from models.autoencoder import Autoencoder  # noqa: E402

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train Autoencoder on normal MNIST digits (GPU-ready).")
    p.add_argument("--data-dir", type=str, default="data", help="Directory with *.pt files from step 1.")
    p.add_argument("--model-out", type=str, default="models/autoencoder.pth", help="Where to save the model weights.")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam).")
    p.add_argument("--latent-dim", type=int, default=64, help="Autoencoder latent dimension.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    p.add_argument("--seed", type=int, default=123, help="Random seed.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                   help="Device selection: 'auto' picks cuda if available.")
    p.add_argument("--amp", dest="amp", action="store_true", help="Enable automatic mixed precision (AMP).")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    p.set_defaults(amp=True)
    return p.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available. Check your PyTorch/CUDA install.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_train_dataset(data_dir: str) -> TensorDataset:
    path = os.path.join(data_dir, "train_normal.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find {path}. Run scripts/01_prepare_data.py first.")
    # Safe load (only tensors and basic containers)
    blob = torch.load(path, weights_only=True, map_location="cpu")
    imgs = blob["images"]  # [N,1,28,28] float32
    return TensorDataset(imgs)


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Device & performance knobs
    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # autotune convs for current shapes
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[AnomalyMap] Using CUDA: {gpu_name}")
    else:
        print("[AnomalyMap] Using CPU")

    # Data
    train_ds = load_train_dataset(args.data_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    print(f"[AnomalyMap] Loaded train set with {len(train_ds)} samples | batch={args.batch_size}")

    # Model, optimizer, loss
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction="mean")

    # New AMP API (PyTorch 2.0+)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # Training loop
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    step_losses, epoch_losses = [], []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # autocast context for CUDA, disabled otherwise
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                x_hat = model(x_batch)
                loss = criterion(x_hat, x_batch)

            # Correct AMP sequence
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            n_batches += 1
            step_losses.append(loss.item())

            if n_batches % 100 == 0:
                print(f"  [epoch {epoch:02d}] step {n_batches:04d} | loss {loss.item():.6f}")

        epoch_loss = running / max(n_batches, 1)
        epoch_losses.append(epoch_loss)
        print(f"[AnomalyMap] Epoch {epoch:02d} | avg loss {epoch_loss:.6f}")

    # Save weights
    torch.save(model.state_dict(), args.model_out)
    print(f"[AnomalyMap] Saved weights -> {args.model_out}")

    # Plot loss curve
    plt.figure(figsize=(7, 4.2))
    plt.plot(step_losses, linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    amp_tag = " + AMP" if (args.amp and device.type == "cuda") else ""
    plt.title(f"Autoencoder Training Loss ({device.type}{amp_tag})")
    plt.grid(True, alpha=0.3)
    out_plot = os.path.join("results", "training_loss_plot.png")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=160)
    print(f"[AnomalyMap] Saved plot -> {out_plot}")

    dt = time.time() - t0
    print(f"[AnomalyMap] Done. Time: {dt:.1f}s | Final epoch loss: {epoch_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
