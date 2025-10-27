# scripts/04_visualize_heatmaps.py
"""
Visualize per-pixel anomaly maps (absolute reconstruction error).

Creates side-by-side panels for chosen samples:
    [ input | reconstruction | heatmap ]

Loads:
  - models/autoencoder.pth
  - data/test_anomaly.pt  (default split)
  - data/test_normal.pt   (if --split normal)

Outputs:
  - results/heatmaps/sample_<k>.png

Examples (from project root):
  # default: 8 random anomalies
  python scripts/04_visualize_heatmaps.py --device cuda

  # exact indices from anomaly split
  python scripts/04_visualize_heatmaps.py --indices 3 10 42

  # visualize normals instead
  python scripts/04_visualize_heatmaps.py --split normal --num 6

  # bigger figures and a different colormap
  python scripts/04_visualize_heatmaps.py --fig-scale 2.0 --cmap inferno
"""

import argparse
import os
import sys
import random
from typing import List, Tuple

import torch
import numpy as np

# project imports
sys.path.append(os.path.abspath("."))
from models.autoencoder import Autoencoder  # noqa: E402

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Generate anomaly heatmaps from AE reconstructions.")
    p.add_argument("--data-dir", type=str, default="data", help="Directory containing test_normal.pt / test_anomaly.pt")
    p.add_argument("--weights", type=str, default="models/autoencoder.pth", help="Trained AE weights")
    p.add_argument("--latent-dim", type=int, default=64, help="Latent size (must match training)")
    p.add_argument("--split", type=str, default="anomaly", choices=["anomaly", "normal"], help="Which test split to use")
    p.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit indices to visualize from the split")
    p.add_argument("--num", type=int, default=8, help="If no indices, pick this many random samples")
    p.add_argument("--seed", type=int, default=123, help="Random seed for sampling")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    p.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP on CUDA")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    p.set_defaults(amp=True)
    p.add_argument("--fig-scale", type=float, default=1.5, help="Scale factor for figure size")
    p.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap for heatmap")
    return p.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(data_dir: str, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    fname = "test_anomaly.pt" if split == "anomaly" else "test_normal.pt"
    path = os.path.join(data_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run scripts/01_prepare_data.py first.")
    # Safe load for tensors/containers to silence pickle warning
    blob = torch.load(path, weights_only=True, map_location="cpu")
    return blob["images"].float(), blob["labels"].long()


def choose_indices(total: int, indices: List[int] | None, k: int, seed: int) -> List[int]:
    if indices is not None and len(indices) > 0:
        # sanitize and clip to range
        return [int(max(0, min(total - 1, i))) for i in indices]
    rng = random.Random(seed)
    if k >= total:
        return list(range(total))
    return rng.sample(range(total), k)


def _to_numpy_img(x: torch.Tensor) -> np.ndarray:
    """
    x: Tensor [1,28,28] or [28,28] in [0,1]
    returns: np.ndarray [28,28] float32 in [0,1]
    """
    if x.dim() == 3:
        x = x[0]
    x = x.detach().cpu().float().clamp(0, 1)
    return x.numpy()


def _norm_heatmap(h: torch.Tensor) -> np.ndarray:
    """
    h: Tensor [1,28,28] absolute error
    Normalize per-image for visibility: h / (h.max() + eps)
    """
    h = h.detach().cpu().float()
    h = h / (h.max() + 1e-8)
    return _to_numpy_img(h)


def save_triptych(img: torch.Tensor, recon: torch.Tensor, heat: torch.Tensor, out_path: str,
                  fig_scale: float = 1.5, cmap: str = "magma"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img_np = _to_numpy_img(img)
    rec_np = _to_numpy_img(recon)
    h_np = _norm_heatmap(heat)

    fig_w, fig_h = 9.0 * fig_scale, 3.0 * fig_scale
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    titles = ["Input", "Reconstruction", "Anomaly Map"]

    # Input
    axes[0].imshow(img_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    # Recon
    axes[1].imshow(rec_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    # Heatmap
    im = axes[2].imshow(h_np, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[2].set_title(titles[2])
    axes[2].axis("off")

    # colorbar for heatmap
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Abs Error")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

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

    # Load data split
    imgs, labels = load_split(args.data_dir, args.split)  # imgs: [N,1,28,28] in [0,1]
    N = imgs.shape[0]
    sel_idx = choose_indices(N, args.indices, args.num, args.seed)
    print(f"[AnomalyMap] Split='{args.split}' | total={N} | visualizing {len(sel_idx)} samples: {sel_idx[:8]}{'...' if len(sel_idx)>8 else ''}")

    # Model
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Missing weights at {args.weights}. Train with scripts/02_train_autoencoder.py first.")
    # Load state dict safely (no pickle execution), remove invalid kwarg and map to device
    state = torch.load(args.weights, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # AMP autocast ctx (new API with fallback)
    try:
        autocast_ctx = torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda"))
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda"))

    out_dir = os.path.join("results", "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    # Generate figures
    with autocast_ctx:
        for k, i in enumerate(sel_idx, start=1):
            x = imgs[i:i+1].to(device, non_blocking=True)            # [1,1,28,28]
            x_hat = model(x).clamp(0.0, 1.0)                         # ensure displayable range
            diff = (x_hat - x).abs()                                 # [1,1,28,28]

            # Save figure
            out_path = os.path.join(out_dir, f"sample_{k}.png")
            save_triptych(x[0], x_hat[0], diff[0], out_path, fig_scale=args.fig_scale, cmap=args.cmap)

    print(f"[AnomalyMap] Saved {len(sel_idx)} heatmaps -> {out_dir}")


if __name__ == "__main__":
    main()
