"""Train the conditional GAN baseline.

Non-saturating GAN loss + R1 gradient penalty (every --r1-every D steps) +
EMA generator. Saves periodic samples and checkpoints.

Run from project root:
    python src/train_gan.py --index data/processed/index.csv --img-size 128
"""

from __future__ import annotations
import argparse, copy, csv, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import Generator, Discriminator  # noqa: E402


class CardDataset(Dataset):
    def __init__(self, index_csv: Path, img_size: int,
                 rarity: str | None = None):
        with open(index_csv, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # Optional single-rarity filter; re-indexes class to 0 so the
        # conditional model still works (with n_classes=1).
        if rarity is not None:
            rows = [r for r in rows if r["rarity"] == rarity]
            if not rows:
                raise SystemExit(f"No images found for rarity={rarity!r}")
            for r in rows:
                r["rarity_idx"] = "0"
        self.items = [(r["path"], int(r["rarity_idx"])) for r in rows]
        self.n_classes = max(y for _, y in self.items) + 1
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


@torch.no_grad()
def ema_update(ema: torch.nn.Module, src: torch.nn.Module, decay: float = 0.999):
    for pe, p in zip(ema.parameters(), src.parameters()):
        pe.data.mul_(decay).add_(p.data, alpha=1 - decay)
    for be, b in zip(ema.buffers(), src.buffers()):
        be.data.copy_(b.data)


def r1_penalty(d_real: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(d_real.sum(), x_real,
                            create_graph=True, only_inputs=True)[0]
    return g.pow(2).reshape(g.size(0), -1).sum(1).mean()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=Path("data/processed/index.csv"))
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--rarity", type=str, default=None,
                   help="Train on a single rarity (folder name in index.csv). "
                        "Skips conditioning by collapsing to n_classes=1.")
    p.add_argument("--epochs", type=int, default=None,
                   help="If set, overrides --steps as ceil(epochs*len(ds)/batch).")
    p.add_argument("--resume", type=Path, default=None,
                   help="Load weights from a checkpoint (compatible-shapes only; "
                        "mismatched class-embedding layers are reinitialized).")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--r1-gamma", type=float, default=10.0)
    p.add_argument("--r1-every", type=int, default=16)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--samples", type=Path, default=Path("samples"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    # Per-rarity runs get their own checkpoint/sample subdirs so runs
    # don't clobber each other.
    if args.rarity:
        args.out = args.out / args.rarity
        args.samples = args.samples / args.rarity
    args.out.mkdir(parents=True, exist_ok=True)
    args.samples.mkdir(parents=True, exist_ok=True)

    ds = CardDataset(args.index, args.img_size, rarity=args.rarity)
    n_classes = ds.n_classes
    steps_per_epoch = max(1, len(ds) // args.batch)
    if args.epochs is not None:
        args.steps = args.epochs * steps_per_epoch
    print(f"Dataset: {len(ds)} images, {n_classes} class(es), "
          f"{steps_per_epoch} steps/epoch, target={args.steps} steps "
          f"(~{args.steps / steps_per_epoch:.1f} epochs), device={args.device}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True)

    dev = torch.device(args.device)
    G = Generator(args.z_dim, n_classes, args.ch, args.img_size).to(dev)
    D = Discriminator(n_classes, args.ch, args.img_size).to(dev)
    G_ema = copy.deepcopy(G).eval()
    for pp in G_ema.parameters():
        pp.requires_grad_(False)

    # Resume from a previous checkpoint (e.g. fine-tune base on one rarity).
    # Tensors whose shapes don't match (typically class embeddings when going
    # 14-class base -> 1-class rarity) are dropped and stay freshly initialized.
    if args.resume:
        rk = torch.load(args.resume, map_location=dev, weights_only=False)
        def _load_compat(model, sd):
            own = model.state_dict()
            keep = {k: v for k, v in sd.items()
                    if k in own and own[k].shape == v.shape}
            skipped = [k for k in sd if k not in keep]
            model.load_state_dict(keep, strict=False)
            return skipped
        sk_g = _load_compat(G, rk["G"])
        sk_d = _load_compat(D, rk["D"])
        _load_compat(G_ema, rk["G_ema"])
        print(f"Resumed from {args.resume} (base step={rk.get('step')}). "
              f"Reinitialized {len(set(sk_g) | set(sk_d))} mismatched tensors: "
              f"{sorted(set(sk_g) | set(sk_d))}")

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.99))

    # Fixed noise/labels for tracking sample progress over time.
    n_fixed = min(64, 8 * n_classes)
    fixed_z = torch.randn(n_fixed, args.z_dim, device=dev)
    fixed_y = torch.arange(n_fixed, device=dev) % n_classes

    step = 0
    t0 = time.time()
    while step < args.steps:
        for x_real, y_real in dl:
            x_real = x_real.to(dev, non_blocking=True)
            y_real = y_real.to(dev, non_blocking=True)
            bs = x_real.size(0)

            # ===== D step =====
            do_r1 = (step % args.r1_every == 0)
            if do_r1:
                x_real.requires_grad_(True)
            z = torch.randn(bs, args.z_dim, device=dev)
            y_fake = torch.randint(0, n_classes, (bs,), device=dev)
            with torch.no_grad():
                x_fake = G(z, y_fake)
            d_real = D(x_real, y_real)
            d_fake = D(x_fake, y_fake)
            loss_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
            if do_r1:
                loss_d = loss_d + (args.r1_gamma / 2) * r1_penalty(d_real, x_real)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ===== G step =====
            z = torch.randn(bs, args.z_dim, device=dev)
            y_fake = torch.randint(0, n_classes, (bs,), device=dev)
            x_fake = G(z, y_fake)
            d_fake = D(x_fake, y_fake)
            loss_g = F.softplus(-d_fake).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            ema_update(G_ema, G, decay=args.ema_decay)

            if step % 50 == 0:
                it_s = (step + 1) / max(time.time() - t0, 1e-6)
                print(f"step {step:>6}  d={loss_d.item():.3f}  "
                      f"g={loss_g.item():.3f}  ({it_s:.1f} it/s)")

            if step % args.sample_every == 0:
                G_ema.eval()
                with torch.no_grad():
                    grid = G_ema(fixed_z, fixed_y).clamp(-1, 1) * 0.5 + 0.5
                save_image(grid, args.samples / f"step_{step:06d}.png", nrow=8)

            if step > 0 and step % args.ckpt_every == 0:
                torch.save({
                    "G": G.state_dict(), "D": D.state_dict(),
                    "G_ema": G_ema.state_dict(),
                    "n_classes": n_classes, "z_dim": args.z_dim,
                    "ch": args.ch, "img_size": args.img_size, "step": step,
                }, args.out / f"ckpt_{step:06d}.pt")

            step += 1
            if step >= args.steps:
                break

    torch.save({
        "G": G.state_dict(), "D": D.state_dict(),
        "G_ema": G_ema.state_dict(),
        "n_classes": n_classes, "z_dim": args.z_dim,
        "ch": args.ch, "img_size": args.img_size, "step": step,
    }, args.out / "ckpt_final.pt")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
