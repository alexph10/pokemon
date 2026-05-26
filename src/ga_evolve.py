"""Latent-space GA over a trained conditional GAN.

Genome  = (z ∈ R^z_dim, rarity_idx)
Fitness = D(G(z, y))  (+ optional novelty term in z-space)
Operators: tournament selection, blend/SLERP crossover, Gaussian mutation.

Outputs per-generation grids and final top-K individuals.

Run:
    python src/ga_evolve.py --ckpt checkpoints/ckpt_final.pt --gens 50
"""

from __future__ import annotations
import argparse, math, random, sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import Generator, Discriminator  # noqa: E402


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    # Spherical interpolation; falls back to lerp if vectors are colinear.
    an = a / (a.norm() + 1e-8)
    bn = b / (b.norm() + 1e-8)
    omega = torch.acos((an * bn).sum().clamp(-1 + 1e-7, 1 - 1e-7))
    so = torch.sin(omega)
    if so.item() < 1e-6:
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


def crossover(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "slerp":
        return slerp(a, b, random.random())
    alpha = random.uniform(0.2, 0.8)  # arithmetic blend
    return alpha * a + (1 - alpha) * b


def mutate(z: torch.Tensor, sigma: float) -> torch.Tensor:
    return z + sigma * torch.randn_like(z)


def tournament(pop_idx: list[int], fitness: torch.Tensor, k: int = 3) -> int:
    cand = random.sample(pop_idx, k)
    return max(cand, key=lambda i: fitness[i].item())


@torch.no_grad()
def evaluate(G, D, Z: torch.Tensor, Y: torch.Tensor,
             novelty_weight: float = 0.0):
    X = G(Z, Y)
    score = D(X, Y)
    if novelty_weight > 0:
        # Mean cosine distance to the rest of the population in z-space.
        Zn = Z / (Z.norm(dim=1, keepdim=True) + 1e-8)
        sim = Zn @ Zn.t()
        nov = 1 - sim.mean(dim=1)
        score = score + novelty_weight * nov
    return score, X


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--pop", type=int, default=64)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--elite", type=int, default=4)
    p.add_argument("--mut-rate", type=float, default=0.9)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--cx", choices=["blend", "slerp"], default="slerp")
    p.add_argument("--rarity-idx", type=int, default=None,
                   help="Lock all genomes to this rarity (default: random)")
    p.add_argument("--novelty", type=float, default=0.0,
                   help="Cosine-distance novelty bonus weight")
    p.add_argument("--out", type=Path, default=Path("generated/ga"))
    p.add_argument("--save-top", type=int, default=16)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    G = Generator(ck["z_dim"], ck["n_classes"], ck["ch"], ck["img_size"]).to(dev)
    D = Discriminator(ck["n_classes"], ck["ch"], ck["img_size"]).to(dev)
    G.load_state_dict(ck["G_ema"]); D.load_state_dict(ck["D"])
    G.eval(); D.eval()
    n_classes, z_dim = ck["n_classes"], ck["z_dim"]
    print(f"Loaded ckpt step={ck.get('step')}  n_classes={n_classes}  z_dim={z_dim}")

    args.out.mkdir(parents=True, exist_ok=True)

    # Initial population.
    Z = torch.randn(args.pop, z_dim, device=dev)
    Y = (torch.full((args.pop,), args.rarity_idx, dtype=torch.long, device=dev)
         if args.rarity_idx is not None
         else torch.randint(0, n_classes, (args.pop,), device=dev))

    history: list[tuple[int, float, float]] = []
    for gen in range(args.gens):
        fit, X = evaluate(G, D, Z, Y, novelty_weight=args.novelty)
        order = torch.argsort(fit, descending=True)
        best, mean = fit[order[0]].item(), fit.mean().item()
        history.append((gen, best, mean))
        print(f"gen {gen:03d}  best={best:+.3f}  mean={mean:+.3f}")

        # Save a snapshot grid of top-K this generation.
        top = order[:args.save_top]
        grid = (X[top].clamp(-1, 1) * 0.5 + 0.5)
        nrow = max(1, int(math.sqrt(args.save_top)))
        save_image(grid, args.out / f"gen_{gen:03d}.png", nrow=nrow)

        # Build next generation: elitism + tournament breeding + mutation.
        elites = order[:args.elite].tolist()
        new_Z = [Z[i].clone() for i in elites]
        new_Y = [Y[i].clone() for i in elites]
        pop_idx = list(range(args.pop))
        while len(new_Z) < args.pop:
            a = tournament(pop_idx, fit); b = tournament(pop_idx, fit)
            child = crossover(Z[a], Z[b], mode=args.cx)
            if random.random() < args.mut_rate:
                child = mutate(child, args.sigma)
            new_Z.append(child)
            new_Y.append(Y[a] if random.random() < 0.5 else Y[b])
        Z = torch.stack(new_Z); Y = torch.stack(new_Y)

    # Final unconditional evaluation + per-image dump of top-K.
    fit, X = evaluate(G, D, Z, Y, novelty_weight=0.0)
    order = torch.argsort(fit, descending=True)
    final_dir = args.out / "final_top"
    final_dir.mkdir(exist_ok=True)
    for rank, i in enumerate(order[:args.save_top].tolist()):
        img = (X[i].clamp(-1, 1) * 0.5 + 0.5)
        save_image(img, final_dir / f"rank_{rank:02d}_fit{fit[i].item():+.2f}.png")

    with open(args.out / "history.csv", "w", encoding="utf-8") as fh:
        fh.write("gen,best,mean\n")
        for g, bv, mv in history:
            fh.write(f"{g},{bv:.4f},{mv:.4f}\n")
    print(f"Wrote {final_dir} and {args.out / 'history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
