"""Preprocess scraped Pokémon card PNGs for GAN training.

Reads  data/raw/pokemon_cards/<rarity>/*.png
Writes data/processed/pokemon_cards/<rarity>/*.png  (square, padded, resized)
       data/processed/index.csv  (path, rarity, rarity_idx)
"""

from __future__ import annotations
import argparse, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

DEFAULT_IN = Path("data/raw/pokemon_cards")
DEFAULT_OUT = Path("data/processed/pokemon_cards")
DEFAULT_SIZE = 128
PAD_COLOR = (255, 255, 255)


def resize_pad(img: Image.Image, size: int) -> Image.Image:
    # Letterbox to square so we preserve aspect (cards are 5:7).
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def process_one(src: Path, dst: Path, size: int) -> bool:
    if dst.exists():
        return True
    try:
        with Image.open(src) as im:
            resize_pad(im, size).save(dst, format="PNG", optimize=True)
        return True
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-per-rarity", type=int, default=0,
                   help="Skip rarities with fewer than this many images")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rarity_dirs = sorted(d for d in args.in_dir.iterdir() if d.is_dir())

    # Filter rarities by size and assign a stable integer index.
    rarities = [d.name for d in rarity_dirs
                if sum(1 for _ in d.glob("*.png")) >= args.min_per_rarity]
    rarity_idx = {r: i for i, r in enumerate(sorted(rarities))}

    rows: list[tuple[str, str, int]] = []
    ok = fail = 0
    for rarity in sorted(rarities):
        src_dir = args.in_dir / rarity
        dst_dir = args.out / rarity
        dst_dir.mkdir(parents=True, exist_ok=True)
        srcs = sorted(src_dir.glob("*.png"))
        print(f"[{rarity}] {len(srcs)} images -> {dst_dir}")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(process_one, s, dst_dir / s.name, args.size): s
                    for s in srcs}
            for fut in as_completed(futs):
                s = futs[fut]
                if fut.result():
                    ok += 1
                    rows.append(((dst_dir / s.name).as_posix(),
                                 rarity, rarity_idx[rarity]))
                else:
                    fail += 1

    index_path = args.out.parent / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "rarity", "rarity_idx"])
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {index_path}")
    print(f"Rarities: {rarity_idx}")
    print(f"Totals: ok={ok} fail={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
