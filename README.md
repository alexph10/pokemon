#### pokemon

GAN implementation for generating Pokémon cards, with a genetic-algorithm
search layer on top of the trained generator's latent space.

#### Pipeline

```
scrape → preprocess → train base GAN → fine-tune per rarity → GA latent search
```

#### Setup

```
pip install -r requirements.txt
```

#### 1. Scrape

Pulls high-res renders from the official Pokémon TCG API into
`data/raw/pokemon_cards/<rarity>/`.

```
python src/scrape.py --per-rarity 10000 --workers 16
```

#### 2. Preprocess

Letterbox-pads to 128×128 and writes `data/processed/index.csv`. Keep only
rarities with at least 200 images so fine-tuning has something to learn from.

```
python src/preprocess.py --size 128 --min-per-rarity 200 --workers 8
```

#### 3. Train — Strategy B (base + per-rarity fine-tune)

Strategy B trains **one shared base model** across all rarities, then
**fine-tunes a specialist** for each rarity from that base. This gives every
rarity (even the small ones) the benefit of seeing the full card distribution
during base training, then sharpens to the target style.

##### 3a. Train the base (all rarities, conditional)

```
python src/train_gan.py --img-size 128 --batch 32 --epochs 80
```

Outputs: `checkpoints/ckpt_final.pt` + sample grids in `samples/`.

##### 3b. Fine-tune one specialist per rarity

`--resume` loads weights from the base; mismatched class-embedding tensors
(14 classes → 1 class) are reinitialized automatically. Lower LR (`5e-5`) and
fewer epochs prevent the small dataset from destroying the base's structure.

```
python src/train_gan.py --rarity rare_holo_v --resume checkpoints/ckpt_final.pt --epochs 100 --lr-g 5e-5 --lr-d 5e-5 --batch 32
```

Outputs land in `checkpoints/<rarity>/` and `samples/<rarity>/`.

**Per-rarity epoch budget:**

| Dataset size | Epochs |
|---|---|
| ≤ 500 imgs | 200–300 |
| 500 – 2 k | 100–200 |
| 2 k – 10 k | 50–100 |

##### 3c. (Optional) Fine-tune every rarity in one shot (PowerShell)

```
foreach ($r in (Get-ChildItem data/processed/pokemon_cards -Directory).Name) {
    python src/train_gan.py --rarity $r --resume checkpoints/ckpt_final.pt --epochs 150 --lr-g 5e-5 --lr-d 5e-5 --batch 32
}
```

#### 4. GA latent-space search

Evolves `z` vectors against a specialist's discriminator. Add `--novelty` to
bias toward diverse outputs.

```
python src/ga_evolve.py --ckpt checkpoints/rare_holo_v/ckpt_final.pt --gens 100 --pop 128 --novelty 0.2
```

Outputs: `generated/ga/gen_*.png` (per-generation top-K grids) +
`generated/ga/final_top/rank_*.png`.

#### Layout

```
src/
  scrape.py        Pokémon TCG API scraper
  preprocess.py    resize + index
  models.py        conditional Generator + projection Discriminator
  train_gan.py     training loop (R1 + EMA, supports --rarity, --resume)
  ga_evolve.py     latent-space genetic algorithm
data/
  raw/pokemon_cards/<rarity>/...png
  processed/pokemon_cards/<rarity>/...png
  processed/index.csv
checkpoints/       ckpt_final.pt (base) + <rarity>/ckpt_final.pt (specialists)
samples/           training-time sample grids
generated/         GA outputs
```
