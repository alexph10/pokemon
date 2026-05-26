"""Scrape high-res Pokémon card renders from the pokemontcg.io API.

Output: data/raw/pokemon_cards/<rarity_slug>/<set>_<num>_<id>.png
Usage:  python src/scrape.py --per-rarity 10000 --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_BASE = "https://api.pokemontcg.io/v2"
DEFAULT_OUT_DIR = Path("data/raw/pokemon_cards")
DEFAULT_PER_RARITY = 10_000
DEFAULT_WORKERS = 12
PAGE_SIZE = 250  # API max
USER_AGENT = "pokemon-gan-scraper/1.0 (+https://github.com/alexph10/pokemon)"

# Static fallback if /rarities endpoint is unreachable.
KNOWN_RARITIES: tuple[str, ...] = (
    "Common", "Uncommon", "Rare", "Rare Holo", "Rare Holo EX",
    "Rare Holo GX", "Rare Holo LV.X", "Rare Holo Star", "Rare Holo V",
    "Rare Holo VMAX", "Rare Holo VSTAR", "Rare BREAK", "Rare Prime",
    "Rare Prism Star", "Rare ACE", "Rare Shining", "Rare Shiny",
    "Rare Shiny GX", "Rare Ultra", "Rare Secret", "Rare Rainbow",
    "Rare Radiant", "Amazing Rare", "Promo", "LEGEND",
    "Trainer Gallery Rare Holo", "Classic Collection", "Double Rare",
    "Ultra Rare", "Illustration Rare", "Special Illustration Rare",
    "Hyper Rare", "ACE SPEC Rare", "Shiny Rare", "Shiny Ultra Rare",
)

logger = logging.getLogger("pokemon_scrape")


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def _build_session(api_key: str | None) -> requests.Session:
    # Session with retry/backoff for transient API + CDN failures.
    session = requests.Session()
    retry = Retry(
        total=6, backoff_factor=1.5,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET", "HEAD")),
        raise_on_status=False, respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT,
                            "Accept-Encoding": "gzip, deflate"})
    if api_key:
        session.headers["X-Api-Key"] = api_key
    return session


_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    return _slug_re.sub("_", value.strip().lower()).strip("_") or "unknown"


def safe_filename(card: dict) -> str:
    # Stable, collision-free filename for a card record.
    set_id = slugify(str(card.get("set", {}).get("id") or "set"))
    number = slugify(str(card.get("number") or "0"))
    card_id = slugify(str(card.get("id") or "card"))
    return f"{set_id}_{number}_{card_id}.png"


@dataclass
class RarityResult:
    rarity: str
    requested: int
    available: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    manifest: list[dict] = field(default_factory=list)


def fetch_rarities(session: requests.Session) -> list[str]:
    # Live fetch with static fallback.
    try:
        r = session.get(f"{API_BASE}/rarities", timeout=30)
        r.raise_for_status()
        data = r.json().get("data") or []
        if data:
            return sorted({str(x) for x in data})
    except requests.RequestException as exc:
        logger.warning("Could not fetch /rarities (%s); using static list.", exc)
    return list(KNOWN_RARITIES)


def fetch_cards_for_rarity(session: requests.Session, rarity: str,
                           max_cards: int) -> list[dict]:
    # Page through /cards for a single rarity (Lucene-style query).
    cards: list[dict] = []
    page = 1
    q = f'rarity:"{rarity}"'
    while len(cards) < max_cards:
        params = {
            "q": q, "page": page, "pageSize": PAGE_SIZE,
            "select": "id,name,number,rarity,set,images",
            "orderBy": "set.releaseDate,number",
        }
        try:
            r = session.get(f"{API_BASE}/cards", params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
        except requests.RequestException as exc:
            logger.error("Metadata fetch failed for %r page %d: %s",
                         rarity, page, exc)
            break

        batch = payload.get("data") or []
        if not batch:
            break
        cards.extend(batch)
        total = int(payload.get("totalCount") or 0)
        logger.debug("  %s: page %d -> +%d (cum %d / api total %d)",
                     rarity, page, len(batch), len(cards), total)
        if total and len(cards) >= total:
            break
        if len(batch) < PAGE_SIZE:
            break
        page += 1
        time.sleep(0.2)  # polite pause between metadata pages
    return cards[:max_cards]


def download_image(session: requests.Session, url: str,
                   dest: Path) -> tuple[bool, str]:
    # Atomic download via .part file; returns (ok, message).
    if dest.exists() and dest.stat().st_size > 0:
        return True, "exists"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, timeout=120, stream=True) as r:
            if r.status_code != 200:
                return False, f"http {r.status_code}"
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        fh.write(chunk)
        if tmp.stat().st_size < 1024:
            tmp.unlink(missing_ok=True)
            return False, "file too small"
        tmp.replace(dest)
        return True, "downloaded"
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        return False, f"error: {exc.__class__.__name__}"
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return False, f"io error: {exc}"


def download_rarity(session: requests.Session, rarity: str,
                    cards: list[dict], out_root: Path, workers: int,
                    dry_run: bool) -> RarityResult:
    result = RarityResult(rarity=rarity, requested=len(cards),
                          available=len(cards))
    rarity_dir = out_root / slugify(rarity)
    rarity_dir.mkdir(parents=True, exist_ok=True)

    # Build job list + manifest entries.
    jobs: list[tuple[dict, str, Path]] = []
    for card in cards:
        images = card.get("images") or {}
        url = images.get("large") or images.get("small")
        if not url:
            result.failed += 1
            continue
        dest = rarity_dir / safe_filename(card)
        jobs.append((card, url, dest))
        result.manifest.append({
            "id": card.get("id"),
            "name": card.get("name"),
            "number": card.get("number"),
            "set_id": (card.get("set") or {}).get("id"),
            "set_name": (card.get("set") or {}).get("name"),
            "rarity": card.get("rarity"),
            "image_url": url,
            "file": str(dest.relative_to(out_root)),
        })

    if dry_run:
        logger.info("[dry-run] %s: would download %d images to %s",
                    rarity, len(jobs), rarity_dir)
        return result

    logger.info("Downloading %d images for rarity %r -> %s",
                len(jobs), rarity, rarity_dir)

    def _task(job: tuple[dict, str, Path]) -> tuple[str, bool, str]:
        _card, url, dest = job
        time.sleep(random.uniform(0.0, 0.05))  # jitter to avoid lockstep
        ok, msg = download_image(session, url, dest)
        return dest.name, ok, msg

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_task, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            name, ok, msg = fut.result()
            if ok and msg == "exists":
                result.skipped_existing += 1
            elif ok:
                result.downloaded += 1
            else:
                result.failed += 1
                logger.debug("  fail %s: %s", name, msg)
            if i % 100 == 0 or i == len(futures):
                logger.info("  %s: %d/%d (ok=%d, skipped=%d, fail=%d)",
                            rarity, i, len(futures), result.downloaded,
                            result.skipped_existing, result.failed)

    # Per-rarity manifest for reproducibility.
    with open(rarity_dir / "_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({
            "rarity": rarity, "requested": result.requested,
            "available": result.available, "downloaded": result.downloaded,
            "skipped_existing": result.skipped_existing,
            "failed": result.failed, "cards": result.manifest,
        }, fh, indent=2)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--per-rarity", type=int, default=DEFAULT_PER_RARITY,
                   help="Max images per rarity (catalog has ~19k cards total)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--rarities", type=str, default=None,
                   help="Comma-separated rarities; default = all known")
    p.add_argument("--api-key", type=str,
                   default=os.environ.get("POKEMONTCG_IO_API_KEY"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", out_root.resolve())

    session = _build_session(args.api_key)

    rarities = ([r.strip() for r in args.rarities.split(",") if r.strip()]
                if args.rarities else fetch_rarities(session))
    logger.info("Targeting %d rarities (cap %d each).",
                len(rarities), args.per_rarity)

    summary: list[RarityResult] = []
    for rarity in rarities:
        logger.info("Rarity: %s", rarity)
        try:
            cards = fetch_cards_for_rarity(session, rarity, args.per_rarity)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to enumerate %r: %s", rarity, exc)
            continue
        if not cards:
            logger.info("  no cards returned for %r", rarity)
            summary.append(RarityResult(rarity=rarity,
                                        requested=args.per_rarity))
            continue
        summary.append(download_rarity(session, rarity, cards, out_root,
                                       args.workers, args.dry_run))

    # Final report + top-level summary file.
    logger.info("SUMMARY")
    total_dl = total_skip = total_fail = 0
    for r in summary:
        logger.info("  %-35s req=%-6d avail=%-6d ok=%-6d skip=%-6d fail=%d",
                    r.rarity, r.requested, r.available, r.downloaded,
                    r.skipped_existing, r.failed)
        total_dl += r.downloaded
        total_skip += r.skipped_existing
        total_fail += r.failed
    logger.info("Totals: downloaded=%d  skipped=%d  failed=%d",
                total_dl, total_skip, total_fail)

    summary_path = out_root / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump([{
            "rarity": r.rarity, "requested": r.requested,
            "available": r.available, "downloaded": r.downloaded,
            "skipped_existing": r.skipped_existing, "failed": r.failed,
        } for r in summary], fh, indent=2)
    logger.info("Wrote %s", summary_path)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
