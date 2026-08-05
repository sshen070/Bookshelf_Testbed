"""One-off/reusable utility: convert HEIC/HEIF captures to JPEG in place.

The pipeline can read HEIC directly (src/utils/imio.py), but JPEG is easier
to eyeball in a normal file browser and more portable. Run this whenever a
fresh batch of iPhone photos lands in data/reference/ or data/raw/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()


def convert_dir(root: str, quality: int = 95, delete_original: bool = True) -> None:
    root_path = Path(root)
    heic_paths = sorted(
        {p.resolve() for p in root_path.rglob("*") if p.suffix.lower() in (".heic", ".heif")}
    )
    if not heic_paths:
        print(f"No HEIC/HEIF files found under {root_path}")
        return

    converted = 0
    for path in heic_paths:
        out_path = path.with_suffix(".jpg")
        if out_path.exists():
            print(f"Skipping {path.name}: {out_path.name} already exists")
            continue
        with Image.open(path) as im:
            im.convert("RGB").save(out_path, "JPEG", quality=quality)
        converted += 1
        if delete_original:
            path.unlink()

    print(f"Converted {converted}/{len(heic_paths)} HEIC files under {root_path} to JPEG.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HEIC/HEIF images to JPEG in place.")
    parser.add_argument("root", nargs="?", default="data", help="Directory to search recursively (default: data)")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--keep-original", action="store_true", help="Keep the .heic file alongside the new .jpg")
    args = parser.parse_args()
    convert_dir(args.root, args.quality, delete_original=not args.keep_original)
