from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
PATTERN_SUFFIX = re.compile(r"_P[0-9]+$", re.IGNORECASE)
SOURCE_SUFFIX = re.compile(r"_p[0-9]+$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit byte-identical, decoded-pixel-identical, and conservative "
            "pHash near-duplicate candidates in pattern_6541."
        )
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("pattern_6541")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pattern_6541_duplicate_audit_v1.json"),
    )
    parser.add_argument("--max-phash-distance", type=int, default=4)
    parser.add_argument("--candidate-limit", type=int, default=10000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def phash64(grayscale_32: np.ndarray) -> int:
    coefficients = cv2.dct(grayscale_32.astype(np.float32))[:8, :8]
    threshold = float(np.median(coefficients.reshape(-1)[1:]))
    bits = coefficients > threshold
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first.astype(np.float32) - float(first.mean())
    second_centered = second.astype(np.float32) - float(second.mean())
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator <= 1e-12:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def duplicate_groups(by_hash: dict[str, list[int]], records: list[dict]) -> list:
    return [
        {
            "hash": digest,
            "paths": [records[index]["path"] for index in indices],
            "patterns": sorted(
                {records[index]["pattern"] for index in indices}
            ),
        }
        for digest, indices in sorted(by_hash.items())
        if len(indices) > 1
    ]


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve(strict=True)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    records: list[dict] = []
    byte_hashes: dict[str, list[int]] = defaultdict(list)
    pixel_hashes: dict[str, list[int]] = defaultdict(list)
    thumbnails = np.empty((len(paths), 32 * 32), dtype=np.uint8)
    phashes = np.empty(len(paths), dtype=np.uint64)
    invalid = []

    for index, path in enumerate(paths):
        relative = path.relative_to(root).as_posix()
        pattern = path.parent.name
        name = PATTERN_SUFFIX.sub("", pattern)
        source = SOURCE_SUFFIX.sub("", path.stem)
        byte_hash = file_sha256(path)
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                rgb = np.asarray(image)
                gray = np.asarray(
                    image.convert("L").resize(
                        (32, 32), Image.Resampling.LANCZOS
                    ),
                    dtype=np.uint8,
                )
        except Exception as error:
            invalid.append({"path": relative, "error": repr(error)})
            rgb = np.zeros((1, 1, 3), dtype=np.uint8)
            gray = np.zeros((32, 32), dtype=np.uint8)
        pixel_hash = pixel_sha256(rgb)
        records.append(
            {
                "path": relative,
                "pattern": pattern,
                "name": name,
                "source": source,
                "byte_sha256": byte_hash,
                "pixel_sha256": pixel_hash,
            }
        )
        byte_hashes[byte_hash].append(index)
        pixel_hashes[pixel_hash].append(index)
        thumbnails[index] = gray.reshape(-1)
        phashes[index] = phash64(gray)

    lookup = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    candidates = []
    candidate_count = 0
    high_confidence = []
    chunk_size = 256
    for start in range(0, len(records), chunk_size):
        stop = min(len(records), start + chunk_size)
        xor = phashes[start:stop, None] ^ phashes[None, :]
        byte_view = xor.view(np.uint8).reshape(stop - start, len(records), 8)
        distances = lookup[byte_view].sum(axis=2)
        rows, columns = np.where(distances <= args.max_phash_distance)
        for local_row, column in zip(rows.tolist(), columns.tolist()):
            row = start + local_row
            if column <= row:
                continue
            first = records[row]
            second = records[column]
            if (
                first["pattern"] == second["pattern"]
                and first["source"] == second["source"]
            ):
                continue
            candidate_count += 1
            distance = int(distances[local_row, column])
            first_gray = thumbnails[row]
            second_gray = thumbnails[column]
            mae = float(
                np.abs(
                    first_gray.astype(np.int16)
                    - second_gray.astype(np.int16)
                ).mean()
            )
            corr = correlation(first_gray, second_gray)
            item = {
                "first": first["path"],
                "second": second["path"],
                "first_pattern": first["pattern"],
                "second_pattern": second["pattern"],
                "same_name": first["name"] == second["name"],
                "phash_distance": distance,
                "gray32_mae": mae,
                "gray32_correlation": corr,
            }
            if len(candidates) < args.candidate_limit:
                candidates.append(item)
            if distance <= 2 and mae <= 4.0 and corr >= 0.995:
                high_confidence.append(item)

    byte_groups = duplicate_groups(byte_hashes, records)
    pixel_groups = duplicate_groups(pixel_hashes, records)
    report = {
        "format": "pattern_6541_duplicate_audit_v1",
        "dataset_root": str(root),
        "image_count": len(records),
        "invalid_images": invalid,
        "byte_duplicate_groups": byte_groups,
        "decoded_pixel_duplicate_groups": pixel_groups,
        "cross_pattern_byte_duplicate_groups": [
            group for group in byte_groups if len(group["patterns"]) > 1
        ],
        "cross_pattern_pixel_duplicate_groups": [
            group for group in pixel_groups if len(group["patterns"]) > 1
        ],
        "phash": {
            "maximum_hamming_distance": args.max_phash_distance,
            "cross_source_candidate_count": candidate_count,
            "candidate_list_truncated": candidate_count > len(candidates),
            "candidates": sorted(
                candidates,
                key=lambda item: (
                    item["phash_distance"],
                    item["gray32_mae"],
                    item["first"],
                    item["second"],
                ),
            ),
            "high_confidence_rule": (
                "distance<=2 and gray32_mae<=4 and correlation>=0.995"
            ),
            "high_confidence_near_duplicates": sorted(
                high_confidence,
                key=lambda item: (
                    item["phash_distance"],
                    item["gray32_mae"],
                    item["first"],
                    item["second"],
                ),
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "image_count": len(records),
                "invalid_image_count": len(invalid),
                "byte_duplicate_group_count": len(byte_groups),
                "pixel_duplicate_group_count": len(pixel_groups),
                "cross_pattern_byte_duplicate_group_count": len(
                    report["cross_pattern_byte_duplicate_groups"]
                ),
                "cross_pattern_pixel_duplicate_group_count": len(
                    report["cross_pattern_pixel_duplicate_groups"]
                ),
                "phash_cross_source_candidate_count": candidate_count,
                "high_confidence_near_duplicate_count": len(
                    high_confidence
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
