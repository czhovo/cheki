import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import ConvexHull
from quadrilateral_fitter import QuadrilateralFitter

try:
    from config import MODEL_DIR, MASK_THRESHOLD
except ImportError:
    MODEL_DIR = "./sam3_model/facebook/sam3"
    MASK_THRESHOLD = 0.5


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PAPER_PROMPT_TRIALS = [
    ("polaroid photo paper frame", 0.5),
    ("polaroid photo paper frame", 0.4),
    ("polaroid photo", 0.4),
    ("photo paper frame", 0.4),
]
IMAGE_AREA_PROMPT = "the image area of the polaroid photo"
PAPER_SIZE = (800, 1272)
EXPECTED_AREA_VERTICES = np.array([[55, 100], [745, 100], [745, 1022], [55, 1022]], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract all polaroids from one image or all images in a directory."
    )
    parser.add_argument("input", help="Input image path or a directory containing images.")
    parser.add_argument("-o", "--output-dir", default="outs", help="Directory for extracted images.")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Local SAM3 model directory.")
    parser.add_argument("--device", default=None, help="Device override, for example cuda or cpu.")
    parser.add_argument("--wb", dest="wb", action="store_true", default=True, help="Enable white balance.")
    parser.add_argument("--no-wb", dest="wb", action="store_false", help="Disable white balance.")
    parser.add_argument("--denoise", dest="denoise", action="store_true", default=True, help="Enable LAB denoise after white balance.")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false", help="Disable LAB denoise.")
    parser.add_argument(
        "--sharpen",
        choices=("off", "low"),
        default="low",
        help="Apply LAB L-channel USM after denoise: off or low.",
    )
    parser.add_argument("--debug", action="store_true", help="Save one overlay image per input for inspection.")
    parser.add_argument("--min-area-ratio", type=float, default=0.002, help="Ignore tiny paper masks.")
    return parser.parse_args()


def iter_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported image type: {path}")
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    raise FileNotFoundError(f"Input not found: {path}")


def save_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def order_quad(points):
    pts = np.asarray(points, dtype=np.float64)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmax(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmin(diffs)],
        ],
        dtype=np.float64,
    )


def points_inside_quad(points, quad):
    pts = np.asarray(points, dtype=np.float64)
    q = np.asarray(quad, dtype=np.float64)
    signed_area = sum(q[i, 0] * q[(i + 1) % 4, 1] - q[(i + 1) % 4, 0] * q[i, 1] for i in range(4))
    is_cw = signed_area > 0
    inside = np.ones(len(pts), dtype=bool)
    for i in range(4):
        p1, p2 = q[i], q[(i + 1) % 4]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        cross = dx * (pts[:, 1] - p1[1]) - dy * (pts[:, 0] - p1[0])
        inside &= (cross >= 0) if is_cw else (cross <= 0)
    return inside


def shrink_quad(vertices, factor=0.03):
    center = np.asarray(vertices, dtype=np.float64).mean(axis=0)
    return np.asarray(vertices, dtype=np.float64) + factor * (center - np.asarray(vertices, dtype=np.float64))


def fit_quadrilateral(points):
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]
    p1 = hull_pts[np.argmin(hull_pts[:, 0])]
    p2 = hull_pts[np.argmax(np.sum((hull_pts - p1) ** 2, axis=1))]
    vector = p2 - p1
    norm = np.linalg.norm(vector)
    if norm < 1:
        return np.array([p1] * 4, dtype=np.float64)
    signed = (vector[0] * (hull_pts[:, 1] - p1[1]) - vector[1] * (hull_pts[:, 0] - p1[0])) / norm
    approx = np.array([p1, p2, hull_pts[np.argmax(signed)], hull_pts[np.argmin(signed)]])
    angles = np.arctan2(approx[:, 1] - approx.mean(0)[1], approx[:, 0] - approx.mean(0)[0])
    approx = approx[np.argsort(angles)]
    exterior = points[~points_inside_quad(points, shrink_quad(approx, 0.03))]
    if len(exterior) < 4:
        return approx
    fitter = QuadrilateralFitter(polygon=exterior)
    vertices = np.array(fitter.fit(), dtype=np.float64)
    angles = np.arctan2(vertices[:, 1] - vertices.mean(0)[1], vertices[:, 0] - vertices.mean(0)[0])
    return vertices[np.argsort(angles)]


def line_from_pts(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) > abs(dy):
        slope = dy / (dx + 1e-9)
        return "h", slope, p1[1] - slope * p1[0]
    slope = dx / (dy + 1e-9)
    return "v", slope, p1[0] - slope * p1[1]


def ransac_quad_fit(exterior, init_quad, filter_dist=80, inlier_dist=5):
    init_lines = [
        line_from_pts(init_quad[0], init_quad[1]),
        line_from_pts(init_quad[1], init_quad[2]),
        line_from_pts(init_quad[2], init_quad[3]),
        line_from_pts(init_quad[3], init_quad[0]),
    ]
    ex, ey = exterior[:, 0], exterior[:, 1]
    dists = []
    for orient, m, b in init_lines:
        dists.append(np.abs(ey - (m * ex + b)) if orient == "h" else np.abs(ex - (m * ey + b)))
    edge_idx = np.argmin(np.column_stack(dists), axis=1)

    def ransac_line(pts, orient, init_slope, init_intercept):
        if len(pts) < 5:
            return init_slope, init_intercept
        if orient == "h":
            pts = pts[np.abs(pts[:, 1] - (init_slope * pts[:, 0] + init_intercept)) < filter_dist]
            x = pts[:, 0:1].astype(np.float32)
            y = pts[:, 1:2].astype(np.float32)
        else:
            pts = pts[np.abs(pts[:, 0] - (init_slope * pts[:, 1] + init_intercept)) < filter_dist]
            x = pts[:, 1:2].astype(np.float32)
            y = pts[:, 0:1].astype(np.float32)
        if len(pts) < 5:
            return init_slope, init_intercept

        best_inliers, best_model = 0, (init_slope, init_intercept)
        rng = np.random.default_rng(42)
        for _ in range(min(500, len(pts) * 10)):
            idx = rng.choice(len(pts), min(5, len(pts)), replace=False)
            a = np.hstack([x[idx], np.ones((len(idx), 1))])
            try:
                coeff = np.linalg.lstsq(a, y[idx], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            slope, intercept = float(coeff[0, 0]), float(coeff[1, 0])
            inliers = int(np.sum(np.abs(y - (slope * x + intercept)) < inlier_dist))
            if inliers > best_inliers:
                best_inliers, best_model = inliers, (slope, intercept)

        slope, intercept = best_model
        in_mask = (np.abs(y - (slope * x + intercept)) < inlier_dist).flatten()
        if in_mask.sum() >= 3:
            a = np.hstack([x[in_mask], np.ones((in_mask.sum(), 1))])
            coeff = np.linalg.lstsq(a, y[in_mask], rcond=None)[0]
            return float(coeff[0, 0]), float(coeff[1, 0])
        return best_model

    refined = [ransac_line(exterior[edge_idx == ei], *init_lines[ei]) for ei in range(4)]

    def intersect(hl, vl):
        mh, bh = hl
        mv, bv = vl
        denom = 1 - mv * mh
        if abs(denom) < 1e-9:
            return None
        x = (mv * bh + bv) / denom
        return np.array([x, mh * x + bh])

    corners = [
        intersect(refined[0], refined[3]),
        intersect(refined[0], refined[1]),
        intersect(refined[2], refined[1]),
        intersect(refined[2], refined[3]),
    ]
    result = np.array([c if c is not None else init_quad[i] for i, c in enumerate(corners)], dtype=np.float64)
    return order_quad(result)


def mask_to_quad(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty contour")
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(points) < 8:
        raise ValueError("not enough contour points")
    return fit_quadrilateral(points)


def rectify(image, vertices, size=PAPER_SIZE):
    w, h = size
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(vertices.astype(np.float32), dst)
    rectified = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_CUBIC)
    valid_mask = cv2.warpPerspective(np.ones(image.shape[:2], dtype=np.uint8), matrix, (w, h)).astype(bool)
    return rectified, valid_mask


def clear_torch_cache(torch_module):
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def segment(torch_module, processor, model, device, image, prompt, threshold):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch_module.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]
    masks = results["masks"].cpu().numpy() if len(results["masks"]) else np.empty((0, image.height, image.width))
    del inputs, outputs, results
    clear_torch_cache(torch_module)
    return masks


def detect_paper_masks(torch_module, processor, model, device, image, min_area_ratio):
    image_area = image.width * image.height
    for prompt, threshold in PAPER_PROMPT_TRIALS:
        masks = segment(torch_module, processor, model, device, image, prompt, threshold)
        masks = [m.astype(bool) for m in masks if int(m.sum()) >= image_area * min_area_ratio]
        masks = remove_duplicate_masks(masks)
        if masks:
            return masks, prompt, threshold
    return [], None, None


def remove_duplicate_masks(masks, iou_threshold=0.85):
    kept = []
    for mask in sorted(masks, key=lambda m: int(m.sum()), reverse=True):
        duplicate = False
        for old in kept:
            inter = np.logical_and(mask, old).sum()
            union = np.logical_or(mask, old).sum()
            if union and inter / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(mask)
    return sorted(kept, key=lambda m: (np.argwhere(m)[:, 1].mean(), np.argwhere(m)[:, 0].mean()))


def detect_image_area_vertices(torch_module, processor, model, device, rectified_paper):
    rectified_pil = Image.fromarray(rectified_paper)
    masks = segment(torch_module, processor, model, device, rectified_pil, IMAGE_AREA_PROMPT, 0.4)
    if len(masks) == 0:
        return EXPECTED_AREA_VERTICES.copy(), True
    largest = max((m.astype(bool) for m in masks), key=lambda m: int(m.sum()))
    try:
        vertices = mask_to_quad(largest)
    except Exception:
        return EXPECTED_AREA_VERTICES.copy(), True
    max_dev = np.max(np.linalg.norm(vertices - EXPECTED_AREA_VERTICES, axis=1))
    if max_dev > 150:
        return EXPECTED_AREA_VERTICES.copy(), True
    return vertices, False


def white_balance(rectified_paper, area_vertices, valid_mask):
    h, w = rectified_paper.shape[:2]
    border_mask = np.ones((h, w), dtype=np.uint8)
    margin = 5
    inner = np.array(
        [
            [area_vertices[0][0] - margin, area_vertices[0][1] - margin],
            [area_vertices[1][0] + margin, area_vertices[1][1] - margin],
            [area_vertices[2][0] + margin, area_vertices[2][1] + margin],
            [area_vertices[3][0] - margin, area_vertices[3][1] + margin],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(border_mask, [inner], 0)
    border_mask = border_mask.astype(bool)
    is_bright = np.all(rectified_paper > 170, axis=2)
    is_neutral = np.std(rectified_paper.astype(np.float32), axis=2) < 25
    is_white = is_bright & is_neutral & border_mask

    blocks = []
    for y in range(0, h - 32, 16):
        for x in range(0, w - 32, 16):
            block = is_white[y : y + 32, x : x + 32]
            if np.sum(block) / 1024 > 0.8:
                pixels = rectified_paper[y : y + 32, x : x + 32][block]
                blocks.append({"mean": pixels.mean(axis=0), "var": pixels.var(axis=0).mean()})

    if blocks:
        blocks.sort(key=lambda b: b["var"])
        ref_white = np.mean([b["mean"] for b in blocks[:10]], axis=0)
        gains = np.clip(240.0 / np.maximum(ref_white, 1.0), 0.5, 2.0)
        output = np.clip(rectified_paper.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    else:
        ref_white = np.array([0.0, 0.0, 0.0])
        gains = np.array([1.0, 1.0, 1.0])
        output = rectified_paper.copy()

    missing_mask = cv2.dilate(
        (~valid_mask).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ).astype(bool)
    if missing_mask.any():
        output[missing_mask] = (240, 240, 240)
    return output, {
        "white_blocks": len(blocks),
        "reference_white_rgb": [round(float(v), 2) for v in ref_white],
        "gains_rgb": [round(float(v), 4) for v in gains],
        "missing_pixels_filled": int(missing_mask.sum()),
    }


def denoise_lab_luma_chroma(rgb_image, l_h=3.5, ab_h=6.0):
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.fastNlMeansDenoising(l_channel, None, h=float(l_h), templateWindowSize=7, searchWindowSize=21)
    a_channel = cv2.fastNlMeansDenoising(a_channel, None, h=float(ab_h), templateWindowSize=7, searchWindowSize=21)
    b_channel = cv2.fastNlMeansDenoising(b_channel, None, h=float(ab_h), templateWindowSize=7, searchWindowSize=21)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)


def sharpen_luma_unsharp(rgb_image, sigma, amount, threshold):
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_float = l_channel.astype(np.float32)
    blurred = cv2.GaussianBlur(l_float, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    detail = l_float - blurred
    if threshold > 0:
        detail = np.where(np.abs(detail) >= threshold, detail, 0.0)
    l_out = np.clip(l_float + amount * detail, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_out, a_channel, b_channel]), cv2.COLOR_LAB2RGB)


def sharpen_output(rgb_image, mode):
    if mode == "off":
        return rgb_image, None
    if mode == "low":
        params = {"sigma": 1.0, "amount": 0.45, "threshold": 3.0}
    else:
        raise ValueError(f"Unknown sharpen mode: {mode}")
    return sharpen_luma_unsharp(rgb_image, **params), {
        "mode": mode,
        "algorithm": "LAB L-channel unsharp mask",
        **params,
    }


def make_debug_overlay(image, items):
    overlay = image.copy()
    canvas = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 170, 255), (255, 0, 255), (255, 220, 0)]
    for idx, item in enumerate(items, start=1):
        color = colors[(idx - 1) % len(colors)]
        mask = item["mask"]
        vertices = item["vertices"].astype(np.int32)
        overlay[mask] = overlay[mask] * 0.55 + np.array(color, dtype=np.float32) * 0.45
        cv2.polylines(canvas, [vertices], True, color, 6, cv2.LINE_AA)
        center = tuple(np.round(item["vertices"].mean(axis=0)).astype(int))
        cv2.putText(canvas, str(idx), center, cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5, cv2.LINE_AA)
    return np.clip(overlay * 0.55 + canvas * 0.45, 0, 255).astype(np.uint8)


def process_image(path, output_dir, torch_module, processor, model, device, use_wb, use_denoise, sharpen_mode, debug, min_area_ratio):
    print(f"\n处理: {path}")
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img_np = np.array(image)
    masks, prompt, threshold = detect_paper_masks(torch_module, processor, model, device, image, min_area_ratio)
    print(f"  纸框检测: {len(masks)} 张", end="")
    if prompt:
        print(f" ({prompt}, threshold={threshold})")
    else:
        print()

    output_dir.mkdir(parents=True, exist_ok=True)
    base = path.stem
    report = {
        "input": str(path),
        "white_balance": use_wb,
        "denoise": {
            "enabled": use_denoise,
            "algorithm": "LAB per-channel fastNlMeansDenoising" if use_denoise else None,
            "l_h": 3.5 if use_denoise else None,
            "ab_h": 6.0 if use_denoise else None,
            "position": "after white_balance" if use_denoise else None,
        },
        "sharpen": {
            "mode": sharpen_mode,
            "position": "after denoise",
            "algorithm": "LAB L-channel unsharp mask" if sharpen_mode != "off" else None,
        },
        "paper_count": len(masks),
        "outputs": [],
    }
    debug_items = []

    for idx, mask in enumerate(masks, start=1):
        try:
            vertices = mask_to_quad(mask)
            rectified, valid_mask = rectify(img_np, vertices)
            wb_info = None
            if use_wb:
                area_vertices, area_fallback = detect_image_area_vertices(
                    torch_module, processor, model, device, rectified
                )
                output, wb_info = white_balance(rectified, area_vertices, valid_mask)
            else:
                area_vertices, area_fallback = None, None
                output = rectified
            denoise_info = None
            if use_denoise:
                output = denoise_lab_luma_chroma(output, 3.5, 6.0)
                denoise_info = {"algorithm": "LAB per-channel fastNlMeansDenoising", "l_h": 3.5, "ab_h": 6.0}
            output, sharpen_info = sharpen_output(output, sharpen_mode)
            out_path = output_dir / f"{base}_p{idx:02d}.png"
            save_rgb(out_path, output)
            debug_items.append({"mask": mask, "vertices": vertices})
            item = {
                "index": idx,
                "output": str(out_path),
                "vertices": vertices.round(1).tolist(),
                "area_vertices": None if area_vertices is None else area_vertices.round(1).tolist(),
                "used_area_fallback": area_fallback,
                "white_balance": wb_info,
                "denoise": denoise_info,
                "sharpen": sharpen_info,
            }
            report["outputs"].append(item)
            print(f"  p{idx:02d}: 已保存 {out_path}")
        except Exception as exc:
            report["outputs"].append({"index": idx, "error": str(exc)})
            print(f"  p{idx:02d}: 跳过 ({exc})")
        finally:
            gc.collect()
            clear_torch_cache(torch_module)

    if debug and debug_items:
        debug_path = output_dir / f"{base}_debug_overlay.png"
        save_rgb(debug_path, make_debug_overlay(img_np.astype(np.float32), debug_items))
        report["debug_overlay"] = str(debug_path)
        print(f"  debug: 已保存 {debug_path}")

    report_path = output_dir / f"{base}_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    args = parse_args()
    start = time.time()
    images = iter_images(args.input)
    if not images:
        print("未找到可处理的图片。")
        return 1

    import torch
    from transformers import Sam3Model, Sam3Processor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载 SAM3: {args.model_dir}")
    print(f"设备: {device}")
    model = Sam3Model.from_pretrained(args.model_dir).to(device)
    processor = Sam3Processor.from_pretrained(args.model_dir)

    all_reports = []
    output_dir = Path(args.output_dir)
    for image_path in images:
        all_reports.append(
            process_image(
                image_path,
                output_dir,
                torch,
                processor,
                model,
                device,
                args.wb,
                args.denoise,
                args.sharpen,
                args.debug,
                args.min_area_ratio,
            )
        )

    summary = {
        "input": args.input,
        "image_count": len(images),
        "polaroid_count": sum(len(r["outputs"]) for r in all_reports),
        "seconds": round(time.time() - start, 2),
    }
    with (output_dir / "batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n完成: {summary['image_count']} 张图片, {summary['polaroid_count']} 张拍立得, 输出目录 {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
