"""
拍立得提取脚本
输入一张图片，使用 SAM3 检测所有拍立得纸框，拟合四边形并透视矫正，
输出三栏可视化：原图+四边形标注 / 四边形局部 / 提取结果（一行排列）
用法：python extract_polaroids.py <图片路径> [--output <输出路径>]
"""

import argparse
import gc
import sys
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体（Windows 按优先级尝试）
_CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "DengXian", "FangSong", "KaiTi", "STSong"]
_fonts = {f.name for f in fm.fontManager.ttflist}
for _font in _CJK_CANDIDATES:
    if _font in _fonts:
        plt.rcParams["font.family"] = _font
        plt.rcParams["axes.unicode_minus"] = False
        break
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor
from quadrilateral_fitter import QuadrilateralFitter
from scipy.spatial import ConvexHull
from modelscope import snapshot_download

# ===== 工具函数（与 pipeline 保持一致） =====

def _points_inside_quad(points, quad):
    pts, q = np.asarray(points, dtype=np.float64), np.asarray(quad, dtype=np.float64)
    signed_area = sum(q[i, 0] * q[(i + 1) % 4, 1] - q[(i + 1) % 4, 0] * q[i, 1] for i in range(4))
    is_cw = signed_area > 0
    inside = np.ones(len(pts), dtype=bool)
    for i in range(4):
        p1, p2 = q[i], q[(i + 1) % 4]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        cross = dx * (pts[:, 1] - p1[1]) - dy * (pts[:, 0] - p1[0])
        inside &= (cross >= 0) if is_cw else (cross <= 0)
    return inside


def _shrink_quad(vertices, factor=0.03):
    c = np.asarray(vertices, dtype=np.float64).mean(axis=0)
    return np.asarray(vertices) + factor * (c - vertices)


def fit_quadrilateral(points):
    """从轮廓点拟合四边形（ConvexHull + QuadrilateralFitter）"""
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]
    p1 = hull_pts[np.argmin(hull_pts[:, 0])]
    p2 = hull_pts[np.argmax(np.sum((hull_pts - p1) ** 2, axis=1))]
    v = p2 - p1
    signed = (v[0] * (hull_pts[:, 1] - p1[1]) - v[1] * (hull_pts[:, 0] - p1[0])) / np.linalg.norm(v)
    approx = np.array([p1, p2, hull_pts[np.argmax(signed)], hull_pts[np.argmin(signed)]])
    approx = approx[np.argsort(np.arctan2(
        approx[:, 1] - approx.mean(axis=0)[1],
        approx[:, 0] - approx.mean(axis=0)[0]
    ))]
    shrunk = _shrink_quad(approx, 0.03)
    inside = _points_inside_quad(points, shrunk)
    exterior = points[~inside]
    if len(exterior) < 4:
        return approx  # 退化回初始近似
    fitter = QuadrilateralFitter(polygon=exterior)
    vertices = np.array(fitter.fit(), dtype=np.float64)
    vertices = vertices[np.argsort(np.arctan2(
        vertices[:, 1] - vertices.mean(axis=0)[1],
        vertices[:, 0] - vertices.mean(axis=0)[0]
    ))]
    return vertices


# ===== 加载 SAM3 模型 =====
print("正在加载 SAM3 模型...")
model_dir = snapshot_download('facebook/sam3', cache_dir='./sam3_model')
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir_local = "./sam3_model/facebook/sam3"
model = Sam3Model.from_pretrained(model_dir_local).to(device)
processor = Sam3Processor.from_pretrained(model_dir_local)
print(f"模型已加载，设备: {device}")


# ===== 命令行参数 =====
parser = argparse.ArgumentParser(description="从图片中提取所有拍立得相片")
parser.add_argument("image", help="输入图片路径")
parser.add_argument("--output", "-o", default=None,
                    help="输出图片路径（默认 outs/{原文件名}_extracted.png）")
parser.add_argument("--threshold", "-t", type=float, default=0.4,
                    help="SAM3 检测阈值 (默认 0.4)")
args = parser.parse_args()

img_path = args.image
threshold = args.threshold

# ===== 加载图片 =====
print(f"\n加载图片: {img_path}")
image = Image.open(img_path).convert("RGB")
image = ImageOps.exif_transpose(image)
img_np = np.array(image)
h_orig, w_orig = img_np.shape[:2]

# ===== 纸框检测 =====
print("检测拍立得纸框...")
inputs_sam = processor(images=image, text="polaroid photo paper frame",
                       return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs_sam)
results = processor.post_process_instance_segmentation(
    outputs, threshold=threshold, mask_threshold=0.5,
    target_sizes=[image.size[::-1]]
)[0]
paper_masks = [m.cpu().numpy() for m in results["masks"]]
num_found = len(paper_masks)
del inputs_sam, outputs, results
torch.cuda.empty_cache()
print(f"检测到 {num_found} 张拍立得")

if num_found == 0:
    print("未检测到任何拍立得纸框，退出。")
    sys.exit(1)

# ===== 对每张拍立得拟合四边形 + 透视矫正 =====
POLAROID_W, POLAROID_H = 800, 1272
COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 120, 255),
    (255, 165, 0), (200, 0, 200), (0, 200, 200),
    (255, 80, 80), (80, 255, 80), (80, 80, 255),
    (255, 200, 0), (255, 0, 200), (0, 255, 200),
]

polaroids = []  # (vertices, rectified_image)

for pidx, paper_mask in enumerate(paper_masks):
    contours, _ = cv2.findContours(
        paper_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        print(f"  拍立得 {pidx + 1}: 轮廓为空，跳过")
        continue

    points = np.vstack([c.reshape(-1, 2) for c in contours])
    try:
        vertices = fit_quadrilateral(points)
    except Exception as e:
        print(f"  拍立得 {pidx + 1}: 四边形拟合失败 ({e})，跳过")
        continue

    # 透视矫正
    src = vertices.astype(np.float32)
    dst = np.array([[0, 0], [POLAROID_W, 0], [POLAROID_W, POLAROID_H], [0, POLAROID_H]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(img_np, M, (POLAROID_W, POLAROID_H))

    polaroids.append((vertices, rectified))
    print(f"  拍立得 {pidx + 1}: 顶点 "
          f"[{' '.join(f'({int(v[0])},{int(v[1])})' for v in vertices)}]")

print(f"成功提取 {len(polaroids)} 张拍立得")


# ===== 绘制三栏可视化 =====
n = len(polaroids)
fig = plt.figure(figsize=(max(18, n * 3.5), 9))

# --- 栏 1: 原图（无标注） ---
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img_np)
ax1.set_title("Original", fontsize=14, fontweight="bold")
ax1.axis("off")

# --- 栏 2: 四边形（原图上绘制每个四边形 + 编号） ---
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(img_np)
for i, (vertices, _) in enumerate(polaroids):
    color = [c / 255 for c in COLORS[i % len(COLORS)]]
    poly = np.vstack([vertices, vertices[0:1]])  # 闭合
    ax2.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.2, edgecolor=color, linewidth=3)
    ax2.scatter(vertices[:, 0], vertices[:, 1], c=[color], s=70, zorder=5, edgecolors="white", linewidths=1)
    # 标注编号
    cx, cy = vertices.mean(axis=0)
    ax2.text(cx, cy, f"#{i + 1}", color="white", fontsize=14, fontweight="bold",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9))
ax2.set_title(f"Quadrilaterals ({n})", fontsize=14, fontweight="bold")
ax2.axis("off")

# --- 栏 3: 提取结果（一行排列） ---
ax3 = fig.add_subplot(1, 3, 3)
if n == 1:
    ax3.imshow(polaroids[0][1])
    ax3.set_title("Extracted", fontsize=14, fontweight="bold")
else:
    # 按统一高度缩放，水平拼接
    target_h = 400
    resized = []
    for _, rect in polaroids:
        h, w = rect.shape[:2]
        new_w = int(w * target_h / h)
        resized.append(cv2.resize(rect, (new_w, target_h)))
    # 在底端对齐拼接（统一高度，不同宽度）
    concat = np.hstack(resized)
    ax3.imshow(concat)
ax3.set_title(f"Extracted ({n})", fontsize=14, fontweight="bold")
ax3.axis("off")

plt.tight_layout(pad=2)

# ===== 保存 =====
import os
if args.output:
    out_path = args.output
else:
    os.makedirs("outs", exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join("outs", f"{base}_extracted.png")

fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\n✓ 已保存: {out_path}")

gc.collect()
torch.cuda.empty_cache()
