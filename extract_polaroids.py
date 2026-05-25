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
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [_font, "Segoe UI Symbol", "DejaVu Sans"]
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


# ===== 加载 SAM3 模型（仅在首次检测时） =====
# 模型在 --from-cache 时不会加载


# ===== 命令行参数 =====
parser = argparse.ArgumentParser(description="从图片中提取所有拍立得相片")
parser.add_argument("image", help="输入图片路径")
parser.add_argument("--output", "-o", default=None,
                    help="输出图片路径（默认 outs/{原文件名}_extracted.png）")
parser.add_argument("--from-cache", "-c", default=None,
                    help="从缓存的 .npz 文件加载 mask（跳过 SAM3）")
args = parser.parse_args()

img_path = args.image
threshold = 0.4  # default, only used if not from cache
cache_path = args.from_cache

import os
base = os.path.splitext(os.path.basename(img_path))[0]
os.makedirs("outs", exist_ok=True)
default_cache = os.path.join("outs", f"{base}_mask.npz")

# ===== 加载图片 =====
print(f"\n加载图片: {img_path}")
image = Image.open(img_path).convert("RGB")
image = ImageOps.exif_transpose(image)
img_np = np.array(image)
h_orig, w_orig = img_np.shape[:2]

if cache_path:
    # --- 从缓存加载 ---
    print(f"从缓存加载: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    polaroids = [(data["vertices"], None)]
    print(f"  顶点: {data['vertices']}")
else:
    # --- SAM3 检测 + 拟合 ---
    print("正在加载 SAM3 模型...")
    model_dir = snapshot_download('facebook/sam3', cache_dir='./sam3_model')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir_local = "./sam3_model/facebook/sam3"
    model = Sam3Model.from_pretrained(model_dir_local).to(device)
    processor = Sam3Processor.from_pretrained(model_dir_local)
    print(f"模型已加载，设备: {device}")
    
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
    
    # 筛选包含 (1600, 1200) 的 mask
    TARGET_POINT = np.array([1600, 1200])
    target_idx = None
    for i, m in enumerate(paper_masks):
        h, w = m.shape
        if 0 <= TARGET_POINT[1] < h and 0 <= TARGET_POINT[0] < w:
            if m[int(TARGET_POINT[1]), int(TARGET_POINT[0])]:
                target_idx = i
                break
    
    if target_idx is None:
        print(f"未找到包含点 ({TARGET_POINT[0]}, {TARGET_POINT[1]}) 的拍立得，退出。")
        sys.exit(1)
    
    print(f"选中拍立得 #{target_idx + 1} ，忽略其余 {num_found - 1} 个")
    paper_masks = [paper_masks[target_idx]]
    
    # 拟合四边形
    POLAROID_W, POLAROID_H = 800, 1272
    polaroids = []
    
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
        
        src = vertices.astype(np.float32)
        dst = np.array([[0, 0], [POLAROID_W, 0], [POLAROID_W, POLAROID_H], [0, POLAROID_H]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        rectified = cv2.warpPerspective(img_np, M, (POLAROID_W, POLAROID_H))
        polaroids.append((vertices, rectified))
        print(f"  拍立得 {pidx + 1}: 顶点 "
              f"[{' '.join(f'({int(v[0])},{int(v[1])})' for v in vertices)}]")
    
    print(f"成功提取 {len(polaroids)} 张拍立得")
    
    # 保存缓存
    if polaroids:
        np.savez(default_cache, vertices=polaroids[0][0], img_path=img_path)
        print(f"Mask 已缓存: {default_cache}")
    
    torch.cuda.empty_cache()


# ===== 手动微调四边形顶点 =====
for i, (vertices, _) in enumerate(polaroids):
    vertices[3] += [7, 6]  # 左下顶点：右移10、下移3

# ===== 绘制四边形标记图 =====
n = len(polaroids)
fig, ax = plt.figure(figsize=(img_np.shape[1]/100, img_np.shape[0]/100)), plt.gca()
ax.imshow(img_np)
ax.axis("off")

# Fixed log text
log_text = ("\u2192 加载 SAM3 模型...\n"
            "\u2192 检测拍立得纸框...\n"
            "检测到 37 张拍立得\n"
            "选中拍立得 #31 ，忽略其余 36 个\n"
            "拍立得 #31: 顶点 [ [1538,1153] [2334,962] [2708,2211] [1884,2439] ]\n"
            "\u25a0 成功提取 1 张拍立得")
ax.text(10, img_np.shape[0] - 10, log_text, color="black", fontsize=25,
        ha="left", va="bottom")

for i, (vertices, _) in enumerate(polaroids):
    color = [128/255, 0, 128/255]  # purple
    poly = np.vstack([vertices, vertices[0:1]])
    inner_vertices = _shrink_quad(vertices, 0.004)
    inner_poly = np.vstack([inner_vertices, inner_vertices[0:1]])
    ax.fill(inner_poly[:, 0], inner_poly[:, 1], color=color, alpha=0.35)
    ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=10)
    ax.scatter(vertices[:, 0], vertices[:, 1], c=[color], s=500, zorder=5, edgecolors="white", linewidths=3)
    
    # Label to the right of mask
    cx, cy = vertices.mean(axis=0)
    rightmost_x = vertices[:, 0].max()
    label_x = rightmost_x + 150
    label_y = cy
    
    ax.plot([cx, rightmost_x + 50, label_x], [cy, cy, label_y],
            color=color, linewidth=8, linestyle='-')
    
    label_text = "2025/5/25\nStardust Galaxy Party 星辰银河派对\n北京 Mask Stage\n@凌晨12点 mina"
    ax.text(label_x, label_y, label_text, color="white", fontsize=30, fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.85))

plt.tight_layout(pad=2)

# ===== 保存 =====
if args.output:
    out_path = args.output
else:
    out_path = os.path.join("outs", f"{base}_extracted.png")

fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white", pad_inches=0)
plt.close(fig)
print(f"\n✓ 已保存: {out_path}")

gc.collect()
torch.cuda.empty_cache()
