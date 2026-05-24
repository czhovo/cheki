"""
拍立得流水线批处理脚本
对 imgs/ 文件夹中每张图片执行完整流水线：
  纸框矫正 → 图像区域提取 → 白平衡 → 墨水检测网格搜索
支持单张图片包含多张拍立得相片（自动检测所有纸框并分别处理）
输出：
  outs/{原文件名}_p{N}_grid.png —— 网格图（每行左侧原图 + jet mask 列）
  outs/{原文件名}_p{N}_data.npz  —— 所有 mask 数据文件
"""

import os
import gc
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor
from quadrilateral_fitter import QuadrilateralFitter
from scipy.spatial import ConvexHull
from modelscope import snapshot_download
import tqdm

# ===== 工具函数 =====
def _points_inside_quad(points, quad):
    pts, q = np.asarray(points, dtype=np.float64), np.asarray(quad, dtype=np.float64)
    signed_area = sum(q[i,0]*q[(i+1)%4,1] - q[(i+1)%4,0]*q[i,1] for i in range(4))
    is_cw = signed_area > 0
    inside = np.ones(len(pts), dtype=bool)
    for i in range(4):
        p1, p2 = q[i], q[(i+1)%4]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        cross = dx*(pts[:,1]-p1[1]) - dy*(pts[:,0]-p1[0])
        inside &= (cross >= 0) if is_cw else (cross <= 0)
    return inside

def _shrink_quad(vertices, factor=0.03):
    c = np.asarray(vertices, dtype=np.float64).mean(axis=0)
    return np.asarray(vertices) + factor * (c - vertices)

def _rectify_quality(vertices):
    w_top = np.linalg.norm(vertices[1]-vertices[0])
    w_bot = np.linalg.norm(vertices[2]-vertices[3])
    h_left = np.linalg.norm(vertices[3]-vertices[0])
    h_right = np.linalg.norm(vertices[2]-vertices[1])
    w_r = min(w_top,w_bot)/max(w_top,w_bot)
    h_r = min(h_left,h_right)/max(h_left,h_right)
    ratio = ((h_left+h_right)/2)/((w_top+w_bot)/2)
    a_s = min(ratio,1.35)/max(ratio,1.35)
    return w_r * h_r * a_s

def fit_quadrilateral(points):
    """从轮廓点拟合四边形（ConvexHull + QuadrilateralFitter）"""
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]
    p1 = hull_pts[np.argmin(hull_pts[:, 0])]
    p2 = hull_pts[np.argmax(np.sum((hull_pts - p1)**2, axis=1))]
    v = p2 - p1
    signed = (v[0]*(hull_pts[:,1]-p1[1]) - v[1]*(hull_pts[:,0]-p1[0])) / np.linalg.norm(v)
    approx = np.array([p1, p2, hull_pts[np.argmax(signed)], hull_pts[np.argmin(signed)]])
    # 按绕中心的极角排序，避免蝴蝶形自交
    approx = approx[np.argsort(np.arctan2(
        approx[:, 1] - approx.mean(axis=0)[1],
        approx[:, 0] - approx.mean(axis=0)[0]
    ))]
    shrunk = _shrink_quad(approx, 0.03)
    inside = _points_inside_quad(points, shrunk)
    exterior = points[~inside]
    fitter = QuadrilateralFitter(polygon=exterior)
    vertices = np.array(fitter.fit(), dtype=np.float64)
    vertices = vertices[np.argsort(np.arctan2(
        vertices[:,1]-vertices.mean(axis=0)[1],
        vertices[:,0]-vertices.mean(axis=0)[0]
    ))]
    return vertices

# ===== 加载 SAM3 模型（全局单例） =====
print("正在下载/加载 SAM3 模型...")
model_dir = snapshot_download('facebook/sam3', cache_dir='./sam3_model')
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "./sam3_model/facebook/sam3"
model = Sam3Model.from_pretrained(model_dir).to(device)
processor = Sam3Processor.from_pretrained(model_dir)
print(f"模型已加载，设备: {device}")

# ===== 创建输出目录 =====
os.makedirs("outs", exist_ok=True)

# ===== 扫描输入图片 =====
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
img_files = sorted([
    f for f in os.listdir("imgs")
    if os.path.splitext(f)[1].lower() in SUPPORTED
])
print(f"找到 {len(img_files)} 张图片待处理")

# ===== 墨水检测配置 =====
PROMPTS = [0-
    "ink",
    "handwriting",
    "writing",
    "text",
    "number",
    "scribble",
    "black ink",
    "the ink marks on the photo",
    "pen writing on the image",
    "ballpoint pen handwriting on the photo",
    "handwritten signature and notes on polaroid",
    "hand-drawn marks and writing on the image",
    "the handwritten words and marks on this polaroid picture",
    "all ink, pen marks, and handwriting visible on the photo",
]
THRESHOLDS = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
MASK_THRESHOLD = 0.7  # SAM3 mask 二值化阈值，越大边缘越锐利

# ===== 逐张处理 =====
for idx, fname in enumerate(img_files):
    img_path = os.path.join("imgs", fname)
    base = os.path.splitext(fname)[0]
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{len(img_files)}] 处理: {fname}")
    print(f"{'='*60}")

    # ---------- 阶段 1: 加载图片 ----------
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    img_np = np.array(image)

    # ---------- 阶段 2: 纸框检测（支持多张拍立得） ----------
    print("  → 纸框检测...")
    inputs_sam = processor(images=image, text="polaroid photo paper frame",
                           return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs_sam)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.4, mask_threshold=MASK_THRESHOLD,
        target_sizes=[image.size[::-1]]
    )[0]
    paper_masks = [m.cpu().numpy() for m in results["masks"]]
    num_polaroids = len(paper_masks)
    del inputs_sam, outputs, results
    torch.cuda.empty_cache()
    print(f"    检测到 {num_polaroids} 张拍立得")

    if num_polaroids == 0:
        print(f"  ⚠ 未检测到拍立得纸框，跳过 {fname}")
        continue

    # ---------- 对每张拍立得分别处理 ----------
    for pidx, paper_mask in enumerate(paper_masks):
        p_label = f"p{pidx+1}"
        print(f"\n  --- [{p_label}] 拍立得 {pidx+1}/{num_polaroids} ---")

        try:
            contours, _ = cv2.findContours(
                paper_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                print(f"    ⚠ 纸框轮廓为空，跳过")
                continue
            points = np.vstack([c.reshape(-1, 2) for c in contours])
            paper_vertices = fit_quadrilateral(points)
        except Exception as e:
            print(f"    ⚠ 四边形拟合失败: {e}，跳过")
            continue

        src = paper_vertices.astype(np.float32)
        w_p, h_p = 800, 1272
        dst = np.array([[0,0],[w_p,0],[w_p,h_p],[0,h_p]], dtype=np.float32)
        M_paper = cv2.getPerspectiveTransform(src, dst)
        rectified_paper = cv2.warpPerspective(img_np, M_paper, (w_p, h_p))
        valid_mask = cv2.warpPerspective(
            np.ones(img_np.shape[:2], dtype=np.uint8), M_paper, (w_p, h_p)
        ).astype(bool)
        rectified_pil = Image.fromarray(rectified_paper)
        print(f"    纸框顶点: [{' '.join(str(row) for row in paper_vertices.astype(int))}]")

        # ---------- 阶段 3: 图像区域提取 ----------
        print(f"    → 图像区域提取...")
        inputs_sam = processor(images=rectified_pil,
                               text="the image area of the polaroid photo",
                               return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs_sam)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.4, mask_threshold=MASK_THRESHOLD,
            target_sizes=[rectified_pil.size[::-1]]
        )[0]
        if len(results["masks"]) == 0:
            print(f"    ⚠ 未检测到图像区域，跳过")
            del inputs_sam, outputs, results
            torch.cuda.empty_cache()
            continue
        area_mask = results["masks"][0].cpu().numpy()
        del inputs_sam, outputs, results
        torch.cuda.empty_cache()

        try:
            contours, _ = cv2.findContours(
                area_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            points = np.vstack([c.reshape(-1, 2) for c in contours])
            area_vertices = fit_quadrilateral(points)
        except Exception as e:
            print(f"    ⚠ 图像区域四边形拟合失败: {e}，跳过")
            continue

        EXPECTED = np.array([[59,113],[741,113],[741,1028],[59,1028]], dtype=np.float64)
        corner_err = np.mean(np.linalg.norm(area_vertices - EXPECTED, axis=1))
        quality = _rectify_quality(area_vertices)
        print(f"    图像区域顶点: [{' '.join(str(row) for row in area_vertices.astype(int))}]")
        print(f"    角点偏差: {corner_err:.1f} px | 几何质量: {quality:.3f}")

        # ---------- 阶段 4: 白平衡 ----------
        print(f"    → 白平衡...")
        h_img, w_img = rectified_paper.shape[:2]
        border_mask = np.ones((h_img, w_img), dtype=np.uint8)
        margin = 5
        inner = np.array([
            [area_vertices[0][0]-margin, area_vertices[0][1]-margin],
            [area_vertices[1][0]+margin, area_vertices[1][1]-margin],
            [area_vertices[2][0]+margin, area_vertices[2][1]+margin],
            [area_vertices[3][0]-margin, area_vertices[3][1]+margin]
        ], dtype=np.int32)
        cv2.fillPoly(border_mask, [inner], 0)
        border_mask = border_mask.astype(bool)

        img_arr = np.array(rectified_pil)
        is_bright = np.all(img_arr > 170, axis=2)
        is_neutral = np.std(img_arr.astype(np.float32), axis=2) < 25
        is_white = is_bright & is_neutral & border_mask

        blocks = []
        for y in range(0, h_img-32, 16):
            for x in range(0, w_img-32, 16):
                block = is_white[y:y+32, x:x+32]
                if np.sum(block) / 1024 > 0.8:
                    pixels = img_arr[y:y+32, x:x+32][block]
                    blocks.append({'x':x,'y':y,
                                   'mean':pixels.mean(axis=0),
                                   'var':pixels.var(axis=0).mean()})

        if blocks:
            blocks.sort(key=lambda b: b['var'])
            best = blocks[:10]
            ref_white = np.mean([b['mean'] for b in best], axis=0)
            target = 240.0
            gains = np.array([target/ref_white[0], target/ref_white[1], target/ref_white[2]])
            wb_image = np.clip(img_arr.astype(np.float32)*gains, 0, 255).astype(np.uint8)
            print(f"    白平衡增益: R={gains[0]:.3f} G={gains[1]:.3f} B={gains[2]:.3f}")
        else:
            wb_image = img_arr.copy()
            print(f"    未找到白色参考区域，跳过白平衡")

        # 缺失区域填充
        missing_mask = ~valid_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        missing_mask = cv2.dilate(missing_mask.astype(np.uint8), kernel).astype(bool)
        if missing_mask.any():
            wb_image[missing_mask] = (240, 240, 240)
            print(f"    填充缺失像素: {missing_mask.sum()}")

        # ---------- 阶段 5: 墨水检测网格搜索 ----------
        print(f"    → 墨水检测 ({len(PROMPTS)} prompts × {len(THRESHOLDS)} thresholds)...")
        wb_pil = Image.fromarray(wb_image)
        all_results = []

        for prompt in tqdm.tqdm(PROMPTS, desc=f"    {base}_{p_label}", leave=False):
            inputs_sam = processor(images=wb_pil, text=prompt,
                                   return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs_sam)

            for thresh in THRESHOLDS:
                results = processor.post_process_instance_segmentation(
                    outputs, threshold=thresh, mask_threshold=MASK_THRESHOLD,
                    target_sizes=[wb_pil.size[::-1]]
                )[0]
                num_masks = len(results["masks"])
                if num_masks == 0:
                    all_results.append({
                        'prompt':prompt, 'threshold':thresh,
                        'num_masks':0, 'has_large':False, 'ink_mask':None
                    })
                    continue
                masks = results["masks"].cpu().numpy()
                mask_areas = np.mean(masks, axis=(1,2))
                has_large = bool(np.any(mask_areas > 0.5))
                ink_mask = np.any(masks, axis=0)
                all_results.append({
                    'prompt':prompt, 'threshold':thresh,
                    'num_masks':num_masks, 'has_large':has_large,
                    'ink_mask':ink_mask
                })

            del inputs_sam, outputs, results
            torch.cuda.empty_cache()

        # ---------- 阶段 6: 生成并保存网格图 + 数据文件 ----------
        print(f"    → 生成网格图和数据文件...")
        n_prompts = len(PROMPTS)
        n_thresh = len(THRESHOLDS)
        n_cols = 1 + n_thresh  # 左侧原图 + 阈值列

        fig, axes = plt.subplots(n_prompts, n_cols,
                                 figsize=(n_cols * 2.2, n_prompts * 2.2))

        # 准备数据数组
        mask_array = np.zeros((n_prompts, n_thresh, *wb_image.shape[:2]), dtype=bool)
        counts_array = np.zeros((n_prompts, n_thresh), dtype=int)
        large_array = np.zeros((n_prompts, n_thresh), dtype=bool)

        for i, prompt in enumerate(PROMPTS):
            # 左侧原图
            ax_orig = axes[i, 0]
            ax_orig.imshow(wb_image)
            ax_orig.set_title(prompt, fontsize=7)
            ax_orig.axis('off')

            for j, thresh in enumerate(THRESHOLDS):
                ax = axes[i, j + 1]
                r = all_results[i * n_thresh + j]
                counts_array[i, j] = r['num_masks']
                large_array[i, j] = r['has_large']

                if r['ink_mask'] is not None:
                    mask_array[i, j] = r['ink_mask']
                    ax.imshow(r['ink_mask'], cmap='viridis')
                else:
                    ax.imshow(np.zeros_like(wb_image[:, :, 0]), cmap='viridis')

                title = f"t={thresh} n={r['num_masks']}"
                if r['has_large']:
                    title += " ⚠️"
                ax.set_title(title, fontsize=7)
                ax.axis('off')

        plt.tight_layout()

        out_path = os.path.join("outs", f"{base}_{p_label}_grid.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ 已保存: {out_path}")

        # 保存数据文件
        data_path = os.path.join("outs", f"{base}_{p_label}_data.npz")
        np.savez_compressed(data_path,
                            prompts=np.array(PROMPTS),
                            thresholds=np.array(THRESHOLDS),
                            masks=mask_array,
                            num_masks=counts_array,
                            has_large=large_array,
                            wb_image=wb_image)
        print(f"    ✓ 已保存: {data_path}")

        # ---------- 释放当前拍立得的显存 ----------
        del all_results, wb_image, rectified_paper, rectified_pil
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  CUDA: {torch.cuda.memory_allocated()/1024**2:.0f} MB / "
          f"{torch.cuda.memory_reserved()/1024**2:.0f} MB")

print(f"\n{'='*60}")
print(f"全部完成！共处理 {len(img_files)} 张图片，输出在 outs/ 目录")
print(f"{'='*60}")
