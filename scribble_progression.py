"""scribble 渐进阈值可视化 — 全部图片 (2σ 规则)"""
import torch, numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor
from quadrilateral_fitter import QuadrilateralFitter
from scipy.spatial import ConvexHull
from modelscope import snapshot_download

_fonts = {f.name for f in fm.fontManager.ttflist}
for _f in ["Microsoft YaHei", "SimHei"]:
    if _f in _fonts: plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False

# --- helpers ---
def _points_inside_quad(p, q):
    p, q = np.asarray(p,dtype=np.float64), np.asarray(q,dtype=np.float64)
    sa = sum(q[i,0]*q[(i+1)%4,1]-q[(i+1)%4,0]*q[i,1] for i in range(4))
    cw = sa > 0
    inside = np.ones(len(p), dtype=bool)
    for i in range(4):
        p1,p2 = q[i],q[(i+1)%4]
        cross = (p2[0]-p1[0])*(p[:,1]-p1[1])-(p2[1]-p1[1])*(p[:,0]-p1[0])
        inside &= (cross>=0) if cw else (cross<=0)
    return inside
def _shrink_quad(v, f=0.03):
    c = np.asarray(v,dtype=np.float64).mean(axis=0)
    return np.asarray(v)+f*(c-v)
def fit_quad(points):
    hull = ConvexHull(points); hp = points[hull.vertices]
    p1 = hp[np.argmin(hp[:,0])]; p2 = hp[np.argmax(np.sum((hp-p1)**2,axis=1))]
    v = p2-p1; s = (v[0]*(hp[:,1]-p1[1])-v[1]*(hp[:,0]-p1[0]))/np.linalg.norm(v)
    approx = np.array([p1,p2,hp[np.argmax(s)],hp[np.argmin(s)]])
    approx = approx[np.argsort(np.arctan2(approx[:,1]-approx.mean(0)[1],approx[:,0]-approx.mean(0)[0]))]
    shrunk = _shrink_quad(approx,0.03); inside = _points_inside_quad(points,shrunk)
    exterior = points[~inside]
    if len(exterior)<4: return approx
    fitter = QuadrilateralFitter(polygon=exterior)
    verts = np.array(fitter.fit(),dtype=np.float64)
    return verts[np.argsort(np.arctan2(verts[:,1]-verts.mean(0)[1],verts[:,0]-verts.mean(0)[0]))]

# --- load model ---
print("Loading SAM3...")
model_dir = snapshot_download('facebook/sam3', cache_dir='./sam3_model')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3Model.from_pretrained("./sam3_model/facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("./sam3_model/facebook/sam3")

THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.1]  # probe only
RATIOS = [0.79, 0.62, 0.49, 0.39, 0.31, 0.24, 0.19, 0.15]
N = len(RATIOS)

# 轮廓噪声过滤：新像素距上一轮 mask 边界 < OUTLINE_DIST 像素视为轮廓壳
OUTLINE_DIST = 5

# Fixed image area boundary (800×1272 rectified polaroid)
EXPECTED_AREA = np.array([[55,100],[745,100],[745,1022],[55,1022]], dtype=np.float64)

# --- data logging ---
import csv
csv_path = "outs/_scribble_dynamic_data.csv"
csv_f = open(csv_path, "w", newline="", encoding="utf-8-sig")
csv_w = csv.writer(csv_f)
csv_w.writerow(["image", "polaroid", "from_th", "to_th", "is_base",
                "n_cur", "n_prev", "n_new", "n_outline", "area_ratio",
                "match_z2", "match_rate_z2",
                "n_border_cur", "n_image_cur",
                "n_border_new", "n_image_new",
                "ar_border", "mr_border", "ar_image", "mr_image",
                "b_mean_r", "b_mean_g", "b_mean_b", "b_std_r", "b_std_g", "b_std_b",
                "i_mean_r", "i_mean_g", "i_mean_b", "i_std_r", "i_std_g", "i_std_b"])

# --- scan images ---
import os, glob
img_files = sorted(glob.glob("imgs/*.jpg")) + sorted(glob.glob("imgs/*.png"))
# img_files = [img_files[0]]
print(f"Found {len(img_files)} images")

for img_idx, img_path in enumerate(img_files):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n{'='*60}")
    print(f"[{img_idx+1}/{len(img_files)}] Processing: {img_name}")
    print(f"{'='*60}")
    
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    img_np = np.array(image)
    print(f"  Size: {img_np.shape}")
    
    # --- detect papers ---
    inputs = processor(images=image, text="polaroid photo paper frame", return_tensors="pt").to(device)
    with torch.no_grad(): outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(outputs, threshold=0.4, mask_threshold=0.5, target_sizes=[image.size[::-1]])[0]
    paper_masks = [m.cpu().numpy() for m in results["masks"]]
    num_polaroids = len(paper_masks)
    print(f"  Detected {num_polaroids} polaroids")
    del inputs, outputs, results
    torch.cuda.empty_cache()
    
    if num_polaroids == 0:
        print(f"  ⚠ No polaroids found, skipping")
        continue
    
    all_data = []
    
    for pi, pmask in enumerate(paper_masks):
        print(f"  --- Polaroid {pi+1}/{num_polaroids} ---")
        contours,_ = cv2.findContours(pmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        points = np.vstack([c.reshape(-1,2) for c in contours])
        pverts = fit_quad(points)
        
        # rectify
        src = pverts.astype(np.float32)
        w,h = 800, 1272
        dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        rect = cv2.warpPerspective(img_np, M, (w,h))
        rpil = Image.fromarray(rect)
        
        # image area detection via SAM3
        inputs_ia = processor(images=rpil, text="the image area of the polaroid photo", return_tensors="pt").to(device)
        with torch.no_grad(): outputs_ia = model(**inputs_ia)
        results_ia = processor.post_process_instance_segmentation(outputs_ia, threshold=0.4, mask_threshold=0.5, target_sizes=[rpil.size[::-1]])[0]
        if len(results_ia["masks"]) == 0:
            print(f"    ⚠ No image area found, using expected")
            averts = EXPECTED_AREA.copy()
        else:
            amask = results_ia["masks"][0].cpu().numpy()
            contours_ia, _ = cv2.findContours(amask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            points_ia = np.vstack([c.reshape(-1,2) for c in contours_ia])
            averts = fit_quad(points_ia)
        del inputs_ia, outputs_ia, results_ia
        torch.cuda.empty_cache()
        
        # Validate averts - fallback to EXPECTED_AREA if NaN or invalid
        if np.any(~np.isfinite(averts)) or np.any(averts < 0) or np.any(averts[:,0] >= w) or np.any(averts[:,1] >= h):
            averts = EXPECTED_AREA.copy()
        
        # white balance (uses detected image area boundary)
        h_img, w_img = rect.shape[:2]
        border_mask = np.ones((h_img, w_img), dtype=np.uint8)
        inner = np.array(averts + [[-5,-5],[5,-5],[5,5],[-5,5]], dtype=np.int32)
        cv2.fillPoly(border_mask, [inner], 0); border_mask = border_mask.astype(bool)
        img_arr = np.array(rpil)
        is_bright = np.all(img_arr > 170, axis=2)
        is_neutral = np.std(img_arr.astype(np.float32), axis=2) < 25
        is_white = is_bright & is_neutral & border_mask
        blocks = []
        for y in range(0, h_img-32, 16):
            for x in range(0, w_img-32, 16):
                block = is_white[y:y+32, x:x+32]
                if np.sum(block) / 1024 > 0.8:
                    px = img_arr[y:y+32, x:x+32][block]
                    blocks.append({'x':x,'y':y,'mean':px.mean(axis=0),'var':px.var(axis=0).mean()})
        if blocks:
            blocks.sort(key=lambda b: b['var'])
            best = blocks[:10]
            ref_white = np.mean([b['mean'] for b in best], axis=0)
            gains = np.array([240/ref_white[0], 240/ref_white[1], 240/ref_white[2]])
            wb = np.clip(img_arr.astype(np.float32)*gains, 0, 255).astype(np.uint8)
        else:
            wb = img_arr.copy()
        
        # scribble: probe for base threshold, then use derived thresholds
        wb_pil = Image.fromarray(wb)
        inputs = processor(images=wb_pil, text="scribble", return_tensors="pt").to(device)
        with torch.no_grad(): outputs = model(**inputs)
        
        # Probe
        base_th = None
        for th in THRESHOLDS:
            r = processor.post_process_instance_segmentation(outputs, threshold=th, mask_threshold=0.5, target_sizes=[wb_pil.size[::-1]])[0]
            if len(r["masks"]) > 0:
                ti = THRESHOLDS.index(th)
                base_th = THRESHOLDS[ti-1] if ti > 0 else 0.6338
                break
        if base_th is None:
            base_th = 0.1
            print("    ⚠ No scribble detected at any probe threshold, using 0.1")
        print(f"    Base threshold: {base_th:.3f}")
        
        threshs = [base_th * r for r in RATIOS]
        masks = []
        for th in threshs:
            r = processor.post_process_instance_segmentation(outputs, threshold=th, mask_threshold=0.5, target_sizes=[wb_pil.size[::-1]])[0]
            if len(r["masks"]) == 0:
                masks.append(np.zeros(wb.shape[:2], dtype=bool))
            else:
                m = r["masks"].cpu().numpy()
                masks.append(np.any(m, axis=0))
        del inputs, outputs, r
        torch.cuda.empty_cache()
        
        all_data.append((wb, masks, averts, threshs, base_th))
        print(f"    Done")
    
    if not all_data:
        print(f"  ⚠ No valid polaroids for {img_name}")
        continue
    
    # --- visualization for this image ---
    print(f"  Generating figure ({len(all_data)} polaroids)...")
    n_rows = len(all_data)
    fig, axes = plt.subplots(n_rows, N, figsize=(N*3.5, n_rows*4))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for pi, (wb, masks, averts, threshs, base_th) in enumerate(all_data):
        H, W = wb.shape[:2]
        # Image area mask from detected vertices
        img_area = np.zeros((H,W), dtype=np.uint8)
        cv2.fillPoly(img_area, [averts.astype(int)], 1)
        img_area = img_area.astype(bool)
        
        # Exclusion zone: 10px band centered on image area boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        img_dilated = cv2.dilate(img_area.astype(np.uint8), kernel, iterations=1).astype(bool)
        img_eroded = cv2.erode(img_area.astype(np.uint8), kernel, iterations=1).astype(bool)
        exclude_zone = img_dilated ^ img_eroded  # XOR = 10px band
        
        # Effective regions (exclude the boundary band)
        border_region = ~img_dilated  # outside dilated boundary
        img_region = img_eroded       # inside eroded boundary
        
        for ti in range(N):
            ax = axes[pi, ti]
            cur_mask = masks[ti]
            n_cur = cur_mask.sum()
            
            if ti == 0:
                overlay = (wb * 0.2).astype(np.uint8)
                if n_cur > 0:
                    overlay[cur_mask] = [0, 220, 0]
                overlay[exclude_zone] = [255, 255, 0]
                
                ax.imshow(overlay)
                ax.set_title(f"t={threshs[ti]:.3f} base  N={n_cur}", fontsize=7)
                ax.axis("off")
                cur_b = cur_mask & border_region
                cur_i = cur_mask & img_region
                csv_w.writerow([img_name, pi+1, threshs[ti], "", 1,
                    n_cur, 0, 0, 0, "",
                    "", "",
                    cur_b.sum(), cur_i.sum(), 0, 0,
                    "", "", "", "",
                    "", "", "", "", "", "",
                    "", "", "", "", "", ""])
            else:
                prev_mask = masks[ti-1]
                new_pixels_raw = cur_mask & ~prev_mask
                n_new_raw = int(new_pixels_raw.sum())
                n_prev = int(prev_mask.sum())
                
                # --- 轮廓噪声过滤 ---
                if n_prev > 0 and n_new_raw > 0:
                    prev_boundary = cv2.Canny(prev_mask.astype(np.uint8) * 255, 0, 1) > 0
                    dist = cv2.distanceTransform((~prev_boundary).astype(np.uint8), cv2.DIST_L2, 5)
                    outline_pixels = new_pixels_raw & (dist < OUTLINE_DIST)
                    n_outline = int(outline_pixels.sum())
                    new_pixels = new_pixels_raw & ~outline_pixels
                else:
                    outline_pixels = np.zeros_like(new_pixels_raw)
                    n_outline = 0
                    new_pixels = new_pixels_raw
                n_new = int(new_pixels.sum())
                area_ratio = n_new / n_cur if n_cur > 0 else 0
                
                prev_border = prev_mask & border_region
                prev_image = prev_mask & img_region
                new_border = new_pixels & border_region
                new_image = new_pixels & img_region
                n_b_new = new_border.sum(); n_i_new = new_image.sum()
                n_b_prev = prev_border.sum(); n_i_prev = prev_image.sum()
                
                z_b = np.array([])
                if n_b_new > 0 and n_b_prev > 0:
                    pb = wb[prev_border].astype(float)
                    pm, ps = pb.mean(axis=0), pb.std(axis=0)
                    nb = wb[new_border].astype(float)
                    z_b = np.abs((nb - pm) / (ps + 1e-6)).mean(axis=1)
                elif n_b_new > 0:
                    z_b = np.zeros(n_b_new)
                
                z_i = np.array([])
                if n_i_new > 0 and n_i_prev > 0:
                    pi_ink = wb[prev_image].astype(float)
                    pm_i, ps_i = pi_ink.mean(axis=0), pi_ink.std(axis=0)
                    ni = wb[new_image].astype(float)
                    z_i = np.abs((ni - pm_i) / (ps_i + 1e-6)).mean(axis=1)
                elif n_i_new > 0:
                    z_i = np.zeros(n_i_new)
                
                ar_b = n_b_new / (n_b_prev + n_b_new) if (n_b_prev + n_b_new) > 0 else 0
                m2_b = (z_b < 2.0).sum() if len(z_b) > 0 else 0
                mr_b = m2_b / n_b_new if n_b_new > 0 else 1.0
                
                ar_i = n_i_new / (n_i_prev + n_i_new) if (n_i_prev + n_i_new) > 0 else 0
                m2_i = (z_i < 2.0).sum() if len(z_i) > 0 else 0
                mr_i = m2_i / n_i_new if n_i_new > 0 else 1.0
                
                overlay = (wb * 0.2).astype(np.uint8)
                overlay[prev_mask] = [60, 60, 60]
                new_z2 = np.zeros(H*W, dtype=bool).reshape(H,W)
                new_bad = np.zeros(H*W, dtype=bool).reshape(H,W)
                if n_b_new > 0:
                    bm = z_b < 2.0
                    new_z2[new_border] = bm; new_bad[new_border] = ~bm
                if n_i_new > 0:
                    im = z_i < 2.0
                    new_z2[new_image] = im; new_bad[new_image] = ~im
                # 可视化：轮廓噪声用洋红色
                overlay[new_z2] = [0, 220, 0]
                overlay[new_bad] = [255, 40, 40]
                if n_outline > 0:
                    overlay[outline_pixels] = [255, 0, 255]  # 洋红 = 轮廓噪声
                overlay[exclude_zone] = [255, 255, 0]
                
                ax.imshow(overlay)
                title_color = "red" if (mr_b < 0.8 or mr_i < 0.8) else "black"
                ax.set_title(
                    f"t={threshs[ti-1]:.3f}→{threshs[ti]:.3f}  all={n_cur} new={n_new}({area_ratio:.1%})\n"
                    f"outline={n_outline}  "
                    f"B: area={ar_b:.1%} match={mr_b:.0%}({m2_b}/{n_b_new})\n"
                    f"I: area={ar_i:.1%} match={mr_i:.0%}({m2_i}/{n_i_new})",
                    fontsize=7, color=title_color)
                ax.axis("off")
                csv_w.writerow([img_name, pi+1,
                    f"{threshs[ti-1]:.4f}", f"{threshs[ti]:.4f}", 0,
                    n_cur, n_prev, n_new, n_outline,
                    f"{area_ratio:.4f}", m2_b+m2_i, f"{(m2_b+m2_i)/n_new if n_new>0 else 1:.4f}",
                    prev_border.sum() + n_b_new, prev_image.sum() + n_i_new,
                    n_b_new, n_i_new,
                    f"{ar_b:.4f}", f"{mr_b:.4f}", f"{ar_i:.4f}", f"{mr_i:.4f}",
                    "", "", "", "", "", "",
                    "", "", "", "", "", ""])
    
    plt.suptitle(f"{img_name} — scribble dynamic threshold\n"
                 "(gray=prev, green=z<2, red=z≥2)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_path = f"outs/_scribble_dynamic_{img_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

csv_f.close()
print(f"Data saved: {csv_path}")
print("\nAll done!")
