"""
自动评分脚本
基于用户手评的最高分作为参考，用 IoU + FP/FN + 碎片/大面积惩罚
为未评分的 mask cell 自动打分。

用法：
  python auto_score.py          # 对所有有用户评分的网格自动补全
  python auto_score.py --dry    # 预览模式，不写入 scores.json
"""

import json
import os
import sys
import glob
import numpy as np
import cv2

# ===== 配置 =====
PROMPTS = [
    "ink", "handwriting", "writing", "text", "number", "scribble",
    "black ink", "the ink marks on the photo",
    "pen writing on the image", "ballpoint pen handwriting on the photo",
    "handwritten signature and notes on polaroid",
    "hand-drawn marks and writing on the image",
    "the handwritten words and marks on this polaroid picture",
    "all ink, pen marks, and handwriting visible on the photo",
]
THRESHOLDS = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
N_ROWS = len(PROMPTS)
N_COLS = len(THRESHOLDS)
SCORES_FILE = "outs/scores.json"
GOLDEN_THRESHOLD = 0.95  # precision × recall ≥ 此值 → 直接给 max_score

# 评分参数
WINDOW_SIZE = 127          # 滑动窗口边长（图像高度的 10%）
MAX_WINDOW_DENSITY = 0.85  # 窗口内密度 > 此值 → 灾难性失败
MAX_FRAGMENTS_OK = 15      # ≤15 个连通分量不扣分
MAX_FRAGMENTS_BAD = 60     # ≥60 个连通分量扣满
EDGE_WEIGHT = 0.05         # 边缘模糊扣分权重（相对 max_score）
FRAG_WEIGHT = 0.03         # 碎片扣分权重
CATASTROPHIC_SCORE = -100  # 大面积 FP 的分数


def load_scores():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_scores(scores):
    os.makedirs("outs", exist_ok=True)
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


def find_data_files():
    """返回所有有对应 data.npz 的 base 名"""
    npz_files = glob.glob("outs/*_data.npz")
    bases = sorted([os.path.splitext(os.path.basename(f))[0].replace("_data", "") for f in npz_files])
    return bases


def get_cell_score(scores, base, row, col):
    entry = scores.get(base, {})
    prompt = PROMPTS[row]
    thresh = str(THRESHOLDS[col])
    return entry.get("prompts", {}).get(prompt, {}).get("thresholds", {}).get(thresh)


def set_cell_score(scores, base, row, col, value, auto=False):
    if base not in scores:
        scores[base] = {"prompts": {}, "notes": ""}
    entry = scores[base]
    prompt = PROMPTS[row]
    if prompt not in entry["prompts"]:
        entry["prompts"][prompt] = {"thresholds": {}}
    key = "auto" if auto else "thresholds"
    if key not in entry["prompts"][prompt]:
        entry["prompts"][prompt][key] = {}
    entry["prompts"][prompt][key][str(THRESHOLDS[col])] = value


def max_window_density(mask):
    """滑动窗口检测：返回 127x127 窗口内的最大像素密度。
    如果任何窗口内密度 > 80%，说明存在大块密集 FP。"""
    if not mask.any():
        return 0.0
    H, W = mask.shape
    win = WINDOW_SIZE
    max_dens = 0.0
    for y in range(0, H - win, win // 2):
        for x in range(0, W - win, win // 2):
            d = mask[y:y+win, x:x+win].mean()
            if d > max_dens:
                max_dens = d
    return max_dens


def count_components(mask):
    """返回连通分量数量"""
    if not mask.any():
        return 0
    nlabels, _, _, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return nlabels - 1  # 减去背景


def edge_fuzziness(mask):
    """评估边缘模糊程度。对 mask 做轻微高斯模糊后重二值化，
    与原 mask 比较 IoU。返回值 0=边缘锐利, 1=边缘极模糊。"""
    if not mask.any():
        return 0.0
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), sigmaX=1.0)
    rethresh = (blurred >= 0.5).astype(np.uint8)
    inter = np.sum((mask & rethresh))
    union = np.sum((mask | rethresh))
    if union == 0:
        return 0.0
    edge_iou = inter / union
    return 1.0 - edge_iou  # 0=完美, 越大越模糊


def score_one_mask(mask, ref_mask, max_score, total_pixels, user_top_masks=None):
    """
    评分 = max_score × precision × recall − 边缘/碎片惩罚
    黄金区：与任一用户手标最高分 mask 的 precision×recall ≥ 0.95 → 直接 max_score
    """
    # 黄金区：与用户手标的任一 top mask 高度一致
    if user_top_masks:
        for utm in user_top_masks:
            inter = np.sum(mask & utm)
            if inter > 0:
                p = inter / max(np.sum(mask), 1)
                r = inter / max(np.sum(utm), 1)
                if p * r >= GOLDEN_THRESHOLD:
                    return max_score
    # 大面积 FP 检测（滑动窗口密度）
    if max_window_density(mask) > MAX_WINDOW_DENSITY:
        return CATASTROPHIC_SCORE

    intersection = np.sum(mask & ref_mask)
    mask_area = np.sum(mask)
    ref_area = np.sum(ref_mask)

    if ref_area == 0:
        return max_score if mask_area == 0 else max(0, max_score * (1 - mask_area / total_pixels * 10))

    if mask_area == 0:
        # 没检测到任何东西，但参考有墨迹 → recall=0 → 0 分
        return 0.0

    precision = intersection / mask_area   # 惩罚 FP
    recall = intersection / ref_area        # 惩罚 FN

    # 黄金区：与参考几乎一致 → 直接给最高分
    if precision * recall >= GOLDEN_THRESHOLD:
        return max_score

    # 边缘模糊惩罚
    edge_penalty = min(1.0, edge_fuzziness(mask) * 5)

    # 碎片惩罚
    n_comp = count_components(mask)
    frag_penalty = max(0.0, min(1.0, (n_comp - MAX_FRAGMENTS_OK) / (MAX_FRAGMENTS_BAD - MAX_FRAGMENTS_OK)))

    score = max_score * precision * recall - max_score * (
        EDGE_WEIGHT * edge_penalty
        + FRAG_WEIGHT * frag_penalty
    )

    score = max(CATASTROPHIC_SCORE + 50, min(max_score, score))
    return round(score, 2)


def build_reference(scores, base, masks):
    """
    根据用户已有评分构建参考 mask。
    返回 (ref_mask, max_score, top_cells, user_top_masks)
      ref_mask: 最高分 cells 的交集（用于基础评分）
      user_top_masks: 用户手标最高分的每个独立 mask（用于黄金区匹配）
    """
    # 收集用户手动评分的 cell（从 thresholds 键，而非 auto）
    user_cells = []
    auto_cells = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            entry = scores.get(base, {}).get("prompts", {}).get(PROMPTS[r], {})
            val = entry.get("thresholds", {}).get(str(THRESHOLDS[c]))
            if val is not None:
                user_cells.append((r, c, float(val)))
            val2 = entry.get("auto", {}).get(str(THRESHOLDS[c]))
            if val2 is not None:
                auto_cells.append((r, c, float(val2)))

    all_scored = user_cells + auto_cells
    if not all_scored:
        return None, None, [], []

    max_score = max(v for _, _, v in all_scored)
    top_cells = [(r, c) for r, c, v in all_scored if v == max_score]

    # 用户手标的 top masks（仅用户手标，用于黄金区匹配）
    user_top_masks = []
    if user_cells:
        user_max = max(v for _, _, v in user_cells)
        for r, c, v in user_cells:
            if v == user_max:
                user_top_masks.append(masks[r, c].copy())

    # 如果存在 10 分 → 以它为唯一 reference
    ten_cells = [(r, c) for r, c, v in all_scored if v == 10.0]
    if ten_cells:
        ref_mask = masks[ten_cells[0][0], ten_cells[0][1]].copy()
        return ref_mask, max_score, ten_cells, user_top_masks

    # 否则取最高分 cells 的交集
    ref_mask = masks[top_cells[0][0], top_cells[0][1]].copy()
    for r, c in top_cells[1:]:
        ref_mask &= masks[r, c]

    return ref_mask, max_score, top_cells, user_top_masks


def auto_score_grid(scores, base, masks, dry_run=False):
    """对一个网格自动补全评分，返回修改数量"""
    ref_mask, max_score, ref_cells, user_top_masks = build_reference(scores, base, masks)

    if ref_mask is None:
        return 0

    total_pixels = masks.shape[2] * masks.shape[3]
    changes = 0

    for r in range(N_ROWS):
        for c in range(N_COLS):
            existing = get_cell_score(scores, base, r, c)
            if existing is not None:
                continue

            cell_mask = masks[r, c]
            score = score_one_mask(cell_mask, ref_mask, max_score, total_pixels, user_top_masks)

            if not dry_run:
                set_cell_score(scores, base, r, c, score, auto=True)
            changes += 1

    return changes


def main():
    dry_run = "--dry" in sys.argv

    scores = load_scores()
    bases = find_data_files()

    print(f"找到 {len(bases)} 个数据文件")
    if dry_run:
        print("*** 预览模式，不会写入 scores.json ***")

    total_cells = 0
    skipped_grids = 0

    for base in bases:
        # 加载 masks
        data_path = f"outs/{base}_data.npz"
        data = np.load(data_path)
        masks = data["masks"]  # (14, 8, H, W)

        changed = auto_score_grid(scores, base, masks, dry_run)
        if changed == 0:
            skipped_grids += 1
            continue

        total_cells += changed
        ref_mask, max_score, ref_cells, user_top_masks = build_reference(scores, base, masks)
        print(f"  {base}: +{changed} cells, ref={len(ref_cells)} top@{max_score} (user_tops={len(user_top_masks)})")

    if not dry_run and total_cells > 0:
        save_scores(scores)

    print(f"\n总计: {total_cells} cells 评分, {skipped_grids} grids 跳过（无用户评分）")
    if not dry_run:
        print(f"已保存到 {SCORES_FILE}")


# ===== 自适应阈值选取 =====

def select_threshold_by_color(base, prompt_name, data_path=None):
    """
    基于颜色一致性的自适应阈值选取。
    用该 prompt 下手标最高分的 mask 提取墨迹颜色分布，
    对更低阈值的新增像素做颜色判定，颜色不匹配的视为 FP。
    返回最佳 threshold 和信心度。
    
    参数:
        base: 网格名，如 "IMG_7562_p1"
        prompt_name: prompt 名，如 "handwriting"
        data_path: .npz 路径，默认自动查找
    
    返回: (best_threshold, confidence) 或 (None, 0) 如果不可用
    """
    import json as _json
    scores = load_scores()
    if base not in scores:
        return None, 0

    if data_path is None:
        data_path = f"outs/{base}_data.npz"
    if not os.path.exists(data_path):
        return None, 0

    data = np.load(data_path)
    if "wb_image" not in data:
        return None, 0  # 旧数据没有 wb_image

    wb = data["wb_image"]
    masks = data["masks"]
    pi = list(data["prompts"]).index(prompt_name)
    threshs = data["thresholds"]

    # 找用户手标的最高分
    entry = scores[base]["prompts"].get(prompt_name, {})
    user_best_val = -999
    user_best_t = None
    for th_str, val in entry.get("thresholds", {}).items():
        v = float(val)
        if v > user_best_val:
            user_best_val = v
            user_best_t = float(th_str)

    if user_best_t is None:
        return None, 0

    best_ti = list(threshs).index(user_best_t)
    best_mask = masks[pi, best_ti]

    if best_mask.sum() < 100:
        return user_best_t, 1.0  # 墨迹太少，直接用用户选的

    # 提取墨迹颜色分布（RGB 均值 + 标准差）
    ink_pixels = wb[best_mask]
    ink_mean = ink_pixels.mean(axis=0)
    ink_std = ink_pixels.std(axis=0) + 1e-6  # 防止除零

    # 对更低阈值，检查新增像素的颜色一致性
    color_match_ratios = []
    for ti in range(best_ti + 1, len(threshs)):
        cur_mask = masks[pi, ti]
        new_pixels = cur_mask & ~best_mask
        if new_pixels.sum() < 50:
            color_match_ratios.append(1.0)  # 新增太少，视为全部匹配
            continue

        new_colors = wb[new_pixels]
        # Mahalanobis-like: 每个通道的 z-score，取平均
        z_scores = np.abs((new_colors - ink_mean) / ink_std)
        avg_z = z_scores.mean(axis=1)
        match_ratio = (avg_z < 2.5).mean()  # z < 2.5 视为颜色匹配
        color_match_ratios.append(match_ratio)

    # 找到颜色匹配率骤降的拐点
    # 如果所有阈值新增像素都与墨迹颜色一致 → 所有阈值都好
    # 如果某阈值新增像素有 >30% 颜色不匹配 → 该阈值及以下都不可靠
    for i, ratio in enumerate(color_match_ratios):
        if ratio < 0.70:  # 新增像素中 <70% 匹配墨迹颜色
            # 回退到上一个好阈值
            if i > 0:
                return float(threshs[best_ti + i]), color_match_ratios[i - 1]
            else:
                return user_best_t, 1.0

    # 所有阈值都通过颜色检验 → 用最低的
    return float(threshs[-1]), color_match_ratios[-1] if color_match_ratios else 1.0


def auto_select_best(base, data_path=None):
    """对一张拍立得自动选出最佳 (prompt, threshold)。"""
    best_pn, best_t, best_conf = None, None, 0
    for pn in ["handwriting", "scribble"]:
        t, conf = select_threshold_by_color(base, pn, data_path)
        if t is not None and conf > best_conf:
            best_pn, best_t, best_conf = pn, t, conf
    return best_pn, best_t, best_conf


if __name__ == "__main__":
    main()
