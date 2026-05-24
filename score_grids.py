"""
墨水检测网格图评分工具
交互式对 outs/ 中的 *_grid.png 逐个打分，用于后续挑选最佳 prompt 和 threshold。

用法：
  python score_grids.py                  # 交互式评分（从未评分的开始）
  python score_grids.py --list           # 列出所有网格及评分状态
  python score_grids.py --stats          # 按 prompt/threshold 汇总统计
  python score_grids.py <文件名前缀>     # 只评某张图，如 IMG_7562_p1

分数无限制，允许小数和负数，你爱打多少打多少。
"""

import os
import json
import sys
import glob

# ===== 配置（与 batch_pipeline.py 保持一致） =====
PROMPTS = [
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

SCORES_FILE = "outs/scores.json"


def load_scores():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_scores(scores):
    os.makedirs("outs", exist_ok=True)
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


def find_grids():
    """返回 outs/ 下所有 *_grid.png 的 base 名列表"""
    grids = glob.glob("outs/*_grid.png")
    bases = sorted([os.path.splitext(os.path.basename(g))[0].replace("_grid", "") for g in grids])
    return bases


def list_status():
    """列出所有网格及其评分状态"""
    scores = load_scores()
    bases = find_grids()
    if not bases:
        print("未找到任何 grid 文件")
        return

    done = 0
    for b in bases:
        if b in scores:
            n_scored = sum(
                1 for p in scores[b].get("prompts", {}).values()
                if "thresholds" in p and p["thresholds"]
            )
            total = len(PROMPTS)
            print(f"  [{b}]  ✓ {n_scored}/{total} prompts 已评分")
            done += 1
        else:
            print(f"  [{b}]  — 未评分")
    print(f"\n总计: {done}/{len(bases)} 已评分, {len(bases)-done} 待评")


def show_stats():
    """按 prompt 和 threshold 汇总统计"""
    scores = load_scores()
    if not scores:
        print("暂无评分数据")
        return

    # 收集所有评分
    prompt_scores = {p: [] for p in PROMPTS}
    thresh_scores = {t: [] for t in THRESHOLDS}

    for base, entry in scores.items():
        for pidx, prompt in enumerate(PROMPTS):
            pd = entry.get("prompts", {}).get(prompt, {}).get("thresholds", {})
            for tidx, thresh in enumerate(THRESHOLDS):
                key = str(thresh)
                if key in pd and pd[key] is not None:
                    s = pd[key]
                    prompt_scores[prompt].append(s)
                    thresh_scores[thresh].append(s)

    # 按 prompt 汇总
    print("\n===== 按 Prompt 汇总 =====")
    print(f"{'Prompt':<55} {'均值':>6} {'最高':>4} {'最低':>4} {'样本':>4}")
    print("-" * 75)
    for p in PROMPTS:
        vals = prompt_scores[p]
        if vals:
            print(f"{p:<55} {sum(vals)/len(vals):6.2f} {max(vals):4} {min(vals):4} {len(vals):4}")

    # 按 threshold 汇总
    print("\n===== 按 Threshold 汇总 =====")
    print(f"{'Threshold':>10} {'均值':>6} {'最高':>4} {'最低':>4} {'样本':>4}")
    print("-" * 30)
    for t in THRESHOLDS:
        vals = thresh_scores[t]
        if vals:
            print(f"{t:10.2f} {sum(vals)/len(vals):6.2f} {max(vals):4} {min(vals):4} {len(vals):4}")


def score_grid(base):
    """交互式对单张网格图评分"""
    scores = load_scores()

    if base not in scores:
        scores[base] = {"prompts": {}, "notes": ""}

    entry = scores[base]

    print(f"\n{'='*60}")
    print(f"评分: {base}_grid.png")
    print(f"输入 8 个分数对应阈值 [{', '.join(str(t) for t in THRESHOLDS)}]")
    print(f"命令: s=跳过  q=保存退出  d=删除重来")
    print(f"{'='*60}")

    for pidx, prompt in enumerate(PROMPTS):
        if prompt not in entry["prompts"]:
            entry["prompts"][prompt] = {"thresholds": {}}

        pd = entry["prompts"][prompt]
        existing = pd.get("thresholds", {})

        # 显示已有的评分
        existing_str = "  ".join(
            f"t={t}:{existing.get(str(t),'?')}" for t in THRESHOLDS
        )
        print(f"\n[{pidx+1}/{len(PROMPTS)}] \"{prompt}\"")
        if any(str(t) in existing for t in THRESHOLDS):
            print(f"  已有: {existing_str}")

        while True:
            raw = input(f"  → ").strip()

            if raw.lower() == "q":
                save_scores(scores)
                print(f"已保存到 {SCORES_FILE}")
                return False
            if raw.lower() == "s":
                break
            if raw.lower() == "d":
                pd["thresholds"] = {}
                print("  已清除，请重新输入")
                continue

            parts = raw.split()
            if len(parts) == 1:
                try:
                    val = float(parts[0])
                    for t in THRESHOLDS:
                        pd["thresholds"][str(t)] = val
                    save_scores(scores)
                    break
                except ValueError:
                    print(f"  请输入数字")
            elif len(parts) == len(THRESHOLDS):
                try:
                    vals = [float(v) for v in parts]
                    for t, v in zip(THRESHOLDS, vals):
                        pd["thresholds"][str(t)] = v
                    save_scores(scores)
                    break
                except ValueError:
                    print(f"  请输入 {len(THRESHOLDS)} 个数字，用空格分隔")
            else:
                print(f"  请输入 {len(THRESHOLDS)} 个数字 (每个阈值一个分数)，或单个数字 (统一分数)")

    # 全局备注
    print(f"\n备注 (回车跳过): ", end="")
    note = input().strip()
    if note:
        entry["notes"] = note

    save_scores(scores)
    print(f"✓ {base} 评分完成，已保存")
    return True


def main():
    bases = find_grids()

    if "--list" in sys.argv:
        list_status()
        return

    if "--stats" in sys.argv:
        show_stats()
        return

    # 过滤特定文件
    target = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            target = arg
            break

    if target:
        # 模糊匹配
        matches = [b for b in bases if target in b]
        if not matches:
            print(f"未找到匹配 '{target}' 的网格文件")
            print(f"可用文件: {', '.join(bases) if bases else '(无)'}")
            return
        for b in matches:
            score_grid(b)
        return

    # 交互模式：逐个评分未完成的
    scores = load_scores()
    pending = [b for b in bases if b not in scores or
               sum(1 for p in PROMPTS
                   if p in scores[b].get("prompts", {})
                   and scores[b]["prompts"][p].get("thresholds")) < len(PROMPTS)]

    if not pending:
        print("所有网格均已评分完毕！")
        return

    print(f"共 {len(bases)} 个网格，{len(pending)} 个待评分\n")
    for b in pending:
        cont = score_grid(b)
        if cont is False:
            print("已退出评分")
            break


if __name__ == "__main__":
    main()
