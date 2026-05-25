"""标记规则拒绝的列 — 在已有 progression PNG 上叠加红色标记"""
import csv, os, glob, numpy as np, cv2

CSV_PATH = "outs/_scribble_dynamic_data.csv"
PNG_DIR = "outs"
N_COLS = 8

def apply_rule(r):
    n_cur = int(r['n_cur']) if r['n_cur'] else 0
    ar_t = float(r['area_ratio']) if r['area_ratio'] else 0
    ar_b = float(r['ar_border']) if r['ar_border'] else 0
    mr_b = float(r['mr_border']) if r['mr_border'] else 1
    ar_i = float(r['ar_image']) if r['ar_image'] else 0
    mr_i = float(r['mr_image']) if r['mr_image'] else 1
    if n_cur > 0 and ar_t < 0.015: return 'reject'
    if (ar_b > 0.15 and (ar_b > 0.50 or mr_b > 0.50)) or (ar_i > 0.15 and (ar_i > 0.50 or mr_i > 0.50)): return 'accept'
    if (ar_b > 0.015 and mr_b < 0.80) or (ar_i > 0.015 and mr_i < 0.80): return 'reject'
    return 'accept'

# Load CSV
rows = []
with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        rows.append(r)

# Build per-polaroid ordered thresholds and decisions
# { (img, pol): [(th, is_base, row_data), ...] }
pol_data = {}
for r in rows:
    pk = (r['image'], int(r['polaroid']))
    th = r['from_th'] if r['is_base'] == '1' else r['to_th']
    is_base = r['is_base'] == '1'
    pol_data.setdefault(pk, []).append((th, is_base, r))

# For each polaroid, find the stop column (first rejected transition)
stop_cols = {}  # {(img, pol): (stop_index, None or row_data)}
for pk, entries in pol_data.items():
    entries.sort(key=lambda x: float(x[0]), reverse=True)  # descending threshold
    stop_idx = None
    for i, (th, is_base, r) in enumerate(entries):
        if is_base:
            continue
        decision = apply_rule(r)
        if decision == 'reject':
            stop_idx = i
            break
    if stop_idx is None:
        stop_idx = N_COLS  # all accepted
    stop_cols[pk] = stop_idx

# Process each PNG
png_files = sorted(glob.glob(os.path.join(PNG_DIR, "_scribble_dynamic_*.png")))
for png_path in png_files:
    base = os.path.splitext(os.path.basename(png_path))[0]
    img_name = base.replace("_scribble_dynamic_", "")
    print(f"Processing {img_name}...")
    
    img = cv2.imread(png_path)
    if img is None:
        print(f"  Cannot read {png_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    # Determine n_rows
    n_rows = 1
    for (imn, pol), idx in stop_cols.items():
        if imn == img_name:
            n_rows = max(n_rows, pol)
    
    col_w = W // N_COLS
    row_h = H // n_rows if n_rows > 0 else H
    
    # Mark rejected columns (from stop_idx to N_COLS-1) with red border
    for pi in range(n_rows):
        pk = (img_name, pi+1)
        si = stop_cols.get(pk, N_COLS)
        if si < N_COLS:
            # Mark from stop column onward with red overlay
            for c in range(si, N_COLS):
                x1 = c * col_w
                y1 = pi * row_h
                x2 = (c+1) * col_w - 1
                y2 = (pi+1) * row_h - 1
                # Red border
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # Draw a bold "R" at the stop column
            cx = si * col_w + col_w // 2
            cy = pi * row_h + row_h // 2
            cv2.putText(img, "R", (cx-20, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
    
    # Save
    out_path = os.path.join(PNG_DIR, f"_marked_{img_name}.png")
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out_path}")

print("\nDone! Red border = rule-rejected columns. Open marked PNGs to review.")
