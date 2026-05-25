import csv, json

RATIOS = [0.87, 0.76, 0.66, 0.57, 0.50, 0.43, 0.38, 0.33, 0.29, 0.25]

rows = []
with open('outs/_scribble_dynamic_data.csv', 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        rows.append(r)

with open('outs/_stop_annotations.json', 'r', encoding='utf-8') as f:
    stops = json.load(f)

th_map = {}
for r in rows:
    pk = (r['image'], int(r['polaroid']))
    th_map.setdefault(pk, []).append(r['to_th'] if r['is_base']=='0' else r['from_th'])

truth = []
for img, polaroids in stops.items():
    for pol, stop_th in polaroids.items():
        pk = (img, int(pol))
        ths = th_map.get(pk, [])
        if not ths:
            continue
        try:
            si = ths.index(stop_th)
        except ValueError:
            continue
        for ti in range(1, si+1):
            truth.append((img, int(pol), ths[ti], 'accept'))
        if si + 1 < len(ths):
            truth.append((img, int(pol), ths[si+1], 'reject'))

n_accept = sum(1 for t in truth if t[3] == 'accept')
n_reject = sum(1 for t in truth if t[3] == 'reject')
print(f'Ground truth: {len(truth)} steps ({n_accept} accept, {n_reject} reject)')
print()

row_idx = {}
for r in rows:
    if r['is_base'] == '1':
        continue
    key = (r['image'], int(r['polaroid']), r['to_th'])
    row_idx[key] = r

# Rule: ar_tot<1% reject; ar_b>20% or ar_i>20% accept; (ar>1% and mr<80%) reject
def rule(r):
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

correct = fp = fn = 0
fp_list = []; fn_list = []
for img, pol, th, label in truth:
    r = row_idx.get((img, pol, th))
    if r is None: continue
    pred = rule(r)
    if pred == label:
        correct += 1
    elif pred == 'accept' and label == 'reject':
        fp += 1
        fp_list.append((img, pol, th, r))
    else:
        fn += 1
        fn_list.append((img, pol, th, r))

total = correct + fp + fn
print(f'Rule: {correct}/{total} ({correct/total*100:.1f}%)  FP={fp}  FN={fn}')
print()

if fp_list:
    print('FP (rule accept, user reject):')
    for img, pol, th, r in fp_list:
        ar_t = float(r['area_ratio']) if r['area_ratio'] else 0
        ar_b = float(r['ar_border']) if r['ar_border'] else 0
        mr_b = float(r['mr_border']) if r['mr_border'] else 1
        ar_i = float(r['ar_image']) if r['ar_image'] else 0
        mr_i = float(r['mr_image']) if r['mr_image'] else 1
        print(f'  {img} p{pol} t={th}: ar_tot={ar_t:.3f} ar_b={ar_b:.3f} mr_b={mr_b:.2f} ar_i={ar_i:.3f} mr_i={mr_i:.2f}')
    print()
if fn_list:
    print('FN (rule reject, user accept):')
    for img, pol, th, r in fn_list:
        ar_t = float(r['area_ratio']) if r['area_ratio'] else 0
        ar_b = float(r['ar_border']) if r['ar_border'] else 0
        mr_b = float(r['mr_border']) if r['mr_border'] else 1
        ar_i = float(r['ar_image']) if r['ar_image'] else 0
        mr_i = float(r['mr_image']) if r['mr_image'] else 1
        print(f'  {img} p{pol} t={th}: ar_tot={ar_t:.3f} ar_b={ar_b:.3f} mr_b={mr_b:.2f} ar_i={ar_i:.3f} mr_i={mr_i:.2f}')
    print()

print('--- Parameter sweep ---')
best = (0,0,0,0,0,0)
for ar_low in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
    for ar_high in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        for mr_th in [0.70, 0.75, 0.80, 0.85, 0.90]:
            c = fp_c = fn_c = 0
            for img, pol, th, label in truth:
                r = row_idx.get((img, pol, th))
                if r is None: continue
                ar_t = float(r['area_ratio']) if r['area_ratio'] else 0
                ar_b = float(r['ar_border']) if r['ar_border'] else 0
                mr_b = float(r['mr_border']) if r['mr_border'] else 1
                ar_i = float(r['ar_image']) if r['ar_image'] else 0
                mr_i = float(r['mr_image']) if r['mr_image'] else 1
                pred = 'accept'
                if ar_t < ar_low: pred = 'reject'
                elif ar_b > ar_high or ar_i > ar_high: pass
                elif (ar_b > ar_low and mr_b < mr_th) or (ar_i > ar_low and mr_i < mr_th): pred = 'reject'
                if pred == label: c += 1
                elif pred == 'accept': fp_c += 1
                else: fn_c += 1
            if c > best[3] or (c == best[3] and fp_c+fn_c < best[4]+best[5]):
                best = (ar_low, ar_high, mr_th, c, fp_c, fn_c)
print(f'Best: ar_low={best[0]:.3f} ar_high={best[1]:.2f} mr_th={best[2]:.2f} -> {best[3]}/{total} ({best[3]/total*100:.1f}%) fp={best[4]} fn={best[5]}')
