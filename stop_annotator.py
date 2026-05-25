"""scribble 动态阈值停止点标注工具

操作：
  点击子图 → 标记为该 polaroid 的停止点（画红色 X）
  再次点击同行更早的列 → 更新停止点
  ← → 切换图片    S 保存    Q 退出
"""

import os, sys, json, glob, csv
import tkinter as tk
from PIL import Image, ImageTk

N_COLS = 8  # dynamic threshold columns
PNG_DIR = "outs"
CSV_PATH = "outs/_scribble_dynamic_data.csv"
STOPS_FILE = "outs/_stop_annotations.json"

def load_csv_data():
    """Returns (detail, thresholds_map)
       detail: { (img_name, polaroid, to_th): {...} }
       thresholds_map: { (img_name, polaroid): [th1, th2, ..., th10] }
    """
    detail = {}
    th_map = {}  # (img, pol) -> list of threshold strings
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return detail, th_map
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        base_th = {}
        for row in reader:
            img = row["image"]; pol = int(row["polaroid"])
            pk = (img, pol)
            if row["is_base"] == "1":
                base_th[pk] = row["from_th"]
                th_map.setdefault(pk, []).append(row["from_th"])
            else:
                key = (img, pol, row["to_th"])
                detail[key] = {
                    "from_th": row["from_th"],
                    "area_ratio": float(row["area_ratio"]) if row["area_ratio"] else 0,
                    "match_rate_z2": float(row["match_rate_z2"]) if row["match_rate_z2"] else 0,
                    "n_cur": int(row["n_cur"]), "n_new": int(row["n_new"]),
                    "ar_border": float(row.get("ar_border", 0) or 0),
                    "mr_border": float(row.get("mr_border", 0) or 0),
                    "ar_image": float(row.get("ar_image", 0) or 0),
                    "mr_image": float(row.get("mr_image", 0) or 0),
                }
                th_map.setdefault(pk, []).append(row["to_th"])
    return detail, th_map

def load_stops():
    if os.path.exists(STOPS_FILE):
        with open(STOPS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_stops(stops):
    os.makedirs("outs", exist_ok=True)
    with open(STOPS_FILE, "w", encoding="utf-8") as f:
        json.dump(stops, f, ensure_ascii=False, indent=2)

def find_progression_pngs():
    return sorted(glob.glob(os.path.join(PNG_DIR, "_marked_*.png")))


class StopAnnotator:
    def __init__(self, png_files, csv_data, th_map, stops):
        self.png_files = png_files
        self.csv_data = csv_data
        self.th_map = th_map  # {(img, pol): [th1, th2, ...]}
        self.stops = stops    # {img_name: {polaroid_str: threshold_str}}
        self.idx = 0
        
        self.root = tk.Tk()
        self.root.title("Stop Point Annotator — click cell to mark stop")
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()-80}+0+0")
        
        # top bar
        bar = tk.Frame(self.root)
        bar.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(bar, text="← Prev (←)", command=self.prev_img).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Next (→)", command=self.next_img).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Save (S)", command=self.do_save).pack(side=tk.LEFT, padx=2)
        self.info_label = tk.Label(bar, text="", font=("", 11))
        self.info_label.pack(side=tk.LEFT, padx=20)
        self.status_label = tk.Label(bar, text="", fg="blue")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # scrollable canvas
        self.canvas = tk.Canvas(self.root, bg="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1*(e.delta//120), "units"))
        
        # key bindings
        self.root.bind("<Left>", lambda e: self.prev_img())
        self.root.bind("<Right>", lambda e: self.next_img())
        self.root.bind("s", lambda e: self.do_save())
        self.root.bind("S", lambda e: self.do_save())
        self.root.bind("q", lambda e: self.root.quit())
        self.root.bind("<Escape>", lambda e: self.root.quit())
        
        self.cell_rects = []  # [(x1,y1,x2,y2, row, col), ...] in canvas coords
        self.x_lines = []     # canvas line ids for X marks
        
        self.load_image()
        self.root.mainloop()
    
    def img_name(self):
        if not self.png_files:
            return None
        path = self.png_files[self.idx]
        base = os.path.splitext(os.path.basename(path))[0]
        return base.replace("_marked_", "")
    
    def load_image(self):
        self.canvas.delete("all")
        self.cell_rects.clear()
        self.x_lines.clear()
        
        if not self.png_files:
            self.canvas.create_text(400, 300, text="No images found", fill="white", font=("", 20))
            return
        
        path = self.png_files[self.idx]
        name = self.img_name()
        self.root.title(f"Stop Annotator [{self.idx+1}/{len(self.png_files)}] — {name}")
        
        pil_img = Image.open(path)
        # Scale to fit screen width (leave margin for scrollbar)
        screen_w = self.root.winfo_screenwidth() - 40
        if pil_img.width > screen_w:
            scale = screen_w / pil_img.width
            new_w = int(pil_img.width * scale)
            new_h = int(pil_img.height * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            scale = 1.0
        
        self.scale = scale
        self.orig_w = pil_img.width / scale  # original image width
        self.orig_h = pil_img.height / scale  # original image height
        self.pil_img = pil_img
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.config(scrollregion=(0, 0, pil_img.width, pil_img.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        # Determine grid layout from thresholds map + stops
        n_rows = 1
        for (img, pol), ths in self.th_map.items():
            if img == name:
                n_rows = max(n_rows, pol)
        img_stops = self.stops.get(name, {})
        for pk in img_stops:
            n_rows = max(n_rows, int(pk))
        if n_rows == 1 and pil_img.height > 800:
            n_rows = max(1, round(pil_img.height / 600))
        
        col_w = pil_img.width / N_COLS
        row_h = pil_img.height / n_rows if n_rows > 0 else pil_img.height
        
        # Draw stop X marks
        for pi in range(n_rows):
            stop_th = img_stops.get(str(pi+1))
            if stop_th is not None:
                ths = self.th_map.get((name, pi+1), [])
                try:
                    ti = ths.index(stop_th)
                except (ValueError, IndexError):
                    continue
                for c in range(ti, N_COLS):
                    x1 = c * col_w + 5
                    y1 = pi * row_h + 5
                    x2 = (c+1) * col_w - 5
                    y2 = (pi+1) * row_h - 5
                    l1 = self.canvas.create_line(x1, y1, x2, y2, fill="red", width=3)
                    l2 = self.canvas.create_line(x2, y1, x1, y2, fill="red", width=3)
                    self.x_lines.extend([l1, l2])
        
        # Create clickable overlay rectangles
        for pi in range(n_rows):
            for c in range(N_COLS):
                x1 = c * col_w
                y1 = pi * row_h
                x2 = (c+1) * col_w
                y2 = (pi+1) * row_h
                rect_id = self.canvas.create_rectangle(x1, y1, x2, y2,
                    fill="", outline="", tags="cell", activeoutline="cyan", activewidth=2)
                self.cell_rects.append((x1, y1, x2, y2, pi, c, rect_id))
        
        self.canvas.tag_bind("cell", "<Button-1>", self.on_cell_click)
        self.canvas.focus_set()
        
        self.info_label.config(text=f"{name}  |  {n_rows} polaroid(s)  |  {N_COLS} thresholds")
        self.status_label.config(text=f"Image {self.idx+1}/{len(self.png_files)}")
    
    def on_cell_click(self, event):
        name = self.img_name()
        if not name:
            return
        
        # Find which cell was clicked
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        n_rows = len(set(r for _,_,_,_,r,_,_ in self.cell_rects))
        
        for x1, y1, x2, y2, pi, ci, rid in self.cell_rects:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                ths = self.th_map.get((name, pi+1), [])
                if ci >= len(ths):
                    return
                th = ths[ci]
                img_stops = self.stops.get(name, {})
                polaroid_key = str(pi+1)
                
                # Toggle: if already set to this threshold, clear it
                if img_stops.get(polaroid_key) == str(th):
                    img_stops.pop(polaroid_key, None)
                    action = "cleared"
                else:
                    img_stops[polaroid_key] = str(th)
                    action = f"stop at t={th}"
                
                if not img_stops:
                    self.stops.pop(name, None)
                else:
                    self.stops[name] = img_stops
                
                # Show data
                csv_key = (name, pi+1, str(th))
                info = self.csv_data.get(csv_key, {})
                self.status_label.config(
                    text=f"P{pi+1} {action} | "
                         f"B: area={info.get('ar_border',0):.1%} match={info.get('mr_border',0):.0%} | "
                         f"I: area={info.get('ar_image',0):.1%} match={info.get('mr_image',0):.0%}"
                )
                
                # Redraw
                self.load_image()
                return
    
    def next_img(self):
        if self.idx < len(self.png_files) - 1:
            self.idx += 1
            self.load_image()
    
    def prev_img(self):
        if self.idx > 0:
            self.idx -= 1
            self.load_image()
    
    def do_save(self):
        save_stops(self.stops)
        self.status_label.config(text="Saved!")
        self.root.after(2000, lambda: self.status_label.config(text=""))


if __name__ == "__main__":
    png_files = find_progression_pngs()
    if not png_files:
        print("No progression PNGs found in outs/. Run scribble_progression.py first.")
        sys.exit(1)
    print(f"Found {len(png_files)} progression images")
    
    csv_data, th_map = load_csv_data()
    stops = load_stops()
    
    StopAnnotator(png_files, csv_data, th_map, stops)
