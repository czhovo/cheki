"""
墨水检测网格图交互式评分工具 (纯 tkinter，流畅不卡)
窗口宽度占满屏幕，右侧垂直滚动条，鼠标滚轮滚动。

用法：
  python score_grids_ui.py                  # 从未评分的网格开始
  python score_grids_ui.py <文件名前缀>     # 打开指定网格

操作：
  鼠标点击 → 选中子图
  数字键 0-9 / - / . → 输入分数
  Backspace → 删除输入 / 清除当前格分数
  Delete → 清除当前格
  Enter / Tab → 确认并跳到下一格
  ← ↑ → ↓ → 移动选中格
  M → 切换最高分筛选（遮暗非最高分 cell）
  S → 保存    N → 下一张    P → 上一张    Q / Esc → 退出
  滚轮 → 上下滚动
"""

import os
import sys
import json
import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

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
N_COLS = len(THRESHOLDS)       # 评分列数（mask 列）
GRID_COLS = N_COLS + 1         # 网格总列数（含左侧原图）
COL_OFFSET = 1                 # 评分列在网格中的起始列号
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
    grids = glob.glob("outs/*_grid.png")
    bases = sorted([os.path.splitext(os.path.basename(g))[0].replace("_grid", "") for g in grids])
    return bases


def get_cell_score(scores, base, row, col):
    entry = scores.get(base, {})
    prompt = PROMPTS[row]
    thresh = str(THRESHOLDS[col])
    pd = entry.get("prompts", {}).get(prompt, {})
    # 用户评分优先
    val = pd.get("thresholds", {}).get(thresh)
    if val is not None:
        return val
    return pd.get("auto", {}).get(thresh)


def is_auto_score(scores, base, row, col):
    """返回 True 如果该 cell 是自动评分的"""
    entry = scores.get(base, {})
    prompt = PROMPTS[row]
    thresh = str(THRESHOLDS[col])
    pd = entry.get("prompts", {}).get(prompt, {})
    # 用户评分优先
    if pd.get("thresholds", {}).get(thresh) is not None:
        return False
    return pd.get("auto", {}).get(thresh) is not None


def set_cell_score(scores, base, row, col, value):
    if base not in scores:
        scores[base] = {"prompts": {}, "notes": ""}
    entry = scores[base]
    prompt = PROMPTS[row]
    if prompt not in entry["prompts"]:
        entry["prompts"][prompt] = {"thresholds": {}}
    entry["prompts"][prompt]["thresholds"][str(THRESHOLDS[col])] = value


# ===== 评分 UI（纯 tkinter，高性能） =====

class GridScorerApp:
    def __init__(self, base, scores):
        self.base = base
        self.scores = scores
        self.row = 0
        self.col = 0
        self.input_buffer = ""
        self.status_msg = ""
        self._done = "quit"

        # ── 加载图片并用 PIL 缩放 ──
        img_path = f"outs/{base}_grid.png"
        self.pil_source = Image.open(img_path)  # 保留源图用于裁剪放大
        self.orig_w, self.orig_h = self.pil_source.size
        self.orig_cell_w = self.orig_w / GRID_COLS
        self.orig_cell_h = self.orig_h / N_ROWS

        # 缩放到窗口宽度
        self.root = tk.Tk()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        win_w = int(sw * 0.95)
        win_h = int(sh * 0.90)

        self.display_w = win_w - 22  # 留滚动条宽度
        scale = self.display_w / self.orig_w
        self.display_h = int(self.orig_h * scale)
        self.scale = scale

        # 缩放图片
        pil_resized = self.pil_source.resize((self.display_w, self.display_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_resized)

        self.cell_w = self.display_w / GRID_COLS
        self.cell_h = self.display_h / N_ROWS

        # ── 窗口 ──
        self.root.title(f"Scoring: {base}_grid.png")
        self.root.geometry(f"{win_w}x{win_h}+{int(sw*0.025)}+{int(sh*0.02)}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self.root.bind_all("<Key>", self._on_key)

        # ── 状态栏 ──
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               font=("Consolas", 11), padding=4, relief="sunken")
        status_bar.pack(side=tk.TOP, fill=tk.X)

        # ── Canvas + 滚动条 ──
        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, bg="#333333",
                                width=self.display_w, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL,
                                         command=self._on_scrollbar)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 放图片
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags=("bg",))
        self.canvas.configure(scrollregion=(0, 0, self.display_w, self.display_h))

        # ── 绘制网格线和分数 ──
        self._draw_grid()

        # 高亮框
        hl_x1, hl_y1, hl_x2, hl_y2 = self._cell_rect()
        self.highlight_rect = self.canvas.create_rectangle(
            hl_x1, hl_y1, hl_x2, hl_y2,
            outline="#FFD700", width=4, tags=("highlight",)
        )

        # 最高分筛选模式（默认开启）
        self.max_only = True
        self._max_score_val = None
        self._init_max_only()

        # ── 绑定鼠标事件 ──
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.focus_set()


        # 放大预览：使用独立的 Toplevel 窗口，拖动流畅无残影
        self.zoom_win = None
        self._zoom_tk = None

        # ── 底部按钮栏 ──
        btn_frame = ttk.Frame(self.root, padding=4)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(btn_frame, text="<- Prev (P)", command=self._on_prev).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Save (S)", command=self._on_save).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Next (N) ->", command=self._on_next).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Clear Cell", command=self._on_clear).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Max Only (M)", command=self._toggle_max_only).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Quit (Q)", command=self._on_quit).pack(side=tk.RIGHT, padx=4)

        self._update_status()

    # ── 坐标计算 ──

    def _cell_rect(self, row=None, col=None):
        """返回 cell 左上角和右下角 canvas 坐标"""
        r = row if row is not None else self.row
        c = col if col is not None else self.col
        x1 = (c + COL_OFFSET) * self.cell_w
        y1 = r * self.cell_h
        x2 = x1 + self.cell_w
        y2 = y1 + self.cell_h
        return x1, y1, x2, y2

    def _draw_grid(self):
        """绘制网格线和已有分数（一次性操作）"""
        for r in range(N_ROWS):
            for c in range(GRID_COLS):
                x1, y1, x2, y2 = self._cell_rect(r, c)
                # 网格线
                self.canvas.create_rectangle(x1, y1, x2, y2,
                                             outline="#555555", width=1,
                                             tags=("grid",))
                # 已有分数（仅 mask 列，跳过原图列）
                if c >= COL_OFFSET:
                    sc = c - COL_OFFSET
                    val = get_cell_score(self.scores, self.base, r, sc)
                    if val is not None:
                        auto = is_auto_score(self.scores, self.base, r, sc)
                        self._draw_score(r, sc, val, auto)

    def _draw_score(self, r, c, val, is_auto=False):
        """在指定 cell 左侧绘制分数文本"""
        x1, y1, x2, y2 = self._cell_rect(r, c)
        lx = x1 + 4
        cy = (y1 + y2) / 2
        color = self._score_color(val)
        txt = str(val)
        char_w = 8
        bw = max(24, len(txt) * char_w + 8)
        # 自动评分：浅蓝底色+细框，手动：纯白+粗框
        bg = "#D8E4FF" if is_auto else "#FFFFFF"
        outline = "#8899BB" if is_auto else "#666666"
        font_style = ("Consolas", 11) if is_auto else ("Consolas", 12, "bold")
        bg_id = self.canvas.create_rectangle(
            lx, cy - 12, lx + bw, cy + 12,
            fill=bg, outline=outline, tags=("score",)
        )
        txt_id = self.canvas.create_text(
            lx + bw / 2, cy, text=txt, fill=color,
            font=font_style, tags=("score",)
        )
        return bg_id, txt_id

    def _score_color(self, val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return "#000000"
        if v < 0:
            return "#CC0000"
        if v <= 3:
            return "#CC6600"
        if v <= 6:
            return "#006600"
        return "#0033CC"

    # ── 更新 ──

    def _update_highlight(self):
        x1, y1, x2, y2 = self._cell_rect()
        self.canvas.coords(self.highlight_rect, x1, y1, x2, y2)
        self.canvas.tag_raise("highlight")
        # 自动滚动
        view_top = self.canvas.canvasy(0)
        view_bot = self.canvas.canvasy(self.canvas.winfo_height())
        if y2 > view_bot or y1 < view_top:
            target = max(0, y1 - self.canvas.winfo_height() / 3)
            self.canvas.yview_moveto(target / self.display_h)

    def _refresh_scores(self):
        """清除并重绘所有分数"""
        self.canvas.delete("score")
        for r in range(N_ROWS):
            for c in range(N_COLS):
                val = get_cell_score(self.scores, self.base, r, c)
                if val is not None:
                    auto = is_auto_score(self.scores, self.base, r, c)
                    self._draw_score(r, c, val, auto)
        self.canvas.tag_raise("highlight")

    def _update_status(self):
        prompt = PROMPTS[self.row]
        thresh = THRESHOLDS[self.col]
        val = get_cell_score(self.scores, self.base, self.row, self.col)
        scored = f"score={val}" if val is not None else "unscored"
        buf = f" [{self.input_buffer}]" if self.input_buffer else ""
        total = N_ROWS * N_COLS
        done = sum(
            1 for r in range(N_ROWS) for c in range(N_COLS)
            if get_cell_score(self.scores, self.base, r, c) is not None
        )
        pct = done / total * 100 if total else 0
        msg = (f"[{self.row+1},{self.col+1}] "
               f"\"{prompt}\"  t={thresh}  {scored}{buf}  |  "
               f"{done}/{total} ({pct:.0f}%)")
        if self.status_msg:
            msg += f"  |  {self.status_msg}"
        self.status_var.set(msg)
        self.root.title(f"Scoring: {self.base}_grid.png  -  {done}/{total} ({pct:.0f}%)")

    # ── 评分逻辑 ──

    def _confirm_score(self):
        if self.input_buffer:
            try:
                val = float(self.input_buffer)
                set_cell_score(self.scores, self.base, self.row, self.col, val)
                self.status_msg = f"Saved {val}"
                self.input_buffer = ""
                self._refresh_scores()
                self._update_status()
                save_scores(self.scores)
            except ValueError:
                self.status_msg = f"Invalid: '{self.input_buffer}'"
                self.input_buffer = ""
                self._update_status()

    def _move(self, dr, dc):
        if self.input_buffer:
            self._confirm_score()
        nr = self.row + dr
        nc = self.col + dc
        if 0 <= nr < N_ROWS and 0 <= nc < N_COLS:
            self.row = nr
            self.col = nc
        self._update_highlight()
        self._update_status()

    def _next_cell(self):
        if self.input_buffer:
            self._confirm_score()
        nc = self.col + 1
        nr = self.row
        if nc >= N_COLS:
            nc = 0
            nr += 1
        if nr < N_ROWS:
            self.row = nr
            self.col = nc
        self._update_highlight()
        self._update_status()

    # ── 事件处理 ──

    def _on_key(self, event):
        key = event.keysym.lower()
        char = event.char

        if key in ("up", "down", "left", "right"):
            dr = -1 if key == "up" else (1 if key == "down" else 0)
            dc = -1 if key == "left" else (1 if key == "right" else 0)
            self._move(dr, dc)
        elif key == "tab":
            self._next_cell()
            return "break"
        elif key == "return":
            self._next_cell()
        elif key == "delete":
            entry = self.scores.get(self.base, {}).get("prompts", {}).get(PROMPTS[self.row], {})
            entry.get("thresholds", {}).pop(str(THRESHOLDS[self.col]), None)
            self.input_buffer = ""
            self.status_msg = "Cleared"
            self._refresh_scores()
            self._update_status()
            save_scores(self.scores)
        elif char and char in "0123456789":
            self.input_buffer += char
            self.status_msg = ""
            self._update_status()
        elif char == "-":
            if self.input_buffer == "":
                self.input_buffer = "-"
                self.status_msg = ""
                self._update_status()
        elif char == ".":
            if "." not in self.input_buffer and self.input_buffer not in ("", "-"):
                self.input_buffer += "."
            elif self.input_buffer in ("", "-"):
                self.input_buffer += "0."
            self.status_msg = ""
            self._update_status()
        elif key == "backspace":
            if self.input_buffer:
                self.input_buffer = self.input_buffer[:-1]
                self.status_msg = ""
                self._update_status()
            else:
                # 无输入缓冲时，清除当前格分数
                self._on_clear()
        elif char and char.lower() == "s":
            self._on_save()
        elif char and char.lower() == "q":
            self._on_quit()
        elif key == "escape":
            self._on_quit()
        elif char and char.lower() == "n":
            self._on_next()
        elif char and char.lower() == "p":
            self._on_prev()
        elif char and char.lower() == "m":
            self._toggle_max_only()

    def _on_click(self, event):
        """鼠标点击选中 cell"""
        if self.input_buffer:
            self._confirm_score()
        # 转换为 canvas 坐标（考虑滚动）
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        col = int(cx // self.cell_w) - COL_OFFSET
        row = int(cy // self.cell_h)
        if col < 0:  # 点击的是左侧原图列，忽略
            return
        col = max(0, min(N_COLS - 1, col))
        row = max(0, min(N_ROWS - 1, row))
        self.row = row
        self.col = col
        self.status_msg = f"Clicked [{row+1},{col+1}]"
        self._update_highlight()
        self._update_status()
        self.canvas.focus_set()

    def _on_double_click(self, event):
        """双击某个子图，弹出可拖动的放大预览窗口"""
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        col = int(cx // self.cell_w) - COL_OFFSET
        row = int(cy // self.cell_h)
        if col < 0 or col >= N_COLS or row < 0 or row >= N_ROWS:
            return

        # 从源图裁剪对应 cell
        src_col = col + COL_OFFSET
        x1 = int(src_col * self.orig_cell_w)
        y1 = int(row * self.orig_cell_h)
        x2 = int((src_col + 1) * self.orig_cell_w)
        y2 = int((row + 1) * self.orig_cell_h)
        crop = self.pil_source.crop((x1, y1, x2, y2))

        zoom_w = int(self.cell_w)
        zoom_h = int(zoom_w * crop.size[1] / crop.size[0])
        crop_resized = crop.resize((zoom_w, zoom_h), Image.LANCZOS)
        self._zoom_tk = ImageTk.PhotoImage(crop_resized)

        # 创建/更新独立预览窗口
        if self.zoom_win is None:
            self.zoom_win = tk.Toplevel(self.root)
            self.zoom_win.overrideredirect(True)  # 无边框
            self.zoom_win.attributes("-topmost", True)
            self.zoom_label = tk.Label(self.zoom_win, image=self._zoom_tk,
                                       bg="#222222", bd=2, relief="solid",
                                       cursor="fleur")
            self.zoom_label.pack()
            # 拖动
            self.zoom_label.bind("<Button-1>", self._zoom_drag_start)
            self.zoom_label.bind("<B1-Motion>", self._zoom_drag_move)
            # 双击预览关闭
            self.zoom_label.bind("<Double-Button-1>", self._zoom_close)
            # 初始位置：鼠标附近
            self.zoom_win.geometry(f"+{event.x_root+20}+{event.y_root+20}")
        else:
            self.zoom_label.configure(image=self._zoom_tk)
        self.status_msg = f"Zoom [{row+1},{col+1}]"

    def _zoom_drag_start(self, event):
        self.zoom_drag_data = {"x": event.x_root, "y": event.y_root}

    def _zoom_drag_move(self, event):
        dx = event.x_root - self.zoom_drag_data["x"]
        dy = event.y_root - self.zoom_drag_data["y"]
        x = self.zoom_win.winfo_x() + dx
        y = self.zoom_win.winfo_y() + dy
        self.zoom_win.geometry(f"+{x}+{y}")
        self.zoom_drag_data["x"] = event.x_root
        self.zoom_drag_data["y"] = event.y_root

    def _zoom_close(self, event=None):
        if self.zoom_win:
            self.zoom_win.destroy()
            self.zoom_win = None
            self._zoom_tk = None

    def _on_scrollbar(self, *args):
        self.canvas.yview(*args)

    def _on_mousewheel(self, event):
        """Windows 鼠标滚轮"""
        delta = -1 * (event.delta // 120)
        self.canvas.yview_scroll(delta, "units")

    # ── 命令 ──

    def _on_save(self):
        if self.input_buffer:
            self._confirm_score()
        save_scores(self.scores)
        self.status_msg = "Saved!"
        self._update_status()

    def _toggle_max_only(self):
        self.max_only = not self.max_only
        if self.max_only:
            self._calc_max_score()
        else:
            self._max_score_val = None
            self.status_msg = "Max-only OFF"
        self._refresh_overlay()
        self._update_status()

    def _init_max_only(self):
        """初始化时计算最高分并应用遮罩"""
        self._calc_max_score()
        self._refresh_overlay()

    def _calc_max_score(self):
        """计算当前网格的最高分"""
        self._max_score_val = max(
            float(get_cell_score(self.scores, self.base, r, c))
            for r in range(N_ROWS) for c in range(N_COLS)
            if get_cell_score(self.scores, self.base, r, c) is not None
        )
        self.status_msg = f"Max-only ON (>= {self._max_score_val})"

    def _refresh_overlay(self):
        """刷新最高分遮罩"""
        self.canvas.delete("max_dim")
        if not self.max_only or self._max_score_val is None:
            return
        for r in range(N_ROWS):
            for c in range(N_COLS):
                val = get_cell_score(self.scores, self.base, r, c)
                if val is not None and float(val) >= self._max_score_val - 0.001:
                    continue  # 最高分 cell 不遮
                # 遮暗非最高分 cell
                x1, y1, x2, y2 = self._cell_rect(r, c)
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill="#111111", stipple="gray50", outline="",
                    tags=("max_dim",)
                )
        self.canvas.tag_raise("highlight")
        self.canvas.tag_raise("score")
        self.canvas.tag_raise("zoom")

    def _on_clear(self):
        entry = self.scores.get(self.base, {}).get("prompts", {}).get(PROMPTS[self.row], {})
        entry.get("thresholds", {}).pop(str(THRESHOLDS[self.col]), None)
        self.input_buffer = ""
        self.status_msg = "Cleared"
        self._refresh_scores()
        self._update_status()
        save_scores(self.scores)

    def _on_next(self):
        if self.input_buffer:
            self._confirm_score()
        save_scores(self.scores)
        self._zoom_close()
        self._done = "next"
        self.root.destroy()

    def _on_prev(self):
        if self.input_buffer:
            self._confirm_score()
        save_scores(self.scores)
        self._zoom_close()
        self._done = "prev"
        self.root.destroy()

    def _on_quit(self):
        if self.input_buffer:
            self._confirm_score()
        save_scores(self.scores)
        self._zoom_close()
        self._done = "quit"
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self._done


# ===== 主程序 =====

def main():
    bases = find_grids()
    if not bases:
        print("未找到任何 grid 文件")
        return

    scores = load_scores()

    target = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            target = arg
            break

    if target:
        matches = [b for b in bases if target in b]
        if not matches:
            print(f"未找到匹配 '{target}' 的网格文件")
            return
        start_idx = bases.index(matches[0])
    else:
        start_idx = 0
        for i, b in enumerate(bases):
            done = sum(
                1 for r in range(N_ROWS) for c in range(N_COLS)
                if get_cell_score(scores, b, r, c) is not None
            )
            if done < N_ROWS * N_COLS:
                start_idx = i
                break

    idx = start_idx
    while 0 <= idx < len(bases):
        base = bases[idx]
        app = GridScorerApp(base, scores)
        action = app.run()

        if action == "quit":
            break
        elif action == "next":
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)
        else:
            break

        scores = load_scores()

    print(f"评分数据已保存到 {SCORES_FILE}")


if __name__ == "__main__":
    main()
