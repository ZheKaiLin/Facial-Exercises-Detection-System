"""
第二步：第二位標註者標註工具（盲標註，不會顯示原始檔名或原始標籤）

操作方式：
    - 滑鼠左鍵拖曳：框選臉部（可重畫，以最後一次拖曳為準）
    - 鍵盤 1~5 或點擊按鈕：選擇表情類別（1=angry 2=happy 3=neutral 4=sad 5=surprised）
    - 「上一張 / 下一張」按鈕或左右方向鍵：切換圖片（切換時自動儲存目前標註）
    - 必須同時完成「框選」與「選類別」，該張才會被視為已標註（左側清單會打勾）
    - 可隨時關閉視窗，下次執行會從上次進度繼續（已存檔的不會遺失）

輸出：
    labels_annotator2/img_XXX.txt   YOLO 格式（class x_center y_center width height，皆為 0~1 正規化值）

使用方式：
    python annotate_gui.py
"""
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

CLASSES = ["angry", "happy", "neutral", "sad", "surprised"]
CLASS_LABEL_ZH = {
    "angry": "1  生氣 angry",
    "happy": "2  開心 happy",
    "neutral": "3  中性 neutral",
    "sad": "4  悲傷 sad",
    "surprised": "5  驚訝 surprised",
}
KEY_TO_CLASS_IDX = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = SCRIPT_DIR / "blind_images"
LABELS_DIR = SCRIPT_DIR / "labels_annotator2"

MAX_W, MAX_H = 860, 620


class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("表情辨識盲標註工具（第二位標註者）")

        LABELS_DIR.mkdir(parents=True, exist_ok=True)

        self.image_paths = sorted(IMAGES_DIR.glob("img_*.*"))
        if not self.image_paths:
            raise RuntimeError(f"找不到待標註圖片，請先執行 prepare_blind_annotation.py（資料夾：{IMAGES_DIR}）")

        self.idx = 0
        self.box_orig = None      # (x1, y1, x2, y2)，原始影像像素座標
        self.cls_idx = None
        self.scale = 1.0
        self.tk_img = None
        self.rect_id = None
        self.drag_start_canvas = None

        self._build_ui()
        self._load_progress_flags()
        self.load_image(0)

    # ---------------- UI ----------------
    def _build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main)
        left.pack(side="left", fill="y")
        tk.Label(left, text="圖片清單", font=("Microsoft JhengHei", 10, "bold")).pack()
        self.listbox = tk.Listbox(left, width=16, height=34, exportselection=False)
        for p in self.image_paths:
            self.listbox.insert("end", p.stem)
        self.listbox.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        scrollbar = tk.Scrollbar(left, command=self.listbox.yview)
        scrollbar.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        right = tk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(right, width=MAX_W, height=MAX_H, bg="gray20", cursor="cross")
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        bottom = tk.Frame(right)
        bottom.pack(fill="x", padx=8, pady=4)

        self.status_label = tk.Label(bottom, text="", font=("Microsoft JhengHei", 11))
        self.status_label.pack(side="top", anchor="w")

        cls_frame = tk.Frame(bottom)
        cls_frame.pack(side="top", fill="x", pady=6)
        self.cls_buttons = []
        for i, cls in enumerate(CLASSES):
            btn = tk.Button(
                cls_frame, text=CLASS_LABEL_ZH[cls], width=16,
                command=lambda i=i: self.select_class(i)
            )
            btn.pack(side="left", padx=3)
            self.cls_buttons.append(btn)

        nav_frame = tk.Frame(bottom)
        nav_frame.pack(side="top", fill="x", pady=6)
        tk.Button(nav_frame, text="< 上一張", width=12, command=self.prev_image).pack(side="left", padx=3)
        tk.Button(nav_frame, text="下一張 >", width=12, command=self.next_image).pack(side="left", padx=3)
        tk.Button(nav_frame, text="清除框選", width=12, command=self.clear_box).pack(side="left", padx=3)
        self.progress_label = tk.Label(nav_frame, text="")
        self.progress_label.pack(side="left", padx=20)

        self.root.bind("<Key>", self.on_key)

    # ---------------- 資料 ----------------
    def label_path(self, path):
        return LABELS_DIR / f"{path.stem}.txt"

    def _load_progress_flags(self):
        self.done_flags = []
        for p in self.image_paths:
            self.done_flags.append(self.label_path(p).exists())
        self._refresh_listbox_flags()

    def _refresh_listbox_flags(self):
        for i, p in enumerate(self.image_paths):
            mark = "✔ " if self.done_flags[i] else "　"
            self.listbox.delete(i)
            self.listbox.insert(i, f"{mark}{p.stem}")

    def load_image(self, idx):
        self.idx = idx
        path = self.image_paths[idx]
        pil_img = Image.open(path).convert("RGB")
        w, h = pil_img.size
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        self.scale = scale
        disp_w, disp_h = int(w * scale), int(h * scale)
        disp_img = pil_img.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(disp_img)

        self.canvas.delete("all")
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.rect_id = None

        self.box_orig = None
        self.cls_idx = None

        lbl_path = self.label_path(path)
        if lbl_path.exists():
            text = lbl_path.read_text(encoding="utf-8").strip()
            if text:
                parts = text.split()
                c = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                self.box_orig = (x1, y1, x2, y2)
                self.cls_idx = c
                self._draw_box_on_canvas()

        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(idx)
        self.listbox.see(idx)

        self._update_status()

    def _draw_box_on_canvas(self):
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        if self.box_orig is None:
            return
        x1, y1, x2, y2 = self.box_orig
        cx1, cy1, cx2, cy2 = x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale
        self.rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="lime", width=2)

    def _update_status(self):
        n = len(self.image_paths)
        done = sum(self.done_flags)
        self.progress_label.config(text=f"已完成 {done} / {n}")
        cls_text = CLASS_LABEL_ZH[CLASSES[self.cls_idx]] if self.cls_idx is not None else "（尚未選擇）"
        box_text = "已框選" if self.box_orig is not None else "（尚未框選）"
        self.status_label.config(
            text=f"[{self.idx + 1}/{n}] {self.image_paths[self.idx].name}    類別：{cls_text}    框選：{box_text}"
        )
        for i, btn in enumerate(self.cls_buttons):
            btn.config(relief="sunken" if self.cls_idx == i else "raised")

    # ---------------- 互動 ----------------
    def on_mouse_down(self, event):
        self.drag_start_canvas = (event.x, event.y)

    def on_mouse_drag(self, event):
        if self.drag_start_canvas is None:
            return
        x0, y0 = self.drag_start_canvas
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="yellow", width=2)

    def on_mouse_up(self, event):
        if self.drag_start_canvas is None:
            return
        x0, y0 = self.drag_start_canvas
        x1, y1 = event.x, event.y
        self.drag_start_canvas = None

        cx1, cx2 = sorted((x0, x1))
        cy1, cy2 = sorted((y0, y1))
        if cx2 - cx1 < 3 or cy2 - cy1 < 3:
            return  # 太小，視為誤觸，不採用

        ox1, oy1 = cx1 / self.scale, cy1 / self.scale
        ox2, oy2 = cx2 / self.scale, cy2 / self.scale
        self.box_orig = (ox1, oy1, ox2, oy2)
        self._draw_box_on_canvas()
        self._update_status()

    def select_class(self, cls_idx):
        self.cls_idx = cls_idx
        self._update_status()

    def clear_box(self):
        self.box_orig = None
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self._update_status()

    def on_key(self, event):
        if event.char in KEY_TO_CLASS_IDX:
            self.select_class(KEY_TO_CLASS_IDX[event.char])
        elif event.keysym == "Left":
            self.prev_image()
        elif event.keysym == "Right":
            self.next_image()

    def on_listbox_select(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.save_current()
        self.load_image(sel[0])

    # ---------------- 存檔 / 導覽 ----------------
    def save_current(self):
        path = self.image_paths[self.idx]
        lbl_path = self.label_path(path)
        if self.box_orig is not None and self.cls_idx is not None:
            pil_img = Image.open(path)
            w, h = pil_img.size
            x1, y1, x2, y2 = self.box_orig
            x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lbl_path.write_text(f"{self.cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            self.done_flags[self.idx] = True
        self._refresh_listbox_flags()

    def next_image(self):
        self.save_current()
        if self.idx < len(self.image_paths) - 1:
            self.load_image(self.idx + 1)
        else:
            self._update_status()

    def prev_image(self):
        self.save_current()
        if self.idx > 0:
            self.load_image(self.idx - 1)


def main():
    root = tk.Tk()
    app = AnnotatorApp(root)

    def on_close():
        app.save_current()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
