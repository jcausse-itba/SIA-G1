"""Interactive Digit Recognition GUI for Multilayer Perceptron models (.pkl).

Features:
- Graphical selection of any .pkl model checkpoint (e.g. models/ej3_mlp_optimized.pkl).
- Freehand mouse drawing canvas with adjustable stroke width and eraser.
- Accuracy-maximizing 28x28 downscaling pipeline:
  * Bounding box cropping
  * Aspect-ratio preserving scaling to 20x20
  * Center of Mass (CoM) alignment to 14x14
  * Anti-aliasing with Lanczos resampling
  * Optional direct 28x28 downsampling / grid mode for comparison
- Real-time 28x28 matrix preview
- 10-class probability distribution with top-class visual highlight.
"""

import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Tuple

# Ensure project root and src/ are in sys.path so unpickling tp_3 modules works
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes softmax probabilities in a numerically stable way."""
    shifted = x - np.max(x)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    return exp_vals / sum_exp if sum_exp > 0 else np.ones_like(x) / len(x)


class PreprocessingPipeline:
    """Downscales freehand drawings to a 28x28 matrix to maximize MLP accuracy."""

    @staticmethod
    def smart_mnist_preprocess(pil_image: Image.Image) -> Tuple[np.ndarray, Image.Image]:
        """MNIST-standard preprocessing:

        1. Find bounding box of drawn digit (pixels > 15).
        2. Crop to bounding box.
        3. Scale bounding box so largest dimension is 20 pixels (aspect-ratio preserved).
        4. Place inside 28x28 blank canvas.
        5. Shift image so Center of Mass of ink is at (13.5, 13.5).
        6. Return normalized float32 array in [0.0, 1.0] and the 28x28 PIL Image.
        """
        arr = np.array(pil_image, dtype=np.float32)
        blank_28 = Image.new("L", (28, 28), 0)
        if arr.max() < 10.0:
            return np.zeros((28, 28), dtype=np.float32), blank_28

        active = arr > 15.0
        if not np.any(active):
            return np.zeros((28, 28), dtype=np.float32), blank_28

        rows = np.any(active, axis=1)
        cols = np.any(active, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        crop = pil_image.crop((xmin, ymin, xmax + 1, ymax + 1))
        w, h = crop.size
        if w <= 0 or h <= 0:
            return np.zeros((28, 28), dtype=np.float32), blank_28

        # Scale so the larger dimension is 20px
        if w > h:
            new_w = 20
            new_h = max(1, int(round(h * 20.0 / w)))
        else:
            new_h = 20
            new_w = max(1, int(round(w * 20.0 / h)))

        crop_resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_28 = Image.new("L", (28, 28), 0)
        pad_x = (28 - new_w) // 2
        pad_y = (28 - new_h) // 2
        img_28.paste(crop_resized, (pad_x, pad_y))

        arr_28 = np.array(img_28, dtype=np.float32)
        total_mass = np.sum(arr_28)
        if total_mass > 0:
            y_indices, x_indices = np.indices((28, 28))
            cy = np.sum(arr_28 * y_indices) / total_mass
            cx = np.sum(arr_28 * x_indices) / total_mass

            shift_x = int(round(13.5 - cx))
            shift_y = int(round(13.5 - cy))

            # Limit shift to avoid moving ink outside canvas boundaries
            shift_x = max(-4, min(4, shift_x))
            shift_y = max(-4, min(4, shift_y))

            if shift_x != 0 or shift_y != 0:
                img_28 = img_28.transform(
                    (28, 28),
                    Image.Transform.AFFINE,
                    (1, 0, -shift_x, 0, 1, -shift_y),
                    resample=Image.Resampling.BILINEAR,
                )
                arr_28 = np.array(img_28, dtype=np.float32)

        norm_matrix = arr_28 / 255.0
        return norm_matrix, img_28

    @staticmethod
    def direct_resize(pil_image: Image.Image) -> Tuple[np.ndarray, Image.Image]:
        """Simple direct resize from drawing canvas to 28x28."""
        img_28 = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
        arr_28 = np.array(img_28, dtype=np.float32) / 255.0
        return arr_28, img_28


class DigitRecognizerApp(tk.Tk):
    """Main Tkinter Application for Interactive Digit Recognition."""

    CANVAS_SIZE = 280  # 10x scale of 28x28
    PREVIEW_SIZE = 140

    def __init__(self) -> None:
        super().__init__()
        self.title("Interactive Digit Recognizer — Multilayer Perceptron")
        self.geometry("980x680")
        self.minsize(920, 640)
        self.configure(bg="#1e1e24")

        # Application state
        self.model: Any = None
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.scaling_method: str | None = None
        self.model_filename = "No model loaded"
        self.model_info = "Please select a .pkl model file"

        self.last_x: int | None = None
        self.last_y: int | None = None
        self.brush_width = tk.IntVar(value=22)
        self.is_eraser = tk.BooleanVar(value=False)
        self.preprocess_mode = tk.StringVar(value="smart")  # 'smart' or 'direct'
        self.auto_predict = tk.BooleanVar(value=True)

        # Offscreen PIL image matching the canvas (Grayscale 'L')
        self.pil_image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.pil_image)

        # UI Build
        self._setup_styles()
        self._build_header()
        self._build_main_content()
        self._build_status_bar()

        # Try to automatically prompt or load default model if available
        self.after(150, self._initial_model_check)

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e24")
        style.configure("Card.TFrame", background="#2b2b36", relief="flat")
        style.configure("TLabel", background="#1e1e24", foreground="#e2e8f0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#ffffff")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10), foreground="#94a3b8")
        style.configure("CardHeader.TLabel", background="#2b2b36", font=("Segoe UI", 11, "bold"), foreground="#38bdf8")

        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#3b82f6",
            foreground="#ffffff",
            padding=(12, 6),
            borderwidth=0,
        )
        style.map("Primary.TButton", background=[("active", "#2563eb")])

        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10),
            background="#475569",
            foreground="#ffffff",
            padding=(10, 5),
            borderwidth=0,
        )
        style.map("Secondary.TButton", background=[("active", "#334155")])

        style.configure(
            "Danger.TButton",
            font=("Segoe UI", 10),
            background="#dc2626",
            foreground="#ffffff",
            padding=(10, 5),
            borderwidth=0,
        )
        style.map("Danger.TButton", background=[("active", "#b91c1c")])

    def _build_header(self) -> None:
        header_frame = tk.Frame(self, bg="#18181f", padx=20, pady=12)
        header_frame.pack(fill=tk.X)

        title_box = tk.Frame(header_frame, bg="#18181f")
        title_box.pack(side=tk.LEFT, fill=tk.Y)

        title_lbl = tk.Label(
            title_box,
            text="Interactive Digit Recognizer (MLP)",
            font=("Segoe UI", 16, "bold"),
            fg="#f8fafc",
            bg="#18181f",
        )
        title_lbl.pack(anchor="w")

        self.model_summary_lbl = tk.Label(
            title_box,
            text="No model loaded",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#18181f",
        )
        self.model_summary_lbl.pack(anchor="w")

        btn_box = tk.Frame(header_frame, bg="#18181f")
        btn_box.pack(side=tk.RIGHT)

        load_btn = tk.Button(
            btn_box,
            text="📁 Select Model (.pkl)...",
            command=self.browse_model_file,
            font=("Segoe UI", 10, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
        )
        load_btn.pack(side=tk.RIGHT, padx=5)

    def _build_main_content(self) -> None:
        main_container = tk.Frame(self, bg="#1e1e24", padx=16, pady=12)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left Column: Canvas & Controls
        left_col = tk.Frame(main_container, bg="#1e1e24")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 16))

        # Canvas Card
        canvas_card = tk.LabelFrame(
            left_col,
            text=" Freehand Drawing Window (280×280) ",
            font=("Segoe UI", 11, "bold"),
            fg="#38bdf8",
            bg="#2b2b36",
            padx=12,
            pady=12,
        )
        canvas_card.pack(fill=tk.X)

        # Drawing Canvas
        self.canvas = tk.Canvas(
            canvas_card,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="#0f172a",
            highlightthickness=2,
            highlightbackground="#334155",
            cursor="crosshair",
        )
        self.canvas.pack(pady=4)

        # Mouse Events for Freehand Drawing
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        # Right Click = Eraser
        self.canvas.bind("<Button-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._on_right_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_mouse_up)

        # Tool Controls (Clear, Eraser, Brush Width)
        tools_frame = tk.Frame(canvas_card, bg="#2b2b36", pady=8)
        tools_frame.pack(fill=tk.X)

        clear_btn = tk.Button(
            tools_frame,
            text="🧹 Clear Canvas",
            command=self.clear_canvas,
            font=("Segoe UI", 9, "bold"),
            bg="#ef4444",
            fg="#ffffff",
            activebackground="#dc2626",
            activeforeground="#ffffff",
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.eraser_btn = tk.Button(
            tools_frame,
            text="✏️ Mode: Brush",
            command=self._toggle_eraser,
            font=("Segoe UI", 9),
            bg="#475569",
            fg="#ffffff",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        self.eraser_btn.pack(side=tk.LEFT, padx=6)

        predict_btn = tk.Button(
            tools_frame,
            text="🔮 Predict",
            command=self.predict_drawing,
            font=("Segoe UI", 9, "bold"),
            bg="#10b981",
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
        )
        predict_btn.pack(side=tk.RIGHT)

        # Brush size and options
        opt_frame = tk.Frame(canvas_card, bg="#2b2b36", pady=4)
        opt_frame.pack(fill=tk.X)

        brush_lbl = tk.Label(opt_frame, text="Brush Size:", font=("Segoe UI", 9), fg="#94a3b8", bg="#2b2b36")
        brush_lbl.pack(side=tk.LEFT, padx=(0, 6))

        brush_slider = ttk.Scale(
            opt_frame,
            from_=12,
            to=36,
            variable=self.brush_width,
            orient=tk.HORIZONTAL,
            length=120,
        )
        brush_slider.pack(side=tk.LEFT)

        auto_chk = tk.Checkbutton(
            opt_frame,
            text="Auto-predict",
            variable=self.auto_predict,
            font=("Segoe UI", 9),
            fg="#cbd5e1",
            bg="#2b2b36",
            selectcolor="#1e293b",
            activebackground="#2b2b36",
            activeforeground="#cbd5e1",
        )
        auto_chk.pack(side=tk.RIGHT)

        # Preprocessing settings card
        prep_card = tk.LabelFrame(
            left_col,
            text=" 28×28 Reduction Settings (Step 3) ",
            font=("Segoe UI", 10, "bold"),
            fg="#94a3b8",
            bg="#2b2b36",
            padx=12,
            pady=8,
        )
        prep_card.pack(fill=tk.X, pady=(10, 0))

        smart_radio = tk.Radiobutton(
            prep_card,
            text="Smart Centering & CoM Alignment (MNIST Standard — Max Accuracy)",
            variable=self.preprocess_mode,
            value="smart",
            command=self._on_prep_mode_change,
            font=("Segoe UI", 9),
            fg="#e2e8f0",
            bg="#2b2b36",
            selectcolor="#1e293b",
            activebackground="#2b2b36",
            activeforeground="#38bdf8",
        )
        smart_radio.pack(anchor="w")

        direct_radio = tk.Radiobutton(
            prep_card,
            text="Direct 28×28 Downsampling (No bounding-box / centering)",
            variable=self.preprocess_mode,
            value="direct",
            command=self._on_prep_mode_change,
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#2b2b36",
            selectcolor="#1e293b",
            activebackground="#2b2b36",
            activeforeground="#e2e8f0",
        )
        direct_radio.pack(anchor="w")

        # Center / Right Column: Matrix Preview & Probability Distribution
        right_col = tk.Frame(main_container, bg="#1e1e24")
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Top Highlight Card
        self.winner_card = tk.Frame(right_col, bg="#2b2b36", padx=16, pady=12, relief="flat")
        self.winner_card.pack(fill=tk.X, pady=(0, 12))

        top_header_box = tk.Frame(self.winner_card, bg="#2b2b36")
        top_header_box.pack(fill=tk.X)

        self.winner_title_lbl = tk.Label(
            top_header_box,
            text="PREDICTED DIGIT",
            font=("Segoe UI", 10, "bold"),
            fg="#94a3b8",
            bg="#2b2b36",
        )
        self.winner_title_lbl.pack(side=tk.LEFT)

        winner_content = tk.Frame(self.winner_card, bg="#2b2b36", pady=4)
        winner_content.pack(fill=tk.X)

        self.winner_digit_lbl = tk.Label(
            winner_content,
            text="—",
            font=("Segoe UI", 48, "bold"),
            fg="#10b981",
            bg="#2b2b36",
            width=3,
        )
        self.winner_digit_lbl.pack(side=tk.LEFT, padx=(0, 12))

        winner_info_box = tk.Frame(winner_content, bg="#2b2b36")
        winner_info_box.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        self.winner_conf_lbl = tk.Label(
            winner_info_box,
            text="Draw a digit on the canvas to begin",
            font=("Segoe UI", 14, "bold"),
            fg="#f8fafc",
            bg="#2b2b36",
            anchor="w",
        )
        self.winner_conf_lbl.pack(fill=tk.X)

        self.winner_sub_lbl = tk.Label(
            winner_info_box,
            text="Model inference evaluates all 10 digit classes (0–9)",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#2b2b36",
            anchor="w",
        )
        self.winner_sub_lbl.pack(fill=tk.X, pady=(2, 0))

        # Bottom Frame: Split between 28x28 Preview and 10 Class Probability List
        detail_frame = tk.Frame(right_col, bg="#1e1e24")
        detail_frame.pack(fill=tk.BOTH, expand=True)

        # 28x28 Preview Box (Left of detail frame)
        preview_box = tk.LabelFrame(
            detail_frame,
            text=" 28×28 Matrix Input ",
            font=("Segoe UI", 10, "bold"),
            fg="#38bdf8",
            bg="#2b2b36",
            padx=10,
            pady=10,
        )
        preview_box.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        self.preview_canvas = tk.Canvas(
            preview_box,
            width=self.PREVIEW_SIZE,
            height=self.PREVIEW_SIZE,
            bg="#0f172a",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.preview_canvas.pack(pady=4)

        self.matrix_stats_lbl = tk.Label(
            preview_box,
            text="784 inputs\nEmpty",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#2b2b36",
            justify=tk.CENTER,
        )
        self.matrix_stats_lbl.pack(pady=(4, 0))

        # 10 Probabilities List (Right of detail frame)
        probs_box = tk.LabelFrame(
            detail_frame,
            text=" 10 Class Probabilities (Step 4) ",
            font=("Segoe UI", 10, "bold"),
            fg="#38bdf8",
            bg="#2b2b36",
            padx=12,
            pady=8,
        )
        probs_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.prob_rows = []
        for digit in range(10):
            row_frame = tk.Frame(probs_box, bg="#2b2b36", pady=2)
            row_frame.pack(fill=tk.X, expand=True)

            d_lbl = tk.Label(
                row_frame,
                text=f"{digit}",
                font=("Segoe UI", 10, "bold"),
                fg="#f1f5f9",
                bg="#334155",
                width=3,
                pady=1,
            )
            d_lbl.pack(side=tk.LEFT, padx=(0, 8))

            bar_canvas = tk.Canvas(
                row_frame,
                height=14,
                width=180,
                bg="#1e293b",
                highlightthickness=0,
            )
            bar_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

            pct_lbl = tk.Label(
                row_frame,
                text="0.0 %",
                font=("Consolas", 10),
                fg="#cbd5e1",
                bg="#2b2b36",
                width=8,
                anchor="e",
            )
            pct_lbl.pack(side=tk.RIGHT, padx=(6, 0))

            badge_lbl = tk.Label(
                row_frame,
                text="",
                font=("Segoe UI", 8, "bold"),
                fg="#10b981",
                bg="#2b2b36",
                width=6,
            )
            badge_lbl.pack(side=tk.RIGHT)

            self.prob_rows.append(
                {
                    "row_frame": row_frame,
                    "digit_label": d_lbl,
                    "bar_canvas": bar_canvas,
                    "pct_label": pct_lbl,
                    "badge_label": badge_lbl,
                }
            )

        self._reset_probabilities()

    def _build_status_bar(self) -> None:
        self.status_bar = tk.Label(
            self,
            text="Ready — Draw a digit or select a model file.",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#18181f",
            anchor="w",
            padx=16,
            pady=4,
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --------------------------------------------------------------------------
    # Model Loading
    # --------------------------------------------------------------------------
    def _initial_model_check(self) -> None:
        """Checks for default model ej3_mlp_optimized.pkl or prompts file dialog."""
        default_model = PROJECT_ROOT / "models" / "ej3_mlp_optimized.pkl"
        if default_model.exists():
            self.load_model(default_model)
        else:
            self.browse_model_file()

    def browse_model_file(self) -> None:
        """Opens a Tkinter file dialog to graphically select any .pkl model file."""
        initial_dir = PROJECT_ROOT / "models"
        if not initial_dir.exists():
            initial_dir = PROJECT_ROOT

        chosen = filedialog.askopenfilename(
            parent=self,
            title="Select Digit-Recognizing Neural Network (.pkl)",
            initialdir=str(initial_dir),
            filetypes=[("Pickle Model Files", "*.pkl"), ("All Files", "*.*")],
        )

        if chosen:
            self.load_model(Path(chosen))

    def load_model(self, file_path: Path) -> bool:
        """Loads and inspects the specified .pkl model checkpoint."""
        try:
            with open(file_path, "rb") as f:
                checkpoint = pickle.load(f)

            if isinstance(checkpoint, dict):
                self.model = checkpoint.get("model", checkpoint)
                self.feature_mean = checkpoint.get("feature_mean")
                self.feature_scale = checkpoint.get("feature_scale")
                self.scaling_method = checkpoint.get("scaling_method", "standardization")
            else:
                self.model = checkpoint
                self.feature_mean = None
                self.feature_scale = None
                self.scaling_method = None

            # Verify model has forward method
            if not hasattr(self.model, "forward"):
                raise ValueError("Loaded object has no 'forward' method (not a valid MLP model).")

            arch_str = ""
            if hasattr(self.model, "layers"):
                sizes = [self.model.layers[0].W.shape[0]] + [l.W.shape[1] for l in self.model.layers]
                arch_str = f"Architecture: {' → '.join(map(str, sizes))}"

            scaling_str = f"Scaling: {self.scaling_method or 'None'}"
            self.model_filename = file_path.name
            self.model_info = f"{arch_str} | {scaling_str}"

            self.model_summary_lbl.config(
                text=f"Loaded: {self.model_filename}  •  {arch_str}  •  {scaling_str}",
                fg="#38bdf8",
            )
            self.status_bar.config(text=f"Model successfully loaded from {file_path.name}")

            # Re-predict current canvas if drawn
            self.predict_drawing()
            return True

        except Exception as e:
            messagebox.showerror(
                "Model Loading Error",
                f"Failed to load model from:\n{file_path}\n\nError details:\n{str(e)}",
            )
            self.status_bar.config(text=f"Error loading model: {e}")
            return False

    # --------------------------------------------------------------------------
    # Freehand Drawing Events
    # --------------------------------------------------------------------------
    def _toggle_eraser(self) -> None:
        self.is_eraser.set(not self.is_eraser.get())
        if self.is_eraser.get():
            self.eraser_btn.config(text="🧹 Mode: Eraser", bg="#f59e0b")
            self.canvas.config(cursor="circle")
        else:
            self.eraser_btn.config(text="✏️ Mode: Brush", bg="#475569")
            self.canvas.config(cursor="crosshair")

    def _on_mouse_down(self, event: tk.Event) -> None:
        self.last_x, self.last_y = event.x, event.y
        color_canvas = "#0f172a" if self.is_eraser.get() else "#ffffff"
        color_pil = 0 if self.is_eraser.get() else 255
        r = self.brush_width.get() // 2

        self.canvas.create_oval(
            event.x - r,
            event.y - r,
            event.x + r,
            event.y + r,
            fill=color_canvas,
            outline=color_canvas,
        )
        self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=color_pil)

        if self.auto_predict.get():
            self.predict_drawing()

    def _on_mouse_move(self, event: tk.Event) -> None:
        if self.last_x is not None and self.last_y is not None:
            color_canvas = "#0f172a" if self.is_eraser.get() else "#ffffff"
            color_pil = 0 if self.is_eraser.get() else 255
            w = self.brush_width.get()
            r = w // 2

            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                width=w,
                fill=color_canvas,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )
            self.draw.line([(self.last_x, self.last_y), (event.x, event.y)], fill=color_pil, width=w, joint="curve")
            self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=color_pil)

        self.last_x, self.last_y = event.x, event.y

    def _on_mouse_up(self, event: tk.Event) -> None:
        self.last_x, self.last_y = None, None
        if self.auto_predict.get():
            self.predict_drawing()

    def _on_right_down(self, event: tk.Event) -> None:
        """Right click acts as an instant eraser."""
        self.last_x, self.last_y = event.x, event.y
        r = self.brush_width.get()
        self.canvas.create_oval(
            event.x - r,
            event.y - r,
            event.x + r,
            event.y + r,
            fill="#0f172a",
            outline="#0f172a",
        )
        self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=0)
        if self.auto_predict.get():
            self.predict_drawing()

    def _on_right_move(self, event: tk.Event) -> None:
        if self.last_x is not None and self.last_y is not None:
            w = self.brush_width.get() * 2
            r = w // 2
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                width=w,
                fill="#0f172a",
                capstyle=tk.ROUND,
            )
            self.draw.line([(self.last_x, self.last_y), (event.x, event.y)], fill=0, width=w)
            self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=0)
        self.last_x, self.last_y = event.x, event.y

    def clear_canvas(self) -> None:
        """Wipes the drawing canvas and clears predictions."""
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.pil_image)
        self._update_preview(np.zeros((28, 28), dtype=np.float32))
        self._reset_probabilities()
        self.status_bar.config(text="Canvas cleared.")

    def _on_prep_mode_change(self) -> None:
        if self.auto_predict.get():
            self.predict_drawing()

    # --------------------------------------------------------------------------
    # Preprocessing & Inference (Steps 3 & 4)
    # --------------------------------------------------------------------------
    def _update_preview(self, matrix_28: np.ndarray) -> None:
        """Renders the exact 28x28 matrix sent to the MLP into the preview canvas."""
        img_arr = (matrix_28 * 255.0).clip(0, 255).astype(np.uint8)
        img_28 = Image.fromarray(img_arr, mode="L")
        # Scaled up for crisp pixel view
        img_zoomed = img_28.resize((self.PREVIEW_SIZE, self.PREVIEW_SIZE), Image.Resampling.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(img_zoomed)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self._preview_photo)

        active = np.sum(matrix_28 > 0.05)
        max_val = np.max(matrix_28) if active > 0 else 0.0
        self.matrix_stats_lbl.config(text=f"Active pixels: {active}/784\nPeak intensity: {max_val:.2f}")

    def predict_drawing(self) -> None:
        """Step 3 & 4: Shrinks drawing to 28x28 matrix, runs MLP, outputs 10 probabilities."""
        if self.model is None:
            self.status_bar.config(text="Warning: No model loaded! Click 'Select Model (.pkl)' to choose a model.")
            return

        # Step 3: Shrink down to 28x28 matrix
        if self.preprocess_mode.get() == "smart":
            matrix_28, _ = PreprocessingPipeline.smart_mnist_preprocess(self.pil_image)
        else:
            matrix_28, _ = PreprocessingPipeline.direct_resize(self.pil_image)

        self._update_preview(matrix_28)

        # If canvas is empty
        if np.max(matrix_28) < 0.01:
            self._reset_probabilities()
            return

        # Flatten matrix to (1, 784)
        flat_input = matrix_28.reshape(1, -1).astype(np.float64)

        # Standardize / scale if model metadata provides mean and scale
        if self.feature_mean is not None and self.feature_scale is not None:
            safe_scale = np.where(self.feature_scale == 0, 1.0, self.feature_scale)
            model_input = (flat_input - self.feature_mean) / safe_scale
        else:
            model_input = flat_input

        # Step 4: Run matrix through the perceptron
        try:
            output = self.model.forward(model_input)[0]

            # Calculate probabilities
            # With MSE training on one-hot targets, output neurons produce direct class confidences.
            # If outputs are non-negative and sum > 0, normalize them.
            positive_scores = np.maximum(output, 0.0)
            if np.sum(positive_scores) > 0.001:
                probs = positive_scores / np.sum(positive_scores)
            else:
                # If all outputs are 0 (e.g. ReLU cutoff), use logits before output activation
                if hasattr(self.model, "layers") and hasattr(self.model.layers[-1], "z"):
                    logits = self.model.layers[-1].z[0]
                    probs = softmax(logits)
                else:
                    probs = softmax(output)

            top_digit = int(np.argmax(probs))
            top_prob = float(probs[top_digit])

            self._display_prediction(probs, top_digit, top_prob, output)
            self.status_bar.config(text=f"Prediction: Digit {top_digit} with {top_prob * 100:.2f}% confidence.")

        except Exception as e:
            self.status_bar.config(text=f"Inference error: {e}")

    def _display_prediction(
        self,
        probs: np.ndarray,
        top_digit: int,
        top_prob: float,
        raw_output: np.ndarray,
    ) -> None:
        """Updates the UI with all 10 probabilities and highlights the winner."""
        # Top Winner Card
        self.winner_digit_lbl.config(text=str(top_digit), fg="#10b981")
        self.winner_conf_lbl.config(text=f"Confidence: {top_prob * 100:.1f} %")
        self.winner_sub_lbl.config(text=f"Raw output neuron activation: {raw_output[top_digit]:.4f}")

        # Update each of the 10 probability bars
        for d in range(10):
            p = float(probs[d])
            row = self.prob_rows[d]
            canvas = row["bar_canvas"]
            canvas.delete("all")

            w = canvas.winfo_width()
            if w <= 1:
                w = 180  # fallback initial width

            bar_width = int(round(p * w))
            is_top = d == top_digit

            if is_top:
                bar_color = "#10b981"  # Vibrant Emerald Green for top pick
                bg_color = "#064e3b"
                row["digit_label"].config(bg="#10b981", fg="#0f172a")
                row["pct_label"].config(fg="#10b981", font=("Consolas", 10, "bold"))
                row["badge_label"].config(text="★ TOP", fg="#10b981")
                row["row_frame"].config(bg="#1e293b")
            else:
                bar_color = "#3b82f6"  # Blue for others
                bg_color = "#1e293b"
                row["digit_label"].config(bg="#334155", fg="#f1f5f9")
                row["pct_label"].config(fg="#94a3b8", font=("Consolas", 10))
                row["badge_label"].config(text="")
                row["row_frame"].config(bg="#2b2b36")

            # Background slot
            canvas.create_rectangle(0, 0, w, 14, fill=bg_color, outline="")
            # Filled bar
            if bar_width > 0:
                canvas.create_rectangle(0, 0, bar_width, 14, fill=bar_color, outline="")

            row["pct_label"].config(text=f"{p * 100:5.1f} %")

    def _reset_probabilities(self) -> None:
        """Resets the probability bars and winner banner."""
        self.winner_digit_lbl.config(text="—", fg="#94a3b8")
        self.winner_conf_lbl.config(text="Draw a digit on the canvas")
        self.winner_sub_lbl.config(text="Awaiting drawing...")

        for d in range(10):
            row = self.prob_rows[d]
            row["bar_canvas"].delete("all")
            w = row["bar_canvas"].winfo_width()
            if w <= 1:
                w = 180
            row["bar_canvas"].create_rectangle(0, 0, w, 14, fill="#1e293b", outline="")
            row["pct_label"].config(text="0.0 %", fg="#94a3b8", font=("Consolas", 10))
            row["digit_label"].config(bg="#334155", fg="#f1f5f9")
            row["badge_label"].config(text="")
            row["row_frame"].config(bg="#2b2b36")


def main() -> None:
    app = DigitRecognizerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
