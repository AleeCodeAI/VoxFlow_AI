"""Backend Evaluation Control Panel — Tkinter GUI."""

import tkinter as tk
from tkinter import scrolledtext
import sys
import io
import threading

# ─── Imports ──────────────────────────────────────────────────────────────────

from evaluations.preprocessor import AIJudge
from evaluations.preprocessor import EvaluationPipeline
from evaluations.transcriber import TranscriptionFunctionalEvaluator
from evaluations.transcriber import LexicalEvaluator

# ─── Eager initialization ─────────────────────────────────────────────────────

lexical_evaluator = LexicalEvaluator()
evaluation_pipeline = EvaluationPipeline()
ai_judge = AIJudge()
transcription_functional = TranscriptionFunctionalEvaluator()


# ─── Evaluation Logic ─────────────────────────────────────────────────────────


class EvaluationControlPanel:
    def run_lexical(self):
        lexical_evaluator.run()

    def run_functional_correctness(self):
        evaluation_pipeline.evaluate()

    def run_ai_judge(self):
        ai_judge.evaluate()

    def run_transcription_functional(self):
        transcription_functional.evaluate()


# ─── Tkinter GUI ──────────────────────────────────────────────────────────────


class App(tk.Tk):
    # ── Palette ──
    BG = "#0f1117"
    CARD_BG = "#1a1d27"
    ACCENT = "#4f8ef7"
    TEXT = "#e8eaf0"
    MUTED = "#6b7280"
    SUCCESS = "#34d399"
    ERROR = "#f87171"
    BORDER = "#2a2d3a"
    LOG_BG = "#0a0c12"

    EVALUATORS = [
        ("Lexical Evaluator", "run_lexical"),
        ("Functional Correctness", "run_functional_correctness"),
        ("AI Judge", "run_ai_judge"),
        ("Transcription Functional Evaluator", "run_transcription_functional"),
    ]

    def __init__(self):
        super().__init__()
        self.panel = EvaluationControlPanel()
        self._buttons = []
        self._build_window()
        self._build_ui()

    # ── Window setup ──────────────────────────────────────────────────────────

    def _build_window(self):
        self.title("Evaluation Control Panel")
        self.configure(bg=self.BG)
        self.resizable(False, False)
        w, h = 540, 620
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=self.BG)
        header.pack(fill="x", padx=28, pady=(24, 6))

        tk.Label(
            header,
            text="EVAL CONTROL PANEL",
            font=("Courier New", 13, "bold"),
            fg=self.ACCENT,
            bg=self.BG,
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Select an evaluator to run",
            font=("Courier New", 9),
            fg=self.MUTED,
            bg=self.BG,
        ).pack(anchor="w")

        tk.Frame(self, bg=self.BORDER, height=1).pack(fill="x", padx=28, pady=(10, 16))

        # Evaluator buttons
        btn_frame = tk.Frame(self, bg=self.BG)
        btn_frame.pack(fill="x", padx=28)

        for label, method in self.EVALUATORS:
            btn = self._make_button(btn_frame, label, method)
            btn.pack(fill="x", pady=5)
            self._buttons.append(btn)

        tk.Frame(self, bg=self.BORDER, height=1).pack(fill="x", padx=28, pady=(16, 0))

        # Log header
        log_header = tk.Frame(self, bg=self.BG)
        log_header.pack(fill="x", padx=28, pady=(10, 4))

        tk.Label(
            log_header,
            text="OUTPUT LOG",
            font=("Courier New", 9, "bold"),
            fg=self.MUTED,
            bg=self.BG,
        ).pack(side="left")

        tk.Button(
            log_header,
            text="Clear",
            font=("Courier New", 8),
            fg=self.MUTED,
            bg=self.BG,
            relief="flat",
            cursor="hand2",
            activeforeground=self.TEXT,
            activebackground=self.BG,
            command=self._clear_log,
        ).pack(side="right")

        # Log box
        self.log = scrolledtext.ScrolledText(
            self,
            font=("Courier New", 9),
            bg=self.LOG_BG,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            relief="flat",
            bd=0,
            wrap="word",
            height=12,
            state="disabled",
            selectbackground=self.ACCENT,
        )
        self.log.pack(fill="both", padx=28, pady=(0, 20))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            self,
            textvariable=self.status_var,
            font=("Courier New", 8),
            fg=self.MUTED,
            bg=self.BG,
            anchor="w",
        ).pack(fill="x", padx=28, pady=(0, 10))

    # ── Button factory ────────────────────────────────────────────────────────

    def _make_button(self, parent, label, method_name):
        frame = tk.Frame(parent, bg=self.CARD_BG)

        def on_enter(_):
            frame.configure(bg=self.ACCENT)
            inner.configure(bg=self.ACCENT)

        def on_leave(_):
            frame.configure(bg=self.CARD_BG)
            inner.configure(bg=self.CARD_BG)

        def on_click():
            self._run_in_thread(label, method_name)

        inner = tk.Label(
            frame,
            text=f"  ▶  {label}",
            font=("Courier New", 10, "bold"),
            fg=self.TEXT,
            bg=self.CARD_BG,
            anchor="w",
            padx=14,
            pady=12,
            cursor="hand2",
        )
        inner.pack(fill="x")

        for widget in (frame, inner):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", lambda _: on_click())

        return frame

    # ── Runner ────────────────────────────────────────────────────────────────

    def _run_in_thread(self, label: str, method_name: str):
        """Run evaluator in a background thread so the UI stays responsive."""
        self._set_buttons_state("disabled")
        self.status_var.set(f"Running: {label}…")
        self._log(f"\n{'─' * 46}\n▶  {label}\n{'─' * 46}\n", color=self.ACCENT)

        def task():
            buffer = io.StringIO()
            old_stdout, sys.stdout = sys.stdout, buffer
            try:
                getattr(self.panel, method_name)()
                output = buffer.getvalue()
                self.after(
                    0,
                    lambda out=output: self._log(
                        out or "(no output)\n", color=self.SUCCESS
                    ),
                )
                self.after(0, lambda: self.status_var.set(f"✓  {label} finished"))
            except Exception as e:
                self.after(
                    0, lambda err=e: self._log(f"[ERROR] {err}\n", color=self.ERROR)
                )
                self.after(0, lambda: self.status_var.set(f"✗  {label} failed"))
            finally:
                sys.stdout = old_stdout
                self.after(0, lambda: self._set_buttons_state("normal"))

        threading.Thread(target=task, daemon=True).start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, text: str, color: str = None):
        self.log.configure(state="normal")
        if color:
            tag = f"c{color.replace('#', '')}"
            self.log.tag_configure(tag, foreground=color)
            self.log.insert("end", text, tag)
        else:
            self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.status_var.set("Ready")

    def _set_buttons_state(self, state: str):
        for btn in self._buttons:
            for child in btn.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
