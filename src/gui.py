import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue
import time
from .ingestion import MarkerIngestion
from .planner import CognitivePlanner
from .judge import RagasJudge
from .config import Config

# --- Theme Constants ---
BG_COLOR = "#2e3440"       # Dark Slate
SIDEBAR_COLOR = "#3b4252"  # Lighter Slate
TEXT_COLOR = "#eceff4"     # White-ish
ACCENT_COLOR = "#88c0d0"   # Blue
SUCCESS_COLOR = "#a3be8c"  # Green
ERROR_COLOR = "#bf616a"    # Red
INPUT_BG = "#434c5e"       # Input field background

class LexiCognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LexiCognition Oral Examiner")
        self.root.geometry("1200x800")
        self.root.configure(bg=BG_COLOR)
        
        # Data State
        self.exam_plan = None
        self.current_q_index = 0
        self.retry_count = 0
        self.exam_complete = False
        self.pipeline_ready = False
        
        # Thread Communication
        self.msg_queue = queue.Queue()
        
        # UI Setup
        self._setup_styles()
        self._build_layout()
        
        # Start Initialization in Thread
        self.log_system("Initializing AI Models... (This may take a moment)")
        threading.Thread(target=self._init_pipeline, daemon=True).start()
        self._check_queue()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame Styles
        style.configure("Main.TFrame", background=BG_COLOR)
        style.configure("Sidebar.TFrame", background=SIDEBAR_COLOR)
        
        # Button Styles
        style.configure("Accent.TButton", 
                        background=ACCENT_COLOR, 
                        foreground=BG_COLOR, 
                        font=("Segoe UI", 10, "bold"),
                        borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#81a1c1")])
        
        # Label Styles
        style.configure("Header.TLabel", background=SIDEBAR_COLOR, foreground="white", font=("Segoe UI", 14, "bold"))
        style.configure("Normal.TLabel", background=SIDEBAR_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=SIDEBAR_COLOR, foreground=ACCENT_COLOR, font=("Consolas", 9))

    def _build_layout(self):
        # 1. Sidebar (Left)
        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=300)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        # Sidebar Content
        ttk.Label(sidebar, text="⚙️ Control Panel", style="Header.TLabel").pack(pady=20, padx=15, anchor="w")
        
        # File Upload
        self.btn_upload = ttk.Button(sidebar, text="📂 Upload PDF", style="Accent.TButton", command=self._upload_pdf_action)
        self.btn_upload.pack(pady=10, padx=15, fill="x")
        self.btn_upload.state(["disabled"]) # Disabled until init finishes
        
        # Progress Info
        self.lbl_topic = ttk.Label(sidebar, text="Topic: N/A", style="Normal.TLabel", wraplength=280)
        self.lbl_topic.pack(pady=20, padx=15, anchor="w")
        
        self.lbl_progress = ttk.Label(sidebar, text="Progress: 0/0", style="Normal.TLabel")
        self.lbl_progress.pack(pady=5, padx=15, anchor="w")
        
        self.progress_bar = ttk.Progressbar(sidebar, orient="horizontal", mode="determinate")
        self.progress_bar.pack(pady=5, padx=15, fill="x")

        # Teacher's Cheat Sheet (Collapsible-ish)
        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", pady=20)
        ttk.Label(sidebar, text="👀 Teacher's View", style="Header.TLabel", font=("Segoe UI", 11, "bold")).pack(padx=15, anchor="w")
        
        self.txt_rubric = tk.Text(sidebar, height=15, bg=INPUT_BG, fg=TEXT_COLOR, 
                                  bd=0, font=("Consolas", 9), wrap="word", padx=5, pady=5)
        self.txt_rubric.pack(padx=15, pady=10, fill="both", expand=True)
        self.txt_rubric.insert("1.0", "Waiting for exam generation...")
        self.txt_rubric.configure(state="disabled")

        # 2. Main Chat Area (Right)
        main_frame = ttk.Frame(self.root, style="Main.TFrame")
        main_frame.pack(side="right", fill="both", expand=True)
        
        # Chat Header
        self.lbl_status = ttk.Label(main_frame, text="🎓 LexiCognition", 
                                   background=BG_COLOR, foreground=ACCENT_COLOR, font=("Segoe UI", 16, "bold"))
        self.lbl_status.pack(pady=15, padx=20, anchor="w")

        # Chat History
        self.chat_display = scrolledtext.ScrolledText(main_frame, bg=BG_COLOR, fg=TEXT_COLOR, 
                                                      font=("Segoe UI", 11), bd=0, padx=20, pady=20, wrap="word")
        self.chat_display.pack(fill="both", expand=True, padx=20)
        self.chat_display.configure(state="disabled")
        
        # Tags for styling
        self.chat_display.tag_config("ai", foreground=ACCENT_COLOR, spacing3=10)
        self.chat_display.tag_config("user", foreground="white", justify="right", rmargin=20, spacing3=10)
        self.chat_display.tag_config("feedback_pass", foreground=SUCCESS_COLOR, font=("Segoe UI", 10, "italic"))
        self.chat_display.tag_config("feedback_fail", foreground=ERROR_COLOR, font=("Segoe UI", 10, "italic"))
        self.chat_display.tag_config("system", foreground="#616e88", font=("Consolas", 10))

        # Input Area
        input_frame = tk.Frame(main_frame, bg=BG_COLOR)
        input_frame.pack(fill="x", pady=20, padx=20)
        
        self.entry_input = tk.Entry(input_frame, bg=INPUT_BG, fg="white", 
                                    font=("Segoe UI", 12), bd=0, insertbackground="white")
        self.entry_input.pack(side="left", fill="both", expand=True, ipady=10, padx=(0, 10))
        self.entry_input.bind("<Return>", self._on_send)
        
        self.btn_send = ttk.Button(input_frame, text="SEND", style="Accent.TButton", command=self._on_send)
        self.btn_send.pack(side="right", ipadx=10)
        self.btn_send.state(["disabled"])

    # --- Logic ---

    def _init_pipeline(self):
        try:
            Config.validate()
            self.ingestor = MarkerIngestion()
            self.planner = CognitivePlanner()
            self.judge = RagasJudge()
            self.msg_queue.put(("INIT_SUCCESS", None))
        except Exception as e:
            self.msg_queue.put(("INIT_FAIL", str(e)))

    def _upload_pdf_action(self):
        path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not path:
            return
        
        self.btn_upload.state(["disabled"])
        self.log_system(f"Processing: {path}")
        self.log_system("Ingesting text and generating Deep-Dive exam plan... (This takes 10-20s)")
        
        threading.Thread(target=self._process_pdf, args=(path,), daemon=True).start()

    def _process_pdf(self, path):
        try:
            raw_text = self.ingestor.process_pdf(path)
            plan = self.planner.generate_exam_plan(raw_text)
            self.msg_queue.put(("PLAN_READY", plan))
        except Exception as e:
            self.msg_queue.put(("ERROR", f"Generation Failed: {str(e)}"))

    def _on_send(self, event=None):
        if str(self.btn_send['state']) == 'disabled':
            return
            
        text = self.entry_input.get().strip()
        if not text:
            return
            
        self.entry_input.delete(0, tk.END)
        self.append_chat("You", text, "user")
        
        # Disable input while processing
        self.entry_input.configure(state="disabled")
        self.btn_send.state(["disabled"])
        
        threading.Thread(target=self._evaluate_answer, args=(text,), daemon=True).start()

    def _evaluate_answer(self, user_input):
        q_data = self.exam_plan.questions[self.current_q_index]
        
        # Extract exemplar correctly based on updated models
        rubric = q_data.rubric
        exemplar_text = getattr(rubric, 'exemplar', None) or getattr(rubric, 'exemplar_answer', None)

        try:
            result = self.judge.evaluate_answer(
                question=q_data.question,
                user_answer=user_input,
                context=q_data.context_snippet,
                criteria=rubric.criteria,
                exemplar=exemplar_text
            )
            self.msg_queue.put(("EVAL_RESULT", result))
        except Exception as e:
            self.msg_queue.put(("ERROR", f"Grading Error: {e}"))

    def _handle_eval_result(self, result):
        # 1. Show Feedback
        tag = "feedback_fail" if result.is_remedial_needed else "feedback_pass"
        self.append_chat("Judge", f"{result.feedback}", tag)
        
        # 2. Logic Branching
        q_data = self.exam_plan.questions[self.current_q_index]
        
        if result.is_remedial_needed and self.retry_count < 2:
            self.retry_count += 1
            hint = f"Not quite. Hint: {q_data.rubric.criteria}"
            self.append_chat("AI", hint, "ai")
        else:
            # Move Next
            if self.retry_count >= 2:
                self.append_chat("AI", "We are stuck. Let's move to the next topic.", "ai")
            else:
                self.append_chat("AI", "Correct. Moving on.", "ai")
                
            self.current_q_index += 1
            self.retry_count = 0
            self._next_question()

        # Re-enable input
        self.entry_input.configure(state="normal")
        self.btn_send.state(["!disabled"])
        self.entry_input.focus()

    def _next_question(self):
        total = len(self.exam_plan.questions)
        
        # Check completion
        if self.current_q_index >= total:
            self.exam_complete = True
            self.append_chat("System", f"Exam Complete! Topic covered: {self.exam_plan.topic}", "system")
            self.entry_input.configure(state="disabled")
            return

        # Update UI Stats
        self.lbl_progress.config(text=f"Question {self.current_q_index + 1} of {total}")
        self.progress_bar['value'] = ((self.current_q_index) / total) * 100
        
        # Update Rubric View
        q = self.exam_plan.questions[self.current_q_index]
        self.txt_rubric.configure(state="normal")
        self.txt_rubric.delete("1.0", tk.END)
        self.txt_rubric.insert("1.0", f"Q{self.current_q_index+1}: {q.question}\n\n")
        self.txt_rubric.insert("end", f"CRITERIA: {q.rubric.criteria}\n\n")
        
        ex = getattr(q.rubric, 'exemplar', None) or getattr(q.rubric, 'exemplar_answer', None)
        if ex:
             self.txt_rubric.insert("end", f"EXEMPLAR: {ex}\n")
             
        self.txt_rubric.configure(state="disabled")

        # Ask Question
        self.append_chat("AI", q.question, "ai")

    # --- UI Helpers ---

    def log_system(self, text):
        self.append_chat("System", text, "system")

    def append_chat(self, sender, message, tag):
        self.chat_display.configure(state="normal")
        if sender:
            self.chat_display.insert(tk.END, f"\n[{sender}]: ", tag)
        self.chat_display.insert(tk.END, f"{message}\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.configure(state="disabled")

    def _check_queue(self):
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()
                
                if msg_type == "INIT_SUCCESS":
                    self.pipeline_ready = True
                    self.btn_upload.state(["!disabled"])
                    self.log_system("System Ready. Please upload a PDF.")
                    
                elif msg_type == "INIT_FAIL":
                    messagebox.showerror("Init Failed", data)
                    
                elif msg_type == "PLAN_READY":
                    self.exam_plan = data
                    self.lbl_topic.config(text=f"Topic: {data.topic}")
                    self.current_q_index = 0
                    self.retry_count = 0
                    self.btn_upload.state(["!disabled"])
                    
                    self.entry_input.configure(state="normal")
                    self.btn_send.state(["!disabled"])
                    
                    self.log_system(f"Exam Generated on: {data.topic}")
                    self._next_question()
                    
                elif msg_type == "EVAL_RESULT":
                    self._handle_eval_result(data)
                    
                elif msg_type == "ERROR":
                    messagebox.showerror("Error", data)
                    self.entry_input.configure(state="normal")
                    self.btn_send.state(["!disabled"])
                    
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = LexiCognitionGUI(root)
    root.mainloop()