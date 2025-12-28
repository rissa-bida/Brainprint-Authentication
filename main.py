import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import time
import threading

# --- MODERN THEME CONFIG ---
THEME_BG = "#0f111a"
THEME_PANEL = "#1a1c29"
THEME_FG = "#e0e0e0"
ACCENT_COLOR = "#00ffe0"
ACCENT_SEC = "#ff46f0"
LOG_BG = "#121421"
LOG_FG = "#00ffcc"
FONT_HEADER = ("Segoe UI", 22, "bold")
FONT_MAIN = ("Segoe UI", 12)
FONT_RESULT = ("Courier New", 18, "bold")
FONT_SMALL = ("Segoe UI", 10)

class BrainprintUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Brainprint Authentication System (LSTM Model)")
        self.root.geometry("1000x700")
        self.root.configure(bg=THEME_BG)

        self.is_running = False
        self.eeg_buffer = []  # store EEG stream data

        # --- HEADER ---
        header_frame = tk.Frame(root, bg=THEME_BG, pady=15)
        header_frame.pack(fill="x")
        
        tk.Label(header_frame, text="BRAINPRINT AUTHENTICATION", 
                 font=FONT_HEADER, bg=THEME_BG, fg=ACCENT_COLOR).pack(side="left", padx=20)

        tk.Label(header_frame, text="Model LSTM-RNN v2.0", 
                 font=FONT_SMALL, bg=THEME_BG, fg="#888888").pack(side="right", padx=20)

        # --- MAIN CONTENT ---
        main_container = tk.Frame(root, bg=THEME_BG)
        main_container.pack(fill="both", expand=True, padx=20, pady=10)

        # LEFT PANEL: EEG
        self.left_panel = tk.LabelFrame(main_container, text="Real-Time EEG Input (4 Channels)", 
                                        bg=THEME_PANEL, fg=THEME_FG, font=FONT_MAIN, bd=0)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.fig = Figure(figsize=(5,4), dpi=100, facecolor=THEME_PANEL)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(THEME_PANEL)
        self.ax.tick_params(axis='x', colors=THEME_FG)
        self.ax.tick_params(axis='y', colors=THEME_FG)
        self.ax.spines['bottom'].set_color(THEME_FG)
        self.ax.spines['left'].set_color(THEME_FG)

        # EEG lines
        self.lines = []
        colors = [ACCENT_COLOR, ACCENT_SEC, "#ffcc00", "#ff0066"]
        for i in range(4):
            line, = self.ax.plot([], [], lw=2, color=colors[i], label=f"Ch {i+1}")
            self.lines.append(line)

        self.ax.set_ylim(-3, 10)
        self.ax.set_xlim(0, 100)
        self.ax.legend(facecolor=THEME_PANEL, labelcolor=THEME_FG, loc='upper right', fontsize='small')
        self.ax.grid(True, color="#2b2b3d", linestyle="--", linewidth=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # RIGHT PANEL: Controls & Output
        self.right_panel = tk.Frame(main_container, bg=THEME_PANEL, width=300)
        self.right_panel.pack(side="right", fill="y")

        tk.Label(self.right_panel, text="SESSION CONTROLS", 
                 bg=THEME_PANEL, fg="#aaaaaa", font=FONT_SMALL).pack(pady=(10,5), anchor="w")

        self.btn_connect = tk.Button(self.right_panel, text="1. Input EEG Data",
                                     bg="#222233", fg=THEME_FG, font=FONT_MAIN,
                                     activebackground="#33334a", command=self.start_stream)
        self.btn_connect.pack(fill="x", pady=5)

        self.btn_auth = tk.Button(self.right_panel, text="2. Authenticate Data",
                                  bg=ACCENT_COLOR, fg=THEME_BG, font=FONT_MAIN,
                                  state="disabled", command=self.run_authentication)
        self.btn_auth.pack(fill="x", pady=5)

        tk.Label(self.right_panel, text="SYSTEM LOG",
                 bg=THEME_PANEL, fg="#aaaaaa", font=FONT_SMALL).pack(pady=(30,5), anchor="w")

        self.log_box = tk.Text(self.right_panel, height=8, width=35,
                               bg=LOG_BG, fg=LOG_FG, font=("Consolas", 9), relief="flat", bd=0)
        self.log_box.pack(fill="x", padx=5)
        self.log("System Idle...")

        tk.Label(self.right_panel, text="AUTHENTICATION RESULT",
                 bg=THEME_PANEL, fg="#aaaaaa", font=FONT_SMALL).pack(pady=(30,5), anchor="w")

        self.res_frame = tk.Frame(self.right_panel, bg="#222233", bd=2, relief="groove")
        self.res_frame.pack(fill="x", pady=5, ipady=10)

        self.lbl_status = tk.Label(self.res_frame, text="WAITING FOR INPUT",
                                   bg="#222233", fg=ACCENT_COLOR, font=FONT_RESULT)
        self.lbl_status.pack()

        self.lbl_name = tk.Label(self.res_frame, text="--",
                                 bg="#222233", fg=ACCENT_COLOR, font=("Segoe UI", 14))
        self.lbl_name.pack(pady=5)

        self.lbl_conf = tk.Label(self.res_frame, text="Confidence --%",
                                 bg="#222233", fg="#bbbbbb", font=FONT_SMALL)
        self.lbl_conf.pack()

    # --- Helper Functions ---
    def log(self, message):
        self.log_box.insert(tk.END, f"{message}\n")
        self.log_box.see(tk.END)

    def start_stream(self):
        """Start EEG data input (simulated or real device)"""
        self.is_running = True
        self.btn_connect.config(state="disabled", text="Streaming Data...")
        self.btn_auth.config(state="normal")
        self.log("EEG Data Input Active.")
        self.log("Receiving EEG Data Stream...")
        threading.Thread(target=self.animate_graph).start()

    def animate_graph(self):
        """Simulated EEG stream; replace with real EEG input"""
        while self.is_running:
            x = np.arange(0, 100)
            y1 = np.sin(x*0.1 + time.time()) + np.random.normal(0,0.2,100)+6
            y2 = np.cos(x*0.15 + time.time()) + np.random.normal(0,0.2,100)+4
            y3 = np.sin(x*0.2 + time.time()) + np.random.normal(0,0.2,100)+2
            y4 = np.cos(x*0.05 + time.time()) + np.random.normal(0,0.2,100)+0

            self.lines[0].set_data(x, y1)
            self.lines[1].set_data(x, y2)
            self.lines[2].set_data(x, y3)
            self.lines[3].set_data(x, y4)

            # Save to buffer for authentication
            self.eeg_buffer.append([y1, y2, y3, y4])
            if len(self.eeg_buffer) > 100:  # keep buffer limited
                self.eeg_buffer.pop(0)

            self.canvas.draw()
            time.sleep(0.05)

    def run_authentication(self):
        self.btn_auth.config(state="disabled")
        threading.Thread(target=self._process_data).start()


    def _process_data(self):
        self.log("Capturing 5s EEG segment...")
        segment = np.array(self.eeg_buffer[-100:])  # grab last 100 samples (simulated 5s)
        time.sleep(1)

        self.log("Preprocessing EEG...")
        # TEMPLATE: Insert real preprocessing here
        # Example: normalization, filtering, artifact removal
        preprocessed = segment  # currently just passes data as-is
        time.sleep(0.8)

        self.log("Extracting Features...")
        # TEMPLATE: Feature extraction if required (optional for raw LSTM input)
        features = preprocessed  # currently just passes data as-is
        time.sleep(0.8)

        self.log("Preparing input tensor for LSTM...")
        # TEMPLATE: Convert to 3D tensor (batch, timesteps, channels)
        model_input = features[np.newaxis, :, :]  # shape: (1, timesteps, channels)
        time.sleep(0.5)

        self.log("Running LSTM Model...")
        # TEMPLATE: Replace this block with actual model prediction
        # e.g., pred_probs = lstm_model.predict(model_input)
        #       predicted_index = np.argmax(pred_probs)
        predicted_user = "Clarissa M."  # simulated output
        confidence = round(np.random.uniform(92, 99.9), 2)
        time.sleep(1.5)

        self.log(f"Match Found: {predicted_user}")
        self._update_result(predicted_user, confidence)


    def _update_result(self, name, conf):
        self.lbl_status.config(text="ACCESS GRANTED", fg=ACCENT_COLOR)
        self.lbl_name.config(text=name.upper())
        self.lbl_conf.config(text=f"Confidence {conf}%")
        self.btn_auth.config(state="normal")
        self.log("Authentication Complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrainprintUI(root)
    root.mainloop()
