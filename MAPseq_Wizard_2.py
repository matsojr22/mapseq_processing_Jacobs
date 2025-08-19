import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
import sys
import json

def auto_update_repo(repo_path, branch="main"):
    try:
        print(f"🔄 Checking for updates in {repo_path}...")
        subprocess.run(["git", "fetch", "--all"], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_path, check=True)
        print("✅ Repository successfully updated to the latest version.")
    except Exception as e:
        print(f"⚠️ Auto-update failed: {e}")

# Detect the repo path (assumes this script is inside the repo)
if getattr(sys, 'frozen', False):
    # We're in a PyInstaller exe
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

repo_path = base_path  # adjust if the repo is in a subfolder

auto_update_repo(repo_path)

# Define script configurations with their arguments
SCRIPT_CONFIGS = {
    "process-nbcm-tsv.py": {
        "description": "Main NBCM data processing pipeline",
        "arguments": [
            {"name": "sample_name", "type": "input", "required": True, "default": "", "help": "Sample name"},
            {"name": "data_file", "type": "file", "required": True, "default": "", "help": "Path to the input nbcm.csv file", "file_types": (("CSV/TSV Files", "*.csv *.tsv"),)},
            {"name": "out_dir", "type": "folder", "required": True, "default": "", "help": "Output directory for saving results"},
            {"name": "alpha", "type": "input", "required": False, "default": "0.05", "help": "Significance threshold for Bonferroni correction"},
            {"name": "labels", "type": "input", "required": False, "default": "", "help": "Comma-separated column labels (e.g., 'target1,target2,target3,target-neg-bio')"},
            {"name": "injection_umi_min", "type": "input", "required": False, "default": "1", "help": "Minimum 'inj' UMI values threshold"},
            {"name": "min_target_count", "type": "input", "required": False, "default": "10", "help": "Minimum UMI count required in at least one target area"},
            {"name": "min_body_to_target_ratio", "type": "input", "required": False, "default": "10", "help": "Minimum fold-difference between 'inj' value and highest target count"},
            {"name": "target_umi_min", "type": "input", "required": False, "default": "2", "help": "Threshold filter for target area UMI counts"},
            {"name": "special_area_1", "type": "input", "required": False, "default": "", "help": "One of your favorite target areas"},
            {"name": "special_area_2", "type": "input", "required": False, "default": "", "help": "Another of your favorite target areas to compare"},
            {"name": "apply_outlier_filtering", "type": "checkbox", "required": False, "default": False, "help": "Enable outlier filtering using mean + 2*std deviation"},
            {"name": "force_user_threshold", "type": "checkbox", "required": False, "default": False, "help": "Override all automatic thresholding and use user-defined target_umi_min"}
        ]
    },
    "helpers/plot_normalized_projection_strength_data.py": {
        "description": "Plot normalized projection strength data with mean and SEM",
        "arguments": [
            {"name": "data_dir", "type": "folder", "required": True, "default": "", "help": "Directory containing *_raw_data.csv files"},
            {"name": "output_dir", "type": "folder", "required": False, "default": "plots", "help": "Output directory for plots"}
        ]
    },
    "helpers/projection_analysis.py": {
        "description": "Analyze projection data following Klingler et al. 2018 methods",
        "arguments": [
            {"name": "input_file", "type": "file", "required": True, "default": "", "help": "Input CSV file with projection data", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "output_dir", "type": "folder", "required": False, "default": "projection_analysis", "help": "Output directory for analysis results"},
            {"name": "comparison_name", "type": "input", "required": False, "default": "", "help": "Name for the comparison (used in plot titles)"}
        ]
    },
    "helpers/motif_clustering.py": {
        "description": "Cluster motif effect size trajectories",
        "arguments": [
            {"name": "input_file", "type": "file", "required": False, "default": "dataframe_all.csv", "help": "Input CSV file with motif data", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "output_prefix", "type": "input", "required": False, "default": "dataframe_all.csv", "help": "Prefix for output files"},
            {"name": "n_clusters", "type": "input", "required": False, "default": "", "help": "Number of clusters (leave empty for automatic determination)"}
        ]
    },
    "helpers/motif_significange_trajectories.py": {
        "description": "Plot motif effect size trajectories across stages",
        "arguments": [
            {"name": "input_dir", "type": "folder", "required": True, "default": "", "help": "Directory containing *_upsetplot.csv files"}
        ]
    },
    "helpers/compare_datasets_pipeline_mapseq.py": {
        "description": "Compare Allen, VSV, and MapSeq datasets with area combining",
        "arguments": [
            {"name": "allen_file", "type": "file", "required": False, "default": "", "help": "Allen dataset file", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "vsv_file", "type": "file", "required": False, "default": "", "help": "VSV dataset file", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "mapseq_file", "type": "file", "required": False, "default": "", "help": "MapSeq dataset file", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "output_dir", "type": "folder", "required": False, "default": "dataset_comparison", "help": "Output directory for comparison results"},
            {"name": "comparison_name", "type": "input", "required": False, "default": "Allen_VSV_MapSeq", "help": "Name for the comparison"}
        ]
    },
    "helpers/compare_vsv_mapseq_two_way.py": {
        "description": "Two-way comparison between VSV and MapSeq datasets",
        "arguments": [
            {"name": "vsv_file", "type": "file", "required": True, "default": "", "help": "VSV dataset file", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "mapseq_file", "type": "file", "required": True, "default": "", "help": "MapSeq dataset file", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "output_dir", "type": "folder", "required": False, "default": "vsv_mapseq_comparison", "help": "Output directory for comparison results"}
        ]
    },
    "helpers/compare_datasets_pipeline.py": {
        "description": "Compare Allen and VSV datasets",
        "arguments": [
            {"name": "allen_file", "type": "file", "required": True, "default": "", "help": "Input CSV file with projection data", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "vsv_file", "type": "file", "required": True, "default": "", "help": "VSV dataset file", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "output_dir", "type": "folder", "required": False, "default": "allen_vsv_comparison", "help": "Output directory for comparison results"}
        ]
    }
}

class MAPseqWizardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MAPseq Processing Wizard 2.0 (Tkinter)")
        self.root.geometry("1000x700")
        
        # Ensure window appears in front on macOS
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.attributes('-topmost', False)
        self.root.focus_force()
        
        # Center the window on screen
        self.center_window()
        
        # Variables
        self.script_var = tk.StringVar()
        self.arg_widgets = {}
        
        self.create_widgets()
        
        # Set initial script
        if SCRIPT_CONFIGS:
            first_script = list(SCRIPT_CONFIGS.keys())[0]
            self.script_var.set(first_script)
            self.update_script_display()
        
        # Ensure buttons are properly configured
        self.root.after(100, self.configure_buttons)
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="MAPseq Processing Wizard 2.0", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)
        
        # Script selection frame
        script_frame = tk.Frame(self.root)
        script_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Label(script_frame, text="Select a script to run:", font=("Helvetica", 12)).pack(anchor="w")
        
        script_combo = ttk.Combobox(script_frame, textvariable=self.script_var, 
                                   values=list(SCRIPT_CONFIGS.keys()), state="readonly", width=60)
        script_combo.pack(fill="x", pady=5)
        script_combo.bind("<<ComboboxSelected>>", self.on_script_change)
        
        # Script description
        self.desc_label = tk.Label(script_frame, text="", fg="blue", wraplength=800)
        self.desc_label.pack(pady=5)
        
        # Separator
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=20, pady=10)
        
        # Arguments frame
        args_frame = tk.Frame(self.root)
        args_frame.pack(fill="both", expand=True, padx=20, pady=5)
        
        tk.Label(args_frame, text="Script Arguments:", font=("Helvetica", 12, "bold")).pack(anchor="w")
        
        # Create scrollable frame for arguments
        canvas = tk.Canvas(args_frame)
        scrollbar = ttk.Scrollbar(args_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Separator
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=20, pady=10)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Create buttons with explicit text and better styling
        run_button = tk.Button(button_frame, text="Run Script", command=self.run_script, 
                              bg="green", fg="white", font=("Helvetica", 12, "bold"),
                              width=12, height=2, relief="raised", bd=3)
        run_button.pack(side="left", padx=5)
        
        clear_button = tk.Button(button_frame, text="Clear All", command=self.clear_all, 
                                bg="orange", fg="white", font=("Helvetica", 12, "bold"),
                                width=12, height=2, relief="raised", bd=3)
        clear_button.pack(side="left", padx=5)
        
        exit_button = tk.Button(button_frame, text="Exit", command=self.root.quit, 
                               bg="red", fg="white", font=("Helvetica", 12, "bold"),
                               width=12, height=2, relief="raised", bd=3)
        exit_button.pack(side="left", padx=5)
        
        # Output frame
        output_frame = tk.Frame(self.root)
        output_frame.pack(fill="both", expand=True, padx=20, pady=5)
        
        tk.Label(output_frame, text="Output:", font=("Helvetica", 12, "bold")).pack(anchor="w")
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, width=80)
        self.output_text.pack(fill="both", expand=True)
    
    def on_script_change(self, event=None):
        self.update_script_display()
    
    def update_script_display(self):
        selected_script = self.script_var.get()
        if selected_script in SCRIPT_CONFIGS:
            # Update description
            self.desc_label.config(text=SCRIPT_CONFIGS[selected_script]["description"])
            
            # Clear existing argument widgets
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            self.arg_widgets = {}
            
            # Create argument widgets
            for arg in SCRIPT_CONFIGS[selected_script]["arguments"]:
                self.create_argument_widget(arg)
    
    def create_argument_widget(self, arg):
        arg_name = arg["name"]
        arg_type = arg["type"]
        required = arg["required"]
        default = arg["default"]
        help_text = arg["help"]
        
        # Create frame for this argument
        arg_frame = tk.Frame(self.scrollable_frame)
        arg_frame.pack(fill="x", pady=2)
        
        # Label with required indicator
        label_text = f"{arg_name}:"
        if required:
            label_text += " *"
        
        label = tk.Label(arg_frame, text=label_text, width=20, anchor="w")
        label.pack(side="left")
        
        # Input widget based on type
        if arg_type == "input":
            entry = tk.Entry(arg_frame, width=40)
            entry.insert(0, default)
            entry.pack(side="left", padx=5)
            self.arg_widgets[arg_name] = entry
        elif arg_type == "file":
            file_frame = tk.Frame(arg_frame)
            file_frame.pack(side="left", padx=5)
            
            entry = tk.Entry(file_frame, width=30)
            entry.insert(0, default)
            entry.pack(side="left")
            
            tk.Button(file_frame, text="Browse", 
                     command=lambda: self.browse_file(entry, arg.get("file_types", (("All Files", "*.*"),)))).pack(side="left", padx=2)
            
            self.arg_widgets[arg_name] = entry
        elif arg_type == "folder":
            folder_frame = tk.Frame(arg_frame)
            folder_frame.pack(side="left", padx=5)
            
            entry = tk.Entry(folder_frame, width=30)
            entry.insert(0, default)
            entry.pack(side="left")
            
            tk.Button(folder_frame, text="Browse", 
                     command=lambda: self.browse_folder(entry)).pack(side="left", padx=2)
            
            self.arg_widgets[arg_name] = entry
        elif arg_type == "checkbox":
            var = tk.BooleanVar(value=default)
            checkbox = tk.Checkbutton(arg_frame, variable=var)
            checkbox.pack(side="left", padx=5)
            self.arg_widgets[arg_name] = var
        
        # Help text
        help_label = tk.Label(arg_frame, text=help_text, fg="gray", wraplength=400)
        help_label.pack(side="left", padx=10)
    
    def browse_file(self, entry_widget, file_types):
        filename = filedialog.askopenfilename(filetypes=file_types)
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)
    
    def browse_folder(self, entry_widget):
        folder = filedialog.askdirectory()
        if folder:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, folder)
    
    def get_argument_values(self):
        values = {}
        for arg_name, widget in self.arg_widgets.items():
            if isinstance(widget, tk.Entry):
                values[arg_name] = widget.get()
            elif isinstance(widget, tk.BooleanVar):
                values[arg_name] = widget.get()
        return values
    
    def validate_arguments(self, values):
        selected_script = self.script_var.get()
        if selected_script not in SCRIPT_CONFIGS:
            return False, "No script selected"
        
        config = SCRIPT_CONFIGS[selected_script]
        missing_args = []
        
        for arg in config["arguments"]:
            if arg["required"]:
                value = values.get(arg["name"], "")
                if not value:
                    missing_args.append(arg["name"])
        
        if missing_args:
            return False, f"Missing required arguments: {', '.join(missing_args)}"
        
        return True, ""
    
    def build_command(self, script_name, values):
        if script_name not in SCRIPT_CONFIGS:
            return None
        
        config = SCRIPT_CONFIGS[script_name]
        cmd = ["conda", "run", "-n", "mapseq_processing", "python", script_name]
        
        for arg in config["arguments"]:
            arg_name = arg["name"]
            arg_type = arg["type"]
            required = arg["required"]
            
            value = values.get(arg_name, "")
            
            # Skip empty optional arguments
            if not required and not value:
                continue
            
            # Handle different argument types
            if arg_type == "checkbox":
                if value:  # Only add if checked
                    cmd.extend([f"--{arg_name}"])
            else:
                if value:  # Only add if not empty
                    cmd.extend([f"--{arg_name}", str(value)])
        
        return cmd
    
    def run_script(self):
        selected_script = self.script_var.get()
        if not selected_script:
            messagebox.showerror("Error", "Please select a script first!")
            return
        
        # Get argument values
        values = self.get_argument_values()
        
        # Validate arguments
        is_valid, error_msg = self.validate_arguments(values)
        if not is_valid:
            messagebox.showerror("Validation Error", error_msg)
            return
        
        # Build and run command
        cmd = self.build_command(selected_script, values)
        if cmd:
            cmd_str = " ".join(cmd)
            self.output_text.insert(tk.END, f"🔧 Running: {cmd_str}\n")
            self.output_text.see(tk.END)
            self.root.update()
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.stdout:
                    self.output_text.insert(tk.END, result.stdout + "\n")
                if result.stderr:
                    self.output_text.insert(tk.END, "STDERR: " + result.stderr + "\n")
                if result.returncode == 0:
                    self.output_text.insert(tk.END, "✅ Script completed successfully!\n")
                else:
                    self.output_text.insert(tk.END, f"❌ Script failed with return code {result.returncode}\n")
                
                self.output_text.see(tk.END)
            except Exception as e:
                self.output_text.insert(tk.END, f"❌ Error running script: {e}\n")
                self.output_text.see(tk.END)
    
    def clear_all(self):
        # Clear all input fields
        for widget in self.arg_widgets.values():
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
            elif isinstance(widget, tk.BooleanVar):
                widget.set(False)
        
        # Clear output
        self.output_text.delete(1.0, tk.END)
    
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def configure_buttons(self):
        """Ensure buttons are properly configured and visible"""
        # Force update of button display
        self.root.update_idletasks()
        
        # Debug: print button states
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, tk.Button):
                        print(f"Button text: '{grandchild.cget('text')}', visible: {grandchild.winfo_viewable()}")

def main():
    root = tk.Tk()
    app = MAPseqWizardGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
