import PySimpleGUI as sg
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
            {"name": "output_prefix", "type": "input", "required": False, "default": "motif_clustering", "help": "Prefix for output files"},
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
            {"name": "allen_file", "type": "file", "required": True, "default": "", "help": "Allen dataset file", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "vsv_file", "type": "file", "required": True, "default": "", "help": "VSV dataset file", "file_types": (("CSV Files", "*.csv"),)},
            {"name": "output_dir", "type": "folder", "required": False, "default": "allen_vsv_comparison", "help": "Output directory for comparison results"}
        ]
    }
}

def create_argument_layout(script_name):
    """Create layout for script arguments"""
    if script_name not in SCRIPT_CONFIGS:
        return []
    
    config = SCRIPT_CONFIGS[script_name]
    layout = []
    
    for arg in config["arguments"]:
        arg_name = arg["name"]
        arg_type = arg["type"]
        required = arg["required"]
        default = arg["default"]
        help_text = arg["help"]
        
        # Create label with required indicator
        label_text = f"{arg_name}:"
        if required:
            label_text += " *"
        
        if arg_type == "input":
            layout.append([
                sg.Text(label_text), 
                sg.Input(key=f"arg_{arg_name}", default_text=default, size=(40, 1)),
                sg.Text(help_text, size=(50, 1), text_color='gray')
            ])
        elif arg_type == "file":
            file_types = arg.get("file_types", (("All Files", "*.*"),))
            layout.append([
                sg.Text(label_text), 
                sg.Input(key=f"arg_{arg_name}", default_text=default, size=(30, 1)),
                sg.FileBrowse(file_types=file_types),
                sg.Text(help_text, size=(40, 1), text_color='gray')
            ])
        elif arg_type == "folder":
            layout.append([
                sg.Text(label_text), 
                sg.Input(key=f"arg_{arg_name}", default_text=default, size=(30, 1)),
                sg.FolderBrowse(),
                sg.Text(help_text, size=(40, 1), text_color='gray')
            ])
        elif arg_type == "checkbox":
            layout.append([
                sg.Text(label_text), 
                sg.Checkbox("", key=f"arg_{arg_name}", default=default),
                sg.Text(help_text, size=(50, 1), text_color='gray')
            ])
    
    return layout

def build_command(script_name, values):
    """Build the command to run the selected script"""
    if script_name not in SCRIPT_CONFIGS:
        return None
    
    config = SCRIPT_CONFIGS[script_name]
    cmd = ["conda", "run", "-n", "mapseq_processing", "python", script_name]
    
    for arg in config["arguments"]:
        arg_name = arg["name"]
        arg_type = arg["type"]
        required = arg["required"]
        
        value = values.get(f"arg_{arg_name}", "")
        
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

def main():
    sg.theme('LightBlue2')
    
    # Create the main layout
    layout = [
        [sg.Text("MAPseq Processing Wizard 2.0", font=('Helvetica', 16, 'bold'))],
        [sg.Text("Select a script to run:", font=('Helvetica', 12))],
        [sg.Combo(
            list(SCRIPT_CONFIGS.keys()), 
            key='script_selector', 
            size=(50, 1),
            enable_events=True,
            default_value=list(SCRIPT_CONFIGS.keys())[0] if SCRIPT_CONFIGS else ""
        )],
        [sg.Text("", key='script_description', size=(80, 2), text_color='blue')],
        [sg.HorizontalSeparator()],
        [sg.Text("Script Arguments:", font=('Helvetica', 12, 'bold'))],
        [sg.Column([], key='arguments_column', scrollable=True, vertical_scroll_only=True, size=(800, 400))],
        [sg.HorizontalSeparator()],
        [sg.Button("Run Script", size=(15, 1)), sg.Button("Clear All", size=(15, 1)), sg.Exit(size=(15, 1))],
        [sg.Multiline(size=(80, 10), key='output', disabled=True, autoscroll=True)]
    ]
    
    window = sg.Window("MAPseq Wizard 2.0", layout, resizable=True, finalize=True)
    
    # Initial script selection
    if SCRIPT_CONFIGS:
        first_script = list(SCRIPT_CONFIGS.keys())[0]
        window['script_selector'].update(first_script)
        window['script_description'].update(SCRIPT_CONFIGS[first_script]["description"])
        
        # Create initial argument layout
        arg_layout = create_argument_layout(first_script)
        window['arguments_column'].update(arg_layout)
    
    while True:
        event, values = window.read()
        
        if event in (None, 'Exit'):
            break
        
        elif event == 'script_selector':
            # Update description and arguments when script changes
            selected_script = values['script_selector']
            if selected_script in SCRIPT_CONFIGS:
                window['script_description'].update(SCRIPT_CONFIGS[selected_script]["description"])
                arg_layout = create_argument_layout(selected_script)
                window['arguments_column'].update(arg_layout)
        
        elif event == 'Run Script':
            selected_script = values['script_selector']
            if not selected_script:
                sg.popup_error("Please select a script first!")
                continue
            
            # Validate required arguments
            config = SCRIPT_CONFIGS[selected_script]
            missing_args = []
            for arg in config["arguments"]:
                if arg["required"]:
                    value = values.get(f"arg_{arg['name']}", "")
                    if not value:
                        missing_args.append(arg['name'])
            
            if missing_args:
                sg.popup_error(f"Missing required arguments: {', '.join(missing_args)}")
                continue
            
            # Build and run command
            cmd = build_command(selected_script, values)
            if cmd:
                cmd_str = " ".join(cmd)
                window['output'].update(f"🔧 Running: {cmd_str}\n")
                window.refresh()
                
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.stdout:
                        window['output'].update(result.stdout + "\n", append=True)
                    if result.stderr:
                        window['output'].update("STDERR: " + result.stderr + "\n", append=True)
                    if result.returncode == 0:
                        window['output'].update("✅ Script completed successfully!\n", append=True)
                    else:
                        window['output'].update(f"❌ Script failed with return code {result.returncode}\n", append=True)
                except Exception as e:
                    window['output'].update(f"❌ Error running script: {e}\n", append=True)
        
        elif event == 'Clear All':
            # Clear all input fields
            for key in values:
                if key.startswith('arg_'):
                    window[key].update('')
            window['output'].update('')
    
    window.close()

if __name__ == "__main__":
    main()
