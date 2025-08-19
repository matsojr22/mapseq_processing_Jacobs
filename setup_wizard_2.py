import os
import subprocess
import platform
import shutil
import sys
import requests
import json
import time

GIT_URL = "https://github.com/Kim-Neuroscience-Lab/mapseq_processing_kimlab.git"
ENV_NAME = "mapseq_processing"

def get_latest_release_info():
    """Get the latest release information from GitHub API"""
    try:
        api_url = "https://api.github.com/repos/Kim-Neuroscience-Lab/mapseq_processing_kimlab/releases/latest"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        release_data = response.json()
        version = release_data['tag_name']
        
        # Find the MAPseq_Wizard_2.exe asset
        for asset in release_data['assets']:
            if asset['name'] == 'MAPseq_Wizard_2.exe':
                return asset['browser_download_url'], version
        
        # Fallback to constructing URL if asset not found
        return f"https://github.com/Kim-Neuroscience-Lab/mapseq_processing_kimlab/releases/download/{version}/MAPseq_Wizard_2.exe", version
        
    except Exception as e:
        print(f"⚠️ Could not fetch latest release info: {e}")
        # Fallback to a known working version
        return "https://github.com/Kim-Neuroscience-Lab/mapseq_processing_kimlab/releases/download/v0.2.0-beta/MAPseq_Wizard_2.exe", "v0.2.0-beta"

def prompt_install_path(default_path):
    print(f"\n📁 Default Miniconda install location: {default_path}")
    custom_path = input("Enter custom install path (or press Enter to use default): ").strip()
    return custom_path if custom_path else default_path

def get_miniconda_url():
    """Get the appropriate Miniconda URL for the current platform"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        if "64" in machine or "x86_64" in machine:
            return "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        else:
            return "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86.exe"
    elif system == "darwin":  # macOS
        if "arm" in machine or "aarch64" in machine:
            return "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else:
            return "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    elif system == "linux":
        if "arm" in machine or "aarch64" in machine:
            return "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        else:
            return "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    else:
        raise ValueError(f"Unsupported platform: {system} {machine}")

def install_miniconda(install_path):
    """Install Miniconda for the current platform"""
    system = platform.system().lower()
    url = get_miniconda_url()
    
    if system == "windows":
        installer = "Miniconda3.exe"
        print("🔍 Downloading Miniconda...")
        
        # Use curl if available, otherwise use requests
        try:
            subprocess.run(["curl", "-L", "-o", installer, url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Using Python requests to download...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(installer, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"🔧 Installing Miniconda to: {install_path}")
        subprocess.run([
            installer,
            "/InstallationType=JustMe",
            "/RegisterPython=0",
            "/AddToPath=1",
            "/S",
            f"/D={install_path}"
        ], check=True)
        
        # Clean up installer
        if os.path.exists(installer):
            os.remove(installer)
            
    else:  # macOS and Linux
        installer = url.split('/')[-1]
        print("🔍 Downloading Miniconda...")
        
        try:
            subprocess.run(["curl", "-L", "-o", installer, url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Using Python requests to download...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(installer, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"🔧 Installing Miniconda to: {install_path}")
        os.makedirs(install_path, exist_ok=True)
        
        # Make installer executable and run it
        os.chmod(installer, 0o755)
        subprocess.run([
            f"./{installer}",
            "-b",  # batch mode
            "-p", install_path,
            "-f"   # force installation
        ], check=True)
        
        # Clean up installer
        if os.path.exists(installer):
            os.remove(installer)

def get_conda_executable(install_path):
    """Get the conda executable path for the current platform"""
    system = platform.system().lower()
    
    if system == "windows":
        return os.path.join(install_path, "Scripts", "conda.exe")
    else:  # macOS and Linux
        return os.path.join(install_path, "bin", "conda")

def conda(cmd, conda_exe):
    """Run conda command with proper environment setup"""
    # Set environment variables for conda
    env = os.environ.copy()
    if platform.system().lower() != "windows":
        # Add conda to PATH for non-Windows systems
        conda_bin_dir = os.path.dirname(conda_exe)
        env['PATH'] = f"{conda_bin_dir}:{env.get('PATH', '')}"
    
    subprocess.run([conda_exe] + cmd, check=True, env=env)

def download_gui_exe(url, target_path):
    """Download the GUI executable"""
    print(f"⬇️  Downloading GUI .exe from: {url}")
    
    # Add retry logic for network issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ GUI exe saved to: {target_path}")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Download attempt {attempt + 1} failed: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise e

def create_env_and_setup(conda_exe, install_dir):
    """Create conda environment and setup the project"""
    print(f"\n📦 Creating environment '{ENV_NAME}'...")
    
    # Check if environment already exists
    try:
        conda(["env", "list"], conda_exe)
        env_list_output = subprocess.run([conda_exe, "env", "list"], 
                                       capture_output=True, text=True, check=True)
        if ENV_NAME in env_list_output.stdout:
            print(f"✅ Environment '{ENV_NAME}' already exists.")
            recreate = input("Do you want to recreate it? (y/n): ").strip().lower()
            if recreate == 'y':
                conda(["env", "remove", "-n", ENV_NAME, "-y"], conda_exe)
                conda(["create", "-y", "-n", ENV_NAME, "python=3.9", "pip"], conda_exe)
        else:
            conda(["create", "-y", "-n", ENV_NAME, "python=3.9", "pip"], conda_exe)
    except subprocess.CalledProcessError:
        # If env list fails, try to create the environment
        conda(["create", "-y", "-n", ENV_NAME, "python=3.9", "pip"], conda_exe)

    print("🔁 Adding channels: conda-forge, bioconda")
    try:
        conda(["config", "--add", "channels", "conda-forge"], conda_exe)
        conda(["config", "--add", "channels", "bioconda"], conda_exe)
    except subprocess.CalledProcessError:
        print("⚠️ Could not add conda channels. Continuing...")

    print("🐙 Cloning project repository...")
    git_dir = os.path.join(install_dir, "mapseq_processing_kimlab")
    if not os.path.exists(git_dir):
        try:
            subprocess.run(["git", "clone", GIT_URL], cwd=install_dir, check=True)
        except subprocess.CalledProcessError:
            print("⚠️ Git clone failed. Please ensure git is installed.")
            return
    else:
        print("📂 Repo already cloned.")
        # Update the repo
        try:
            subprocess.run(["git", "pull"], cwd=git_dir, check=True)
            print("✅ Repository updated.")
        except subprocess.CalledProcessError:
            print("⚠️ Could not update repository. Continuing...")

    # On Windows, download the latest GUI .exe; on other OS, skip and use Python script
    current_system = platform.system()
    if current_system == "Windows":
        gui_exe_url, version = get_latest_release_info()
        print(f"📦 Latest version detected: {version}")
        gui_exe_path = os.path.join(git_dir, "MAPseq_Wizard_2.exe")
        if not os.path.exists(gui_exe_path):
            download_gui_exe(gui_exe_url, gui_exe_path)
        else:
            print(f"✅ GUI exe already exists at: {gui_exe_path}")
    else:
        print("ℹ️ Skipping .exe download on non-Windows. Use: python MAPseq_Wizard_2.py")

    # Install dependencies
    requirements_path = os.path.join(git_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        print("📄 Installing dependencies from cloned requirements.txt...")
        
        # Add requests to requirements if not present
        with open(requirements_path, 'r') as f:
            requirements_content = f.read()
        
        if 'requests' not in requirements_content:
            print("📄 Adding 'requests' to requirements...")
            with open(requirements_path, 'a') as f:
                f.write('\nrequests\n')
        
        try:
            # First, uninstall any existing PySimpleGUI to avoid conflicts
            print("🔄 Uninstalling old PySimpleGUI version...")
            subprocess.run([
                conda_exe, "run", "-n", ENV_NAME, "pip", "uninstall", "PySimpleGUI", "-y"
            ], check=False)  # Don't fail if not installed
            
            # Install dependencies from requirements.txt
            print("📄 Installing dependencies from requirements.txt...")
            subprocess.run([
                conda_exe, "run", "-n", ENV_NAME, "pip", "install", "-r", requirements_path
            ], check=True)
            
            # Install PySimpleGUI version that doesn't require account creation
            print("📄 Installing PySimpleGUI (no account required)...")
            subprocess.run([
                conda_exe, "run", "-n", ENV_NAME, "pip", "install", 
                "PySimpleGUI==4.60.5"
            ], check=True)
            
            print("✅ Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Some dependencies failed to install: {e}")
            print("You may need to install them manually.")
    else:
        print(f"⚠️ No requirements.txt found in {git_dir}")
        
        # Install basic requirements manually
        basic_requirements = [
            "requests", "numpy", "pandas", "matplotlib", 
            "scipy", "scikit-learn", "seaborn", "sympy", "adjustText", 
            "upsetplot", "statsmodels", "networkx", "regex"
        ]
        print("📄 Installing basic dependencies...")
        for req in basic_requirements:
            try:
                subprocess.run([
                    conda_exe, "run", "-n", ENV_NAME, "pip", "install", req
                ], check=True)
                print(f"✅ Installed {req}")
            except subprocess.CalledProcessError:
                print(f"⚠️ Failed to install {req}")
        
        # Install PySimpleGUI version that doesn't require account creation
        print("📄 Installing PySimpleGUI (no account required)...")
        try:
            subprocess.run([
                conda_exe, "run", "-n", ENV_NAME, "pip", "install", 
                "PySimpleGUI==4.60.5"
            ], check=True)
            print("✅ Installed PySimpleGUI")
        except subprocess.CalledProcessError:
            print("⚠️ Failed to install PySimpleGUI")

def main():
    try:
        system = platform.system()
        print(f"🖥️  Detected platform: {system}")
        
        if system == "Windows":
            default_path = os.path.expanduser("~\\Miniconda3")
        elif system == "Darwin":  # macOS
            default_path = os.path.expanduser("~/miniconda3")
        elif system == "Linux":
            default_path = os.path.expanduser("~/miniconda3")
        else:
            print(f"❌ Unsupported platform: {system}")
            input("Press Enter to exit...")
            return

        install_path = prompt_install_path(default_path)

        if not os.path.isdir(install_path):
            os.makedirs(install_path, exist_ok=True)

        conda_exe = get_conda_executable(install_path)

        if not os.path.exists(conda_exe):
            print("\n❗ Conda not found. Installing Miniconda...")
            install_miniconda(install_path)
        else:
            print("✅ Conda already installed.")

        if not os.path.exists(conda_exe):
            raise FileNotFoundError(f"conda executable not found at {conda_exe}")

        create_env_and_setup(conda_exe, install_path)
        
        # Show next steps
        print("\n" + "="*60)
        print("✅ Setup completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Navigate to the project directory:")
        print(f"   cd {os.path.join(install_path, 'mapseq_processing_kimlab')}")
        print("2. Run the MAPseq Wizard 2.0:")
        if system == "Windows":
            print("   MAPseq_Wizard_2.exe")
        else:
            print("   python MAPseq_Wizard_2.py")
        print("\nOr activate the conda environment and run:")
        print(f"   conda activate {ENV_NAME}")
        print("   python MAPseq_Wizard_2.py")
        print("="*60)

    except subprocess.CalledProcessError as e:
        print(f"\n🚨 Subprocess failed: {e}")
        print("This might be due to network issues or missing dependencies.")
    except Exception as e:
        print(f"\n⚠️ Unexpected error: {e}")
        print("Please check your internet connection and try again.")

    input("\n📝 Press Enter to exit...")

if __name__ == "__main__":
    main()
