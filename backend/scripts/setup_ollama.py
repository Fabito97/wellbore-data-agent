import platform
import subprocess
import time
import shutil
import requests
import sys
import os
from pathlib import Path

# Configuration
OLLAMA_MODEL = "phi3:mini"
OLLAMA_URL = "http://localhost:11434"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000


def is_windows():
    return platform.system() == "Windows"


def is_wsl():
    return "microsoft" in platform.release().lower()


def run_command(cmd, check=True, shell=False):
    """Run command with basic error handling"""
    try:
        result = subprocess.run(cmd, check=check, shell=shell)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print(f"❌ Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_wsl_command(cmd):
    """Run command in WSL"""
    return run_command(["wsl", "bash", "-c", cmd])


def wsl_has_command(cmd):
    """Check if command exists in WSL"""
    result = subprocess.run(
        ["wsl", "bash", "-c", f"command -v {cmd}"],
        capture_output=True
    )
    return result.returncode == 0


# def wsl_is_available():
#     """Check if WSL is installed and working"""
#     if not shutil.which("wsl"):
#         return False
#     result = subprocess.run(["wsl.exe", "echo", "test"], capture_output=True, timeout=5)
#     return result.returncode == 0



def is_wsl_available():
    """Check if WSL is installed and has at least one configured distro."""
    if not shutil.which("wsl"):
        return False
    try:
        result = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except subprocess.SubprocessError:
        return False


def show_run_as_admin_message():
    print("⚠️  Please run this in PowerShell (Admin):")
    print("  1. Search 'PowerShell' in Start Menu")
    print("  2. Right-click → 'Run as Administrator'")
    print("  3. Run this script again\n")


def install_wsl():
    """Install WSL with proper shell detection"""
    print("\n⚠️  WSL not detected. This script needs WSL for automated Ollama setup.")

    # Check if we're in PowerShell
    if not os.getenv('PSModulePath'):
        show_run_as_admin_message()
        sys.exit(1)

    consent = input("\nInstall WSL? Requires admin rights and reboot [y/N]: ").strip().lower()
    if consent != "y":
        print("❌ Cannot proceed without WSL.")
        print("Alternative: Install Ollama manually from https://ollama.com/download\n")
        sys.exit(1)

    print("\n📦 Installing WSL...")
    if run_command(["powershell", "-Command", "wsl --install"], check=False):
        print("\n✅ WSL installed. Please reboot and rerun this script.\n")
    else:
        print("\n❌ WSL installation failed.\n")
        show_run_as_admin_message()
        print("Try manually: https://aka.ms/wslinstall if the error persists.")
    sys.exit(0)


def install_ollama_in_wsl():
    """Install Ollama in WSL"""
    if wsl_has_command("ollama"):
        print("✅ Ollama already installed in WSL.")
        return True

    print("📦 Installing Ollama in WSL...")
    return run_wsl_command("curl -fsSL https://ollama.com/install.sh | sh")

def is_ollama_running():
    """Check if Ollama is running"""
    try:
        return requests.get(OLLAMA_URL, timeout=2).status_code == 200
    except:
        return False

def start_ollama_in_wsl():
    """Start Ollama in WSL"""
    if is_ollama_running():
        print("✅ Ollama already running.")
        return True

    print("🚀 Starting Ollama in WSL...")
    subprocess.Popen(
        ["wsl", "bash", "-c", "nohup ollama serve > /dev/null 2>&1 &"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return True


def pull_model_in_wsl(model=OLLAMA_MODEL):
    """Pull model in WSL"""
    result = subprocess.run(
        ["wsl", "bash", "-c", "ollama list"],
        capture_output=True,
        text=True
    )

    if model in result.stdout:
        print(f"✅ Model '{model}' already available.")
        return True

    print(f"📥 Pulling model '{model}'... (this may take a few minutes)")
    success = run_wsl_command(f"ollama pull {model}")
    if not success:
        print(f"❌ Failed to pull model '{model}'")
        print("   Common causes:")
        print("   - Network timeout (slow connection)")
        print("   - Firewall/proxy blocking ollama.ai")
        print("   - No internet connection")
    return success


def install_ollama():
    """Install Ollama on native Linux/macOS"""
    if shutil.which("ollama"):
        print("✅ Ollama already installed.")
        return True

    print("📦 Installing Ollama...")
    if is_wsl() or platform.system() == "Linux":
        return run_command(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
    elif platform.system() == "Darwin":
        if not shutil.which("brew"):
            print("❌ Homebrew not found. Install from https://brew.sh")
            return False
        return run_command(["brew", "install", "ollama"])
    return False


def start_ollama():
    """Start Ollama"""
    if is_ollama_running():
        print("✅ Ollama already running.")
        return True

    print("🚀 Starting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return True




def wait_for_ollama(timeout=10):
    """Wait for Ollama to be ready"""
    print("⏳ Waiting for Ollama...")
    for i in range(timeout):
        if is_ollama_running():
            print("✅ Ollama ready.")
            return True
        time.sleep(1)
        print(f"No response, retrying after attempt #{i+1}...")
    print(f"❌ Ollama not responding after {timeout} attempts.")
    return False


def pull_model(model=OLLAMA_MODEL):
    """Pull model"""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)

    if model in result.stdout:
        print(f"✅ Model '{model}' already available.")
        return True

    print(f"📥 Pulling model '{model}'... (this may take a few minutes)")
    success = run_command(["ollama", "pull", model])
    if not success:
        print(f"❌ Failed to pull model '{model}'")
        print("   Common causes:")
        print("   - Network timeout (slow connection)")
        print("   - Firewall/proxy blocking ollama.ai")
        print("   - No internet connection")
    return success


def find_python_in_venv():
    """Find Python executable in .venv"""
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent

    # Common venv locations
    venv_paths = [
        backend_dir / ".venv",
        backend_dir / ".venv_wsl",
        backend_dir / "venv",
    ]

    for venv_path in venv_paths:
        if not venv_path.exists():
            continue

        if is_windows():
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"

        if python_exe.exists():
            return python_exe

    return None


def start_backend():
    """Start the backend service and hand over control to it"""
    print("🚀 Starting backend...\n")

    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent

    # Try to find Python in venv first
    python_exe = find_python_in_venv()

    if not python_exe:
        print("❌ No virtual environment found!")
        print("   Expected .venv, venv, .venv_wsl in backend directory")
        print(f"   Backend dir: {backend_dir}")
        print("\n   Create one with: python3 -m venv .venv or python -m venv .venv")
        print("\n   Activate the environment with: source .venv_wsl/bin/activate or Script .venv/python/Activate")
        print("   Then install: pip install -r requirements.txt")
        return False

    print(f"   Using: {python_exe}")

    # Verify uvicorn is installed
    check = subprocess.run(
        [str(python_exe), "-m", "uvicorn", "--version"],
        capture_output=True
    )

    if check.returncode != 0:
        print(f"❌ uvicorn not installed in venv!")
        print(f"   Install with: {python_exe} -m pip install uvicorn")
        return False

    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print(f"\n🌐 Backend: http://{BACKEND_HOST}:{BACKEND_PORT}")
    print(f"🤖 Ollama: {OLLAMA_URL}")
    print("\nStarting uvicorn... (Press Ctrl+C to stop)")
    print("=" * 60 + "\n")

    # Start uvicorn in foreground - this blocks and gives terminal control to uvicorn
    try:
        subprocess.run([
            str(python_exe),
            "-m", "uvicorn",
            "app.main:app",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--reload"
        ], cwd=str(backend_dir))
    except KeyboardInterrupt:
        print("\n\n⚠️  Backend stopped by user!\n")
        return True
    except Exception as e:
        print(f"\n❌ Backend crashed: {e}\n")
        return False

    return True


def main():
    print("\n" + "=" * 60)
    print("🔍 OLLAMA + BACKEND SETUP")
    print("=" * 60 + "\n")

    if is_windows():
        print("Platform: Windows\n")

        if not is_wsl_available():
            install_wsl()

        print("✅ WSL available.")
        print("💡 Ollama runs in WSL, backend in Windows\n")

        if not install_ollama_in_wsl():
            print("\n❌ Setup failed: Could not install Ollama")
            sys.exit(1)

        if not start_ollama_in_wsl():
            print("\n❌ Setup failed: Could not start Ollama")
            sys.exit(1)

        if not wait_for_ollama():
            print("\n❌ Setup failed: Ollama not responding")
            sys.exit(1)

        if not pull_model_in_wsl():
            print("\n❌ Setup failed: Model is required for the app to work")
            print(f"\n   Try manually: wsl ollama pull {OLLAMA_MODEL}")
            print("   Then rerun this script\n")
            sys.exit(1)

        # This will block and hand control to uvicorn
        start_backend()


    elif platform.system() in ["Linux", "Darwin", "Windows"]:
        if is_wsl():
            print(f"Platform: {platform.system()}\n")
        else:
            print(f"Platform: {platform.system()}\n")

        if not install_ollama():
            print("\n❌ Setup failed: Could not install Ollama")
            sys.exit(1)

        if not start_ollama():
            print("\n❌ Setup failed: Could not start Ollama")
            sys.exit(1)

        if not wait_for_ollama():
            print("\n❌ Setup failed: Ollama not responding")
            sys.exit(1)

        if not pull_model():
            print(f"\n❌ Setup failed: Model {OLLAMA_MODEL} is required for the app to work")
            print(f"\n   Try manually: ollama pull {OLLAMA_MODEL}")
            print("   Then rerun this script\n")
            sys.exit(1)

        # This will block and hand control to uvicorn
        start_backend()

    else:
        print("❌ Unsupported platform.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted!\n")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)