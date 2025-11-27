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


def run_wsl_command(cmd, capture_output=False):
    """Run command in WSL"""
    if capture_output:
        result = subprocess.run(
            ["wsl", "bash", "-c", cmd],
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout
    return run_command(["wsl", "bash", "-c", cmd])


def wsl_has_command(cmd):
    """Check if command exists in WSL"""
    result = subprocess.run(
        ["wsl", "bash", "-c", f"command -v {cmd}"],
        capture_output=True
    )
    return result.returncode == 0


def wsl_is_available():
    """Check if WSL is installed and working"""
    if not shutil.which("wsl"):
        return False
    try:
        result = subprocess.run(["wsl", "echo", "test"], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
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


def spawn_wsl_terminal_for_ollama():
    """Spawn WSL terminal to run ollama serve (will be closed after ready)"""
    print("🚀 Starting Ollama in WSL terminal...")

    # Try Windows Terminal first (wt.exe), fallback to cmd
    if shutil.which("wt.exe"):
        # Spawn as new tab in Windows Terminal
        process = subprocess.Popen([
            'wt.exe', '-w', '0', 'new-tab', '--title', 'Ollama Setup',
            'wsl', 'bash', '-c', 'ollama serve'
        ])
        print("   Opened in Windows Terminal (new tab)")
    else:
        # Fallback to cmd
        process = subprocess.Popen([
            'cmd', '/c', 'start', 'cmd', '/k', 'wsl bash -c "ollama serve"'
        ])
        print("   Opened in CMD window")

    return process


def pull_model_in_wsl(model=OLLAMA_MODEL):
    """Pull model in WSL"""
    success, output = run_wsl_command("ollama list", capture_output=True)

    if success and model in output:
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


def spawn_terminal_for_ollama():
    """Spawn terminal to run ollama serve on Linux/macOS"""
    print("🚀 Starting Ollama in terminal...")

    # Try different terminal emulators
    terminals = [
        ('gnome-terminal', ['gnome-terminal', '--', 'bash', '-c', 'ollama serve; read -p "Press enter to close..."']),
        ('xterm', ['xterm', '-e', 'bash', '-c', 'ollama serve; read -p "Press enter to close..."']),
        ('konsole', ['konsole', '-e', 'bash', '-c', 'ollama serve; read -p "Press enter to close..."']),
    ]

    for name, cmd in terminals:
        if shutil.which(name):
            process = subprocess.Popen(cmd)
            print(f"   Opened in {name}")
            return process

    # Fallback: run in background
    print("   No terminal emulator found, running in background")
    process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process


def is_ollama_running():
    """Check if Ollama is running"""
    try:
        return requests.get(OLLAMA_URL, timeout=2).status_code == 200
    except:
        return False


def wait_for_ollama(timeout=30, show_progress=True):
    """Wait for Ollama to be ready"""
    if show_progress:
        print("⏳ Waiting for Ollama to be ready...")

    for i in range(timeout):
        if is_ollama_running():
            if show_progress:
                print("✅ Ollama is ready!")
            return True

        if show_progress and i > 0 and i % 5 == 0:
            print(f"   Still waiting... ({i}s)")

        time.sleep(1)

    print(f"❌ Ollama not responding after {timeout}s.")
    return False


def pull_model(model=OLLAMA_MODEL):
    """Pull model on native system"""
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
        backend_dir / "venv",
        backend_dir / "env",
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


def is_port_in_use(port):
    """Check if a port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_backend_in_current_terminal():
    """Start backend in the current terminal (blocking)"""
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent

    python_exe = find_python_in_venv()

    if not python_exe:
        print("❌ No virtual environment found!")
        print("   Expected .venv, venv, or env in backend directory")
        print(f"   Backend dir: {backend_dir}")
        print("\n   Create one with: python3 -m venv .venv")
        print("   Then install: .venv/bin/pip install -r requirements.txt")
        return False

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
    print("\n🚀 Starting backend... (Press Ctrl+C to stop)")
    print("=" * 60 + "\n")

    # Start uvicorn in current terminal (blocking)
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
        print("\n\n⚠️  Backend stopped. Goodbye!\n")
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

        # Check WSL
        if not wsl_is_available():
            install_wsl()

        print("✅ WSL available.")
        print("💡 Ollama runs in WSL, backend in Windows\n")

        # Install Ollama in WSL
        if not install_ollama_in_wsl():
            print("\n❌ Setup failed: Could not install Ollama")
            sys.exit(1)

        # Check if Ollama is already running
        if is_ollama_running():
            print("✅ Ollama already running.")
        else:
            # Spawn WSL terminal with Ollama
            ollama_process = spawn_wsl_terminal_for_ollama()

            # Wait for Ollama to be ready
            if not wait_for_ollama(timeout=30):
                print("\n❌ Setup failed: Ollama not responding")
                print("   Check the WSL terminal for errors")
                sys.exit(1)

            # Close the WSL terminal now that Ollama is ready
            print("✅ Ollama ready! Closing WSL setup terminal...")
            try:
                # Try to close the WSL tab (may not work reliably)
                ollama_process.terminate()
                time.sleep(1)
            except:
                pass
            print("   (You can manually close the WSL tab if still open)\n")

        # Pull model
        if not pull_model_in_wsl():
            print("\n❌ Setup failed: Model is required for the app to work")
            print(f"   Try manually: wsl ollama pull {OLLAMA_MODEL}")
            print("   Then rerun this script\n")
            sys.exit(1)

        # Check if backend is already running
        if is_port_in_use(BACKEND_PORT):
            print(f"\n⚠️  Port {BACKEND_PORT} already in use!")
            print(f"   Backend might already be running at http://{BACKEND_HOST}:{BACKEND_PORT}")
            print("   Close existing backend or change BACKEND_PORT in the script\n")
            sys.exit(1)

        # Start backend in current terminal (blocks here)
        start_backend_in_current_terminal()

    elif platform.system() in ["Linux", "Darwin"]:
        print(f"Platform: {platform.system()}\n")

        # Install Ollama
        if not install_ollama():
            print("\n❌ Setup failed: Could not install Ollama")
            sys.exit(1)

        # Check if Ollama is already running
        if is_ollama_running():
            print("✅ Ollama already running.")
        else:
            # Spawn terminal with Ollama
            ollama_process = spawn_terminal_for_ollama()

            # Wait for Ollama to be ready
            if not wait_for_ollama(timeout=30):
                print("\n❌ Setup failed: Ollama not responding")
                print("   Check the Ollama terminal for errors")
                sys.exit(1)

        # Pull model
        if not pull_model():
            print("\n❌ Setup failed: Model is required for the app to work")
            print(f"   Try manually: ollama pull {OLLAMA_MODEL}")
            print("   Then rerun this script\n")
            sys.exit(1)

        # Check if backend is already running
        if is_port_in_use(BACKEND_PORT):
            print(f"\n⚠️  Port {BACKEND_PORT} already in use!")
            print(f"   Backend might already be running at http://{BACKEND_HOST}:{BACKEND_PORT}")
            print("   Close existing backend or change BACKEND_PORT in the script\n")
            sys.exit(1)

        # Start backend in current terminal (blocks here)
        start_backend_in_current_terminal()

    else:
        print("❌ Unsupported platform.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted. Goodbye!\n")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)