import psutil
import os


def kill_ollama_installer():
    for proc in psutil.process_iter(['pid', 'name']):
        if "OllamaSetup.exe" in proc.info['name']:
            print(f"🛑 Killing process: {proc.info['name']} (PID {proc.info['pid']})")
            proc.kill()
            return True

installer_path = os.path.join(os.environ["TEMP"], "OllamaSetup.exe")
if os.path.exists(installer_path):
    try:
        os.remove(installer_path)
        print("🧹 Removed existing installer.")
    except PermissionError:
        print("⚠️ Installer is locked. Closing any running Ollama setup or restart your system.")
        killed = kill_ollama_installer()
        print(f"Closing Installer was {"successful" if killed else "unsuccessful"}")



