import subprocess, time
from config import GREEN, RED, RESET

def start_pi_stream(user, host, venv, script):
    cmd = f"bash -c 'source {venv} && nohup python3 {script} >/dev/null 2>&1 &'"
    try:
        subprocess.Popen(["ssh", "-f", f"{user}@{host}", cmd])
        print(f"{GREEN}[INFO]{RESET} Starting Pi stream...")
        time.sleep(5)
    except Exception as e:
        print(f"{RED}[ERROR]{RESET} Failed to start Pi stream: {e}")

def stop_pi_stream(user, host, script):
    try:
        subprocess.run(f"ssh {user}@{host} pkill -f '{script}'", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{RED}[INFO]{RESET} Pi stream stopped.")
    except Exception as e:
        print(f"{RED}[ERROR]{RESET} Failed to stop Pi stream: {e}")