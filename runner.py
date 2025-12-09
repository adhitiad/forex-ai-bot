import subprocess
import sys
import time

python_cmd = sys.executable

cmds = [
    [python_cmd, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
    [python_cmd, "ingestor.py"],
    [python_cmd, "brain.py"],
    [python_cmd, "executor.py"],
]

procs = []
try:
    print("🚀 Starting Forex Bot (OANDA)...")
    procs = [subprocess.Popen(c) for c in cmds]
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    for p in procs:
        p.terminate()
