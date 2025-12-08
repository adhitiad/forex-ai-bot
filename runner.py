import subprocess, time, sys
from config import settings

python_cmd = sys.executable

# Choose ingestor based on configuration
ingestor_script = "yfinance_ingestor.py" if settings.USE_YFINANCE else "ingestor.py"

cmds = [
    [python_cmd, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
    [python_cmd, ingestor_script],
    [python_cmd, "brain.py"],
    [python_cmd, "executor.py"],
]

procs = []
try:
    procs = [subprocess.Popen(c) for c in cmds]
    print("🚀 All Services Started. Press Ctrl+C to Stop.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    for p in procs:
        p.terminate()
