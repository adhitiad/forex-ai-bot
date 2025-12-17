import logging
import os
import subprocess
import sys
import time

# Setup Logging agar terlihat rapi
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("Enterprise-Runner")

# Path ke Python Interpreter saat ini (vEnv)
python_cmd = sys.executable

# --- DAFTAR LAYANAN ENTERPRISE ---
# Format: {"name": "Nama Service", "cmd": ["command", "arg1", ...]}
services = [
    # 1. MATA & TELINGA (Data Ingestion)
    {"name": "Ingestor (MT5)", "cmd": [python_cmd, "ingestor.py"]},
    {"name": "Macro Guardian", "cmd": [python_cmd, "macro_engine.py"]},
    # 2. OTAK & ANALISIS (The Council)
    {"name": "Brain V1 (Technical)", "cmd": [python_cmd, "brain.py"]},
    {"name": "Brain RL (Adaptive)", "cmd": [python_cmd, "brain_rl.py"]},
    {"name": "LLM Strategist (Fundamental)", "cmd": [python_cmd, "llm_strategist.py"]},
    # 3. PENGAMBIL KEPUTUSAN (The Chairman)
    {"name": "Fusion Engine (Core Logic)", "cmd": [python_cmd, "fusion_engine.py"]},
    # 4. EKSEKUTOR (The Hand)
    {"name": "Executor (MT5)", "cmd": [python_cmd, "executor.py"]},
    # # 5. KOMUNIKASI & MONITORING
    # {"name": "Notifier (Telegram)", "cmd": [python_cmd, "notifier.py"]},
    {
        "name": "API Gateway",
        "cmd": [
            python_cmd,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
    },
    # {
    #     "name": "Control Tower (Dashboard)",
    #     # Streamlit butuh command khusus "streamlit run"
    #     # Kita panggil via modul python -m streamlit
    #     "cmd": [
    #         python_cmd,
    #         "-m",
    #         "streamlit",
    #         "run",
    #         "dashboard.py",
    #         "--server.port",
    #         "8501",
    #         "--server.headless",
    #         "true",
    #     ],
    # },
]

# Dictionary untuk menyimpan proses yang aktif
active_procs = {}


def start_service(svc):
    """Menjalankan satu service"""
    try:
        logger.info(f"🚀 Starting {svc['name']}...")
        # Start subprocess (non-blocking)
        p = subprocess.Popen(svc["cmd"], shell=False)
        active_procs[svc["name"]] = p
    except Exception as e:
        logger.error(f"❌ Failed to start {svc['name']}: {e}")


def stop_all():
    """Mematikan semua service dengan rapi"""
    logger.info("\n🛑 SHUTDOWN SEQUENCE INITIATED...")
    for name, p in active_procs.items():
        try:
            logger.info(f"Killing {name}...")
            p.terminate()
            p.wait(timeout=3)
        except:
            p.kill()  # Paksa bunuh jika bandel
    logger.info("👋 All systems offline.")


def monitor_loop():
    """Loop utama untuk menjaga service tetap hidup (Self-Healing)"""

    # 1. Start Awal Semua Service
    logger.info(f"⚡ Booting up Forex AI Enterprise ({len(services)} Services)...")
    for svc in services:
        start_service(svc)
        # Beri jeda sedikit agar Redis tidak kaget (Connection spike)
        time.sleep(1.5)

    logger.info("✅ System Fully Operational. Press CTRL+C to stop.")

    # 2. Watchdog Loop
    try:
        while True:
            time.sleep(5)  # Cek setiap 5 detik

            for svc in services:
                name = svc["name"]
                proc = active_procs.get(name)

                # Cek apakah proses mati (poll() returns exit code if dead, None if alive)
                if proc is None or proc.poll() is not None:
                    exit_code = proc.poll() if proc else "N/A"
                    logger.warning(
                        f"⚠️ ALERT: Service '{name}' DIED (Code: {exit_code}). Restarting in 3s..."
                    )

                    # Restart Service
                    time.sleep(3)
                    start_service(svc)

    except KeyboardInterrupt:
        stop_all()
    except Exception as e:
        logger.error(f"Critical Runner Error: {e}")
        stop_all()


if __name__ == "__main__":
    # Pastikan folder logs ada (opsional)
    if not os.path.exists("logs"):
        os.makedirs("logs")

    monitor_loop()
