import logging
import subprocess
import sys
import time

# Setup Logging agar terlihat rapi
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("Runner")

python_cmd = sys.executable

# Daftar Script yang akan dijalankan bersamaan
cmds = [
    # 1. API Backend (FastAPI)
    [python_cmd, "data_service.py"],
    # 2. Ingestor (gRPC Client)
    [python_cmd, "ingestor.py"],
    # 3. Backend API
    [python_cmd, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
    # 3. Brain V1 (Analisis Teknikal & Self-Healing)
    [python_cmd, "brain.py"],
    # 4. Brain V2 (Fusion AI: Berita + Sentimen + ATR)
    [python_cmd, "second_brain.py"],
    # 5. Executor (Eksekusi Order ke OANDA / Paper)
    [python_cmd, "executor.py"],
    # 6. Notifier (Kirim Alert ke Telegram)
    # [python_cmd, "notifier.py"],
]

procs = []


def start_processes():
    logger.info(f"🚀 Starting Forex AI Bot System ({len(cmds)} Services)...")
    processes = []
    for cmd in cmds:
        # Menjalankan setiap script sebagai subprocess
        p = subprocess.Popen(cmd)
        processes.append(p)
        logger.info(f"✅ Started: {' '.join(cmd)}")
        time.sleep(1)  # Beri jeda sedikit agar tidak crash barengan saat start
    return processes


def stop_processes(processes):
    logger.info("\n🛑 Shutting down all services...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except:
            p.kill()
    logger.info("👋 All systems offline.")


if __name__ == "__main__":
    try:
        procs = start_processes()

        # Loop utama agar script tidak mati
        while True:
            time.sleep(1)

            # Cek jika ada proses yang mati mendadak (Crash)
            for i, p in enumerate(procs):
                if p.poll() is not None:
                    dead_cmd = cmds[i]
                    logger.warning(f"⚠️ Process DIED: {dead_cmd[1]}. Restarting...")
                    # Restart proses yang mati
                    procs[i] = subprocess.Popen(dead_cmd)

    except KeyboardInterrupt:
        # Tangkap CTRL+C untuk mematikan semua dengan rapi
        stop_processes(procs)
    except Exception as e:
        logger.error(f"❌ Runner Error: {e}")
        stop_processes(procs)
