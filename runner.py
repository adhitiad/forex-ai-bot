import logging
import os
import signal
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Runner All - Terminal")

# Daftar proses yang berjalan
processes = []


def stream_logs(process, prefix):
    """Helper (opsional) untuk logger.info output subprocess ke terminal utama"""
    # Implementasi sederhana: Output langsung ke stdout masing-masing
    for line in iter(process.stdout.readline, b""):
        logger.info(f"{prefix}: {line.decode().strip()}")
    process.stdout.close()


def start_process(command, name):
    """Meluncurkan script python sebagai subprocess"""
    logger.info(f"🚀 Starting {name}...")
    # Menggunakan sys.executable untuk memastikan environment python yang sama dipakai
    p = subprocess.Popen(
        command,
        shell=True if os.name == "nt" else False,  # Windows butuh shell=True kadang
        cwd=os.getcwd(),
    )
    processes.append((name, p))
    return p


def kill_all():
    """Mematikan semua proses saat script dihentikan"""
    logger.info("\n🛑 Stopping all services...")
    for name, p in processes:
        logger.info(f"   Terminating {name}...")
        if os.name == "nt":
            # Windows kill force
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)])
        else:
            # Unix kill
            p.terminate()
    logger.info("✅ All services stopped.")


def main():
    try:
        # 1. Cek Kebutuhan Dasar
        if not os.path.exists(".env"):
            logger.info("❌ File .env tidak ditemukan! Buat dulu.")
            return

        if not os.path.exists("data/trained_model.pth"):
            logger.info(
                "⚠️ WARNING: Model belum dilatih! Jalankan 'python train.py' dulu agar hasil maksimal."
            )
            time.sleep(2)  # Beri waktu baca warning

        logger.info("⚡ ACTIVATING HFT ECOSYSTEM ⚡")
        logger.info("==============================")

        # 2. Jalankan Data Ingestor (Penyuplai Data)
        # Ingestor wajib jalan agar Brain dapat data dari Redis
        start_process([sys.executable, "ingestor.py"], "Data Ingestor")
        time.sleep(2)  # Beri jeda agar Ingestor connect Redis duluan

        # 3. Jalankan Brain (Otak AI)
        start_process([sys.executable, "brain.py"], "AI Brain")
        time.sleep(2)

        # 4. Jalankan Main API (Uvicorn Server)
        # Menggunakan uvicorn langsung via command line
        uvicorn_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        start_process(uvicorn_cmd, "API Server")

        logger.info("==============================")
        logger.info("✅ System Running!")
        logger.info("   -> API: http://localhost:8000")
        logger.info("   -> WS:  ws://localhost:8000/ws")
        logger.info("   -> Logs: Check terminal output below")
        logger.info("press Ctrl+C to stop everything")
        logger.info("==============================")

        # 5. Monitor Loop (Agar script utama tidak exit)
        while True:
            time.sleep(1)
            # Cek jika ada proses yang crash (zombie)
            for name, p in processes:
                if p.poll() is not None:
                    logger.info(
                        f"❌ {name} died unexpectedly! Restarting system suggested."
                    )
                    kill_all()
                    return

    except KeyboardInterrupt:
        kill_all()


if __name__ == "__main__":
    # Handle sinyal kill dari OS (misal Docker stop)
    signal.signal(signal.SIGINT, lambda x, y: kill_all())
    signal.signal(signal.SIGTERM, lambda x, y: kill_all())
    main()
