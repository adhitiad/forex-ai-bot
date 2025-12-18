import logging
import os
import subprocess
import sys
import time

# --- KONFIGURASI ---
CHECK_INTERVAL = 30  # Cek update setiap 30 detik
MAIN_SCRIPT = "runner.py"
TRAIN_SCRIPT_V1 = "train.py"
TRAIN_SCRIPT_RL = "train_rl.py"
# -------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | UPDATER | %(message)s")


def run_git_command(command):
    """Menjalankan perintah git"""
    try:
        result = subprocess.run(
            command, cwd=os.getcwd(), shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Git Error: {e}")
        return ""


def get_python_exe():
    """Mendapatkan path python di venv"""
    if os.path.exists("venv"):
        return os.path.join("venv", "Scripts", "python.exe")
    return sys.executable


def run_training():
    """Menjalankan training V1 dan RL secara berurutan"""
    python_exe = get_python_exe()

    # 1. Training Brain V1 (Transformer)
    logging.info(f"🧠 Memulai Training Brain V1 ({TRAIN_SCRIPT_V1})...")
    try:
        subprocess.run([python_exe, TRAIN_SCRIPT_V1], check=True)
        logging.info("✅ Training V1 Selesai.")
    except subprocess.CalledProcessError:
        logging.error("❌ Training V1 Gagal! Cek log manual.")

    # 2. Training Brain RL (PPO)
    if os.path.exists(TRAIN_SCRIPT_RL):
        logging.info(f"🦾 Memulai Training Brain RL ({TRAIN_SCRIPT_RL})...")
        try:
            subprocess.run([python_exe, TRAIN_SCRIPT_RL], check=True)
            logging.info("✅ Training RL Selesai.")
        except subprocess.CalledProcessError:
            logging.error("❌ Training RL Gagal!")
    else:
        logging.warning(
            f"⚠️ File {TRAIN_SCRIPT_RL} tidak ditemukan, melewati training RL."
        )


def start_bot():
    """Menyalakan bot utama"""
    logging.info(f"🚀 Memulai {MAIN_SCRIPT}...")
    python_exe = get_python_exe()
    return subprocess.Popen([python_exe, MAIN_SCRIPT])


def stop_bot(process):
    """Mematikan bot utama"""
    if process is None:
        return
    logging.info("🛑 Menghentikan bot untuk maintenance...")
    try:
        process.terminate()
        process.wait(timeout=10)
    except:
        process.kill()
    logging.info("✅ Bot berhenti.")


def check_for_updates():
    """Mengecek update di GitHub"""
    logging.info("🔍 Mengecek update di GitHub...")
    run_git_command("git fetch")
    status = run_git_command("git status -uno")

    if "Your branch is behind" in status:
        logging.info("📦 Update TERDETEKSI! Sedang mendownload...")
        pull_output = run_git_command("git pull")
        logging.info(f"Git Pull: {pull_output}")
        return True
    return False


def main():
    if not os.path.exists(".git"):
        logging.error("❌ Bukan folder Git repository.")
        return

    # --- START AWAL ---
    # Opsional: Uncomment baris di bawah jika ingin training dulu saat script updater baru dinyalakan
    # run_training()

    bot_process = start_bot()

    try:
        while True:
            time.sleep(CHECK_INTERVAL)

            if check_for_updates():
                # 1. Matikan Bot
                stop_bot(bot_process)

                # 2. Lakukan Training Ulang (WAJIB)
                run_training()

                # 3. Nyalakan Bot Lagi
                logging.info("♻️ Restarting Bot dengan otak baru...")
                bot_process = start_bot()

    except KeyboardInterrupt:
        stop_bot(bot_process)
        logging.info("👋 Updater Dimatikan.")


if __name__ == "__main__":
    main()
