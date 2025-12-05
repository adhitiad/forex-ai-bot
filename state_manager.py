import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StateManager")

STATE_FILE = "data/bot_state.json"


class StateManager:
    def __init__(self):
        self.active_position = None
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    self.active_position = json.load(f)
            except:
                self.active_position = None

    def save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.active_position, f)

    def set_active_position(self, symbol, side, price, lot, tp, sl):
        self.active_position = {
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "lot": lot,
            "tp": tp,
            "sl": sl,
        }
        self.save_state()  # PERSISTENCE FIX

    def clear_active_position(self):
        self.active_position = None
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)  # Hapus file state


state_manager = StateManager()
