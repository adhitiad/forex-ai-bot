class StateManager:
    def __init__(self):
        self.active_position = None

    def set_active_position(self, symbol, side, price, lot, tp, sl):
        self.active_position = {
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "lot": lot,
            "tp": tp,
            "sl": sl,
        }

    def get_active_position(self):
        return self.active_position

    def clear_active_position(self):
        self.active_position = None


state_manager = StateManager()
