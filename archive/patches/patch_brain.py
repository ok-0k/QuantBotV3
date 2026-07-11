with open("/home/admin/trading_bot/brain.py", "r") as f:
    lines = f.readlines()

save_function = """
    def save(self) -> None:
        try:
            from db import get_recent_trades
            all_trades = get_recent_trades(limit=1000)
            for s in self.strategies:
                s_trades = [t for t in all_trades if str(t.get("strategy", t.get("strat", ""))).strip() == s.name.strip()]
                if s_trades:
                    s.total_trades = len(s_trades)
                    s.total_pnl = sum(float(t.get("pnl", 0.0)) for t in s_trades)
                    s.wins = sum(1 for t in s_trades if float(t.get("pnl", 0.0)) > 0)
        except Exception as e:
            import logging
            logging.getLogger("quantbot").error(f"Stats Sync Error: {e}")

        from db import save_brain_key
        save_brain_key("brain_state", {
            "strategies":         [self._serialise(s) for s in self.strategies],
            "regime_weights":      getattr(self, "regime_weights", {}),
            "total_trades":        getattr(self, "total_trades", 0),
            "current_regime":      getattr(self, "current_regime", "neutral"),
            "regime_history":      getattr(self, "regime_history", [])[-200:],
            "mutation_log":        getattr(self, "mutation_log", [])[-50:],
            "graveyard":           getattr(self, "graveyard", [])[-100:],
            "generation_log":      getattr(self, "generation_log", [])[-50:],
            "peak_equity":         getattr(self, "peak_equity", 10000.0),
            "circuit_open":        getattr(self, "circuit_open", False),
            "circuit_tripped_at":  getattr(self, "circuit_tripped_at", 0),
            "cb_log":              getattr(self, "cb_log", [])[-20:],
        })
        
        # The Force-Spawn Panic Button
        if not self.strategies and hasattr(self, "_try_generate"):
            for _ in range(14): self._try_generate()

"""

# Inject right above _softmax so we know it's safely inside the Brain class
for i, line in enumerate(lines):
    if "def _softmax" in line:
        lines.insert(i, save_function)
        break

with open("/home/admin/trading_bot/brain.py", "w") as f:
    f.writelines(lines)

print("✅ Save function successfully re-injected!")
