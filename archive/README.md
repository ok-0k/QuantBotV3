# Quarantined tools

## punisher.py
Retrains the XGBoost booster with `xgb.train()` **without an objective**
and with reward-valued (non 0/1) labels — running it and deploying its
output risks silently corrupting the classifier the live ML gate uses.
Quarantined until rewritten with `objective="binary:logistic"` and
proper labels. Do not run as-is.

## strategies.py / strategy_engine_gen2.py / strategy_manager_gen2.py
Wave 4 (W10): moved here from the repo root, where they sat unimported by
any live entry point (`bot.py` is short-only via `brain.py`; nothing
imports these three). Not dangerous like `punisher.py` above — just dead
code that increased audit/maintenance surface and risked someone assuming
it was live. If ever revived, re-audit first: the prior audit found a
hardcoded-dollar (not equity-relative) kill-switch in
`strategy_engine_gen2.py` and a 5-trade live-promotion gate that would be
too permissive to trust with real capital as-is.
