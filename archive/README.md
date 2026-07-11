# Quarantined tools

## punisher.py
Retrains the XGBoost booster with `xgb.train()` **without an objective**
and with reward-valued (non 0/1) labels — running it and deploying its
output risks silently corrupting the classifier the live ML gate uses.
Quarantined until rewritten with `objective="binary:logistic"` and
proper labels. Do not run as-is.
