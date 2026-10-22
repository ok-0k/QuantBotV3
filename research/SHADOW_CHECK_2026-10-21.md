# Evidence-gate shadow check (2026-10-21)

A read-only check run by the PM on 2026-10-21. Nothing in the bot was changed.

## Question

Do the signals the evidence gate refused since 2026-10-14 08:00 UTC predict
falling prices?

## Spec (fixed before looking)

- **Measure:** gross short forward returns at 5, 15, 30, 60, 120 and 240
  minutes from each refused entry's journaled price.
- **Prices:** Binance public 1m klines, through the fetch and forward-return
  functions of `scripts/entry_edge.py` (cache
  `/home/admin/trading_data/research/klines`).
- **Cost bar:** a round trip costs 0.36%.
- **Baseline:** 20,000 random (symbol, minute) draws from the same symbols and
  the same window.
- **Uncertainty:** 95% confidence intervals over day means (clustered by UTC
  day).
- **Data access:** `trading.db` opened with `mode=ro`.
- **Script:** `research/shadow_edge.py`, a copy of
  `/home/admin/.audit-2026-10/shadow_edge_20261021.py`, unchanged except a
  header comment.

## Data

14,652 refused signals (reason `no_validated_edge`) over 46 symbols, from
2026-10-14 08:00 UTC to 2026-10-21. Only signals whose 240-minute horizon had
already passed are included.

## Results

Gross short return in %, mean ± 95% CI, before costs:

| group | +5m | +15m | +30m | +60m | +120m | +240m |
|---|---|---|---|---|---|---|
| refused | −0.008 ± 0.009 | −0.005 ± 0.022 | +0.000 ± 0.053 | +0.007 ± 0.083 | +0.039 ± 0.138 | +0.102 ± 0.391 |
| random | +0.005 ± 0.008 | +0.012 ± 0.022 | +0.015 ± 0.036 | +0.026 ± 0.073 | +0.035 ± 0.142 | +0.041 ± 0.347 |

- ML tiers 80, 85, 90 and 95 are all within these confidence intervals.
- Share of signals whose favourable move reached +0.36% within 30 minutes:
  refused 30.1%, random 46.0%.

## Conclusion

The refused signals are indistinguishable from random entries, and far below
the 0.36% round-trip cost. The gate is not blocking an edge. **No action.**

One week of signals is weak evidence. Even a positive result could only have
led to a pre-registered forward test, never to a change of the gate.
