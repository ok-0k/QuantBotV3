from db import get_recent_trades
from collections import defaultdict

trades = get_recent_trades(limit=5000)
stats = defaultdict(lambda: {"total": 0, "wins": 0, "pnl": 0.0})

for t in trades:
    strat = str(t.get('strategy', t.get('strat', ''))).strip()
    pnl = t.get('pnl')
    if pnl is not None and strat and strat not in ['-', 'None', 'XGBOOST']:
        stats[strat]['total'] += 1
        stats[strat]['pnl'] += float(pnl)
        if float(pnl) > 0:
            stats[strat]['wins'] += 1

print("\n🧬 GEN-0 BATTLEFIELD STATS 🧬")
print(f"{'STRATEGY':<22} | {'TRADES':<6} | {'WINS':<4} | {'WIN %':<6} | {'NET PNL'}")
print("-" * 58)

sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)

if not sorted_stats:
    print("No Gen-0 trades have closed yet. Let them cook!")
else:
    for strat, data in sorted_stats:
        total = data['total']
        wins = data['wins']
        pnl = data['pnl']
        win_rate = (wins / total) * 100
        print(f"{strat:<22} | {total:<6} | {wins:<4} | {win_rate:>5.1f}% | ${pnl:>.2f}")
print("\n")
