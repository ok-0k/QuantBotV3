with open('/home/admin/trading_bot/brain.py', 'r') as f:
    content = f.read()

# Close the Immortality Gap
old_line = 'elif strat.win_rate < 0.30 or strat.total_pnl < TRIAL_MIN_PNL * 2:'
new_line = 'elif strat.win_rate < TRIAL_MIN_WIN_RATE or strat.total_pnl < TRIAL_MIN_PNL:'
content = content.replace(old_line, new_line)

with open('/home/admin/trading_bot/brain.py', 'w') as f:
    f.write(content)

print("✅ Executioner Fixed: No more immortality gap.")
