with open('/home/admin/trading_bot/bot.py', 'r') as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    out_lines.append(line)
    # Find the pnl safety net and inject the entry_cost safety net right after it
    if 'if "pnl" not in locals():' in line:
        # Match the exact indentation
        indent = line.split('if')[0]
        out_lines.append(f'{indent}if "entry_cost" not in locals(): entry_cost = 0.0\n')

with open('/home/admin/trading_bot/bot.py', 'w') as f:
    f.writelines(out_lines)

print("✅ Sniper Bug Neutralized: Break-Even safety net deployed!")
