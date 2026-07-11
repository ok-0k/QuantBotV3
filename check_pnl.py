import sqlite3

# Resolve the same DB the bot uses (honours TRADING_DATA_DIR).
from config import DB_PATH

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Calculate Total Profit (Sum of all winning trades)
    cursor.execute("SELECT SUM(net_pnl) FROM trades WHERE net_pnl > 0")
    total_profit = cursor.fetchone()[0] or 0.0

    # Calculate Total Loss (Sum of all losing trades)
    cursor.execute("SELECT SUM(net_pnl) FROM trades WHERE net_pnl < 0")
    total_loss = cursor.fetchone()[0] or 0.0

    # Calculate Total Fees Paid
    cursor.execute("SELECT SUM(fee_total) FROM trades")
    total_fees = cursor.fetchone()[0] or 0.0

    print("=============================")
    print(" 📊 RECENT DATABASE PNL REPORT")
    print("=============================")
    print(f"Total Amount Made : +${total_profit:.2f}")
    print(f"Total Amount Lost : -${abs(total_loss):.2f}")
    print(f"Total Fees Paid   : -${abs(total_fees):.2f}")
    print("-----------------------------")
    print(f"Net Result        : ${total_profit + total_loss:.2f}")
    print("=============================")

except Exception as e:
    print(f"Error reading database: {e}")
