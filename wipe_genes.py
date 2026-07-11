from db import load_brain_key, save_brain_key

state = load_brain_key('brain_state')
if state:
    # Wipe the active strategies and the graveyard
    state['strategies'] = []
    state['graveyard'] = []
    save_brain_key('brain_state', state)
    print("✅ Gene pool wiped! Old garbage deleted.")
else:
    print("⚠️ No brain state found.")
