#!/bin/bash
# Launch the bot with the project venv. systemd normally runs bot.py directly
# (see the drop-in /etc/systemd/system/quant-bot.service.d/10-paths.conf); this
# wrapper is kept because the base unit's ExecStart still points here.
cd "$(dirname "$0")" && exec ./.venv/bin/python bot.py "$@"
