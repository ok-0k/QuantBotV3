"""The test harness never sees exchange credentials and never touches live data."""

import os

import config


def test_credentials_scrubbed_and_data_sandboxed():
    leaked = [k for k in os.environ if k.startswith("BINANCE_") or k.endswith(("_API_KEY", "_API_SECRET", "_SECRET"))]
    assert leaked == []
    assert not str(config.DB_PATH).startswith("/home/admin/trading_data")
