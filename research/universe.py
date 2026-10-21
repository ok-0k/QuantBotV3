"""
Frozen research universe: config.SYMBOLS as of f2a051a (2026-10), the 50
symbols every study, data check and forward test so far was run on, in the
same order (order matters: e.g. smc ties go by it).

The live bot's config.SYMBOLS may change (2026-10-21: the 4 pairs delisted
from Binance spot were dropped); this list must not, or re-runs of the
pre-registered studies and the frozen carry forward test would see a
different universe. It keeps the delisted pairs' history up to delisting.
Deliberately imports nothing from config.
"""

RESEARCH_SYMBOLS: tuple[str, ...] = (
    # Large caps — highest liquidity
    "BTCUSDT",   "ETHUSDT",   "BNBUSDT",   "SOLUSDT",   "XRPUSDT",
    # Mid caps — good volatility
    "DOGEUSDT",  "ADAUSDT",   "AVAXUSDT",  "DOTUSDT",   "MATICUSDT",
    "LINKUSDT",  "LTCUSDT",   "ATOMUSDT",  "UNIUSDT",   "APTUSDT",
    "NEARUSDT",  "FILUSDT",   "INJUSDT",   "OPUSDT",    "ARBUSDT",
    "SEIUSDT",   "SUIUSDT",   "TIAUSDT",   "STXUSDT",   "RUNEUSDT",
    # DeFi
    "AAVEUSDT",  "MKRUSDT",   "SNXUSDT",   "CRVUSDT",   "COMPUSDT",
    # AI / infra
    "FETUSDT",   "RENDERUSDT","WLDUSDT",   "OCEANUSDT", "AGIXUSDT",
    # Gaming / NFT
    "SANDUSDT",  "MANAUSDT",  "AXSUSDT",   "ENJUSDT",   "GALAUSDT",
    "APEUSDT",   "GMTUSDT",   "BLURUSDT",  "ILVUSDT",   "CHZUSDT",
    # Layer 1 alts
    "HBARUSDT",  "EGLDUSDT",  "FLOWUSDT",  "ICPUSDT",   "QNTUSDT",
)
