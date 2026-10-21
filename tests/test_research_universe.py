"""
The research universe is frozen (research.universe.RESEARCH_SYMBOLS): no
research module or script reads the live bot's config.SYMBOLS, so dropping
symbols from the live bot cannot change a study, data check or forward test.
"""

import ast
from pathlib import Path

from research.universe import RESEARCH_SYMBOLS

REPO = Path(__file__).resolve().parent.parent

# config.SYMBOLS at f2a051a, in order: the universe of every run so far.
F2A051A_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "APTUSDT",
    "NEARUSDT", "FILUSDT", "INJUSDT", "OPUSDT", "ARBUSDT",
    "SEIUSDT", "SUIUSDT", "TIAUSDT", "STXUSDT", "RUNEUSDT",
    "AAVEUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT", "COMPUSDT",
    "FETUSDT", "RENDERUSDT", "WLDUSDT", "OCEANUSDT", "AGIXUSDT",
    "SANDUSDT", "MANAUSDT", "AXSUSDT", "ENJUSDT", "GALAUSDT",
    "APEUSDT", "GMTUSDT", "BLURUSDT", "ILVUSDT", "CHZUSDT",
    "HBARUSDT", "EGLDUSDT", "FLOWUSDT", "ICPUSDT", "QNTUSDT",
]


def test_research_symbols_equal_the_f2a051a_universe():
    assert isinstance(RESEARCH_SYMBOLS, tuple)
    assert list(RESEARCH_SYMBOLS) == F2A051A_SYMBOLS


def _reads_config_symbols(path: Path) -> list[str]:
    hits = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.ImportFrom) and node.module == "config":
            hits += [f"{path.name}:{node.lineno} from config import {a.name}"
                     for a in node.names if a.name in ("SYMBOLS", "*")]
        if (isinstance(node, ast.Attribute) and node.attr == "SYMBOLS"
                and isinstance(node.value, ast.Name) and node.value.id in ("config", "cfg")):
            hits.append(f"{path.name}:{node.lineno} {node.value.id}.SYMBOLS")
    return hits


def test_no_research_module_or_script_reads_config_symbols():
    files = sorted((REPO / "research").glob("*.py")) + sorted((REPO / "scripts").glob("*.py"))
    assert len(files) > 10
    assert [h for f in files for h in _reads_config_symbols(f)] == []


def test_study_code_paths_see_the_frozen_universe():
    from research import carry_basis, carry_forward, crowding_study, data_check, study
    for mod in (study, carry_forward, carry_basis, data_check, crowding_study):
        assert mod.SYMBOLS is RESEARCH_SYMBOLS, mod.__name__
        assert list(mod.SYMBOLS) == F2A051A_SYMBOLS, mod.__name__


def test_smc_runners_see_the_frozen_universe():
    from research import smc_study
    assert smc_study.universe() == F2A051A_SYMBOLS
    # smc_data_check binds the universe inside main(); it must come from research.universe
    tree = ast.parse((REPO / "research" / "smc_data_check.py").read_text())
    names = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("research.universe", "RESEARCH_SYMBOLS") in names
