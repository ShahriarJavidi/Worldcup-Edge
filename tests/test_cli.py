from pathlib import Path

import pytest

from wm import __version__
from wm.cli import main

EXAMPLE = Path(__file__).parent.parent / "examples" / "canada_bosnia_odds.csv"


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_arb_scan_example(capsys):
    assert main(["arb", "scan", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "Global cross-market arbitrage" in out
    assert "No arbitrage" in out


def test_arb_dutch(capsys):
    assert main(["arb", "dutch", "--book", "1.90", "3.65", "4.20", "--book", "2.10", "3.40", "4.60"]) == 0
    assert "3 arbitrage(s)" in capsys.readouterr().out


def test_errors_are_reported_not_raised(capsys):
    assert main(["arb", "dutch", "--book", "0.9", "3", "3"]) == 1
    assert "error:" in capsys.readouterr().err


def test_missing_file(capsys):
    assert main(["arb", "scan", "does-not-exist.csv"]) == 1
