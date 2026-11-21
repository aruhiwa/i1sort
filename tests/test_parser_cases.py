"""Run the hand-checked YAML cases through the real parser (needs the dictionary data)."""

from pathlib import Path

import pytest

from i1sort.config import load_config
from i1sort.evaluate import check_case, load_cases

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def parser_and_filter():
    cfg = load_config()
    if not cfg.resolve(cfg["dictionaries"]["jmdict"]).exists():
        pytest.skip("JMdict not available")
    from i1sort.filters import TokenFilter
    from i1sort.parse.pipeline import Parser

    p = Parser(cfg, use_cache=False)
    return p, TokenFilter(cfg, p.db, ignore=())


CASES = load_cases(HERE / "parser_cases.yaml")


@pytest.mark.parametrize("case", CASES, ids=[str(c.get("text") or c.get("same"))[:30] for c in CASES])
def test_parser_case(case, parser_and_filter):
    parser, tf = parser_and_filter
    r = check_case(case, parser, tf)
    assert r.ok, "; ".join(r.problems) + f" | got: {r.got}"


def test_heldout_accuracy_floor(parser_and_filter):
    """The held-out set is not tuned against; guard against regressions only."""
    parser, tf = parser_and_filter
    results = [check_case(c, parser, tf) for c in load_cases(HERE / "heldout_cases.yaml")]
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    f1 = 2 * tp / (2 * tp + fp + fn)
    assert f1 >= 0.90, f"held-out word F1 dropped to {f1:.3f}"
