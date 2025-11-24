import pytest

from examples import locomo_eval


def test_locomo_baseline_precision_recall():
    metrics = locomo_eval.run_locomo_baseline(top_k=5)

    assert metrics["questions"] == 1986
    assert metrics["top_k"] == 5
    assert metrics["macro_precision"] == pytest.approx(0.06586102719033234, rel=1e-6)
    assert metrics["macro_recall"] == pytest.approx(0.2976704824869922, rel=1e-6)
    assert metrics["macro_f1"] == pytest.approx(0.10569441747689483, rel=1e-6)
    assert metrics["hit_rate"] == pytest.approx(0.32376636455186303, rel=1e-6)
    assert metrics["exact_match_rate"] == pytest.approx(0.2809667673716012, rel=1e-6)
    assert metrics["mrr"] == pytest.approx(0.2282477341389728, rel=1e-6)
    assert metrics["micro_precision"] == pytest.approx(0.06586102719033232, rel=1e-6)
    assert metrics["micro_recall"] == pytest.approx(0.232409381663113, rel=1e-6)
    assert metrics["micro_f1"] == pytest.approx(0.10263653483992466, rel=1e-6)


def test_locomo_category_filter():
    metrics = locomo_eval.run_locomo_baseline(top_k=5, categories={1, 2})

    assert metrics["questions"] == 603
    assert metrics["macro_precision"] == pytest.approx(0.057711442786069655, rel=1e-6)
    assert metrics["macro_recall"] == pytest.approx(0.21190948815326927, rel=1e-6)
    assert metrics["macro_f1"] == pytest.approx(0.08603929698457062, rel=1e-6)
    assert metrics["hit_rate"] == pytest.approx(0.2769485903814262, rel=1e-6)
    assert metrics["exact_match_rate"] == pytest.approx(0.17412935323383086, rel=1e-6)
    assert metrics["mrr"] == pytest.approx(0.17330016583747926, rel=1e-6)
    assert metrics["micro_precision"] == pytest.approx(0.05771144278606965, rel=1e-6)
    assert metrics["micro_recall"] == pytest.approx(0.1383147853736089, rel=1e-6)
    assert metrics["micro_f1"] == pytest.approx(0.08144161010999296, rel=1e-6)
