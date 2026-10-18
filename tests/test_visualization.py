"""Phase 6 test — every report figure renders from the frozen artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from visualization import plots


def test_registry_has_contiguous_unique_figures():
    n = len(plots.FIGURES)
    assert sorted(plots.FIGURES) == list(range(1, n + 1)), (
        "figure numbers must be contiguous 1..N"
    )
    names = [plots.FIGURES[i][0] for i in sorted(plots.FIGURES)]
    assert len(set(names)) == n, "figure file names must be unique"
    assert all(x[:2].isdigit() for x in names), "figures are numbered 01..NN"


@pytest.mark.parametrize("idx", sorted(plots.FIGURES))
def test_each_figure_renders(idx, tmp_path):
    name, fn = plots.FIGURES[idx]
    fig = fn()
    assert fig.axes, f"figure {name} has no axes"
    path = tmp_path / f"{name}.png"
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    assert path.exists()
    assert path.stat().st_size > 10_000, f"{name} looks empty"


def test_render_all(tmp_path):
    written = plots.render(out_dir=tmp_path)
    assert len(written) == len(plots.FIGURES)
    for p in written:
        assert p.exists() and p.stat().st_size > 10_000
    assert sorted(x.name for x in tmp_path.glob("*.png")) == sorted(
        plots.FIGURES[i][0] + ".png" for i in sorted(plots.FIGURES)
    )


def test_source_artifacts_exist():
    """Every figure has a data source that Phase 4/5 must have produced."""
    from config import RESULTS_DIR, TRAIN_SEEDS

    eval_dir = RESULTS_DIR / "evaluation"
    for f in ("per_run.csv", "cell_means.csv", "summary_by_strategy.csv",
              "significance_tests.csv", "stats_tests.md",
              "welch_vs_best_baseline.csv"):
        assert (eval_dir / f).exists(), f"missing {f}"
    assert (RESULTS_DIR / "reward_sensitivity.csv").exists()
    assert (RESULTS_DIR / "sensitivity_summary.csv").exists()   # fig 13
    assert (RESULTS_DIR / "ablation.csv").exists()
    assert (RESULTS_DIR / "overestimation.csv").exists()
    for seed in TRAIN_SEEDS:
        for name in ("q_learning", "sarsa", "dqn", "double_dqn"):
            assert (RESULTS_DIR / "training" / f"{name}_seed{seed}.csv").exists()
