"""Phase 6 test — all ten report figures render from the frozen artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from visualization import plots


def test_registry_has_exactly_ten_figures():
    assert sorted(plots.FIGURES) == list(range(1, 11))
    names = [plots.FIGURES[i][0] for i in sorted(plots.FIGURES)]
    assert len(set(names)) == 10, "figure file names must be unique"
    assert all(n[:2].isdigit() for n in names), "figures are numbered 01..10"


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
    assert len(written) == 10
    for p in written:
        assert p.exists() and p.stat().st_size > 10_000
    assert sorted(x.name for x in tmp_path.glob("*.png")) == sorted(
        plots.FIGURES[i][0] + ".png" for i in sorted(plots.FIGURES)
    )


def test_source_artifacts_exist():
    """Every figure has a data source that Phase 4/5 must have produced."""
    from config import RESULTS_DIR

    eval_dir = RESULTS_DIR / "evaluation"
    for f in ("per_run.csv", "cell_means.csv", "summary_by_strategy.csv",
              "significance_tests.csv"):
        assert (eval_dir / f).exists(), f"missing {f}"
    assert (RESULTS_DIR / "reward_sensitivity.csv").exists()
    for seed in (0, 1, 2):
        for name in ("q_learning", "sarsa", "dqn", "double_dqn"):
            assert (RESULTS_DIR / "training" / f"{name}_seed{seed}.csv").exists()
