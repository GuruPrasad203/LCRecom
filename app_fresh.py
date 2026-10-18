"""Fresh Flask frontend for Adaptive Coding-Practice Recommendation project.

A dashboard-style interface — not a Streamlit clone. Unique UI with modern
design, collapsible panels, and interactive data exploration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from flask import Flask, render_template, request, send_from_directory, jsonify

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    N_ACTIONS,
    N_TOPICS,
    READOUT_EVAL_SEEDS,
    READOUT_N_STUDENTS,
    RESULTS_DIR,
    TOPICS,
    TRAIN_SEEDS,
)  # noqa: E402

from evaluation.compare_models import best_strategy, cell_means, load_runs
from evaluation.strategies import STRATEGY_LABELS, STRATEGY_ORDER, STRATEGY_TYPE
from visualization.plots import FIGURES, render

app = Flask(__name__)
app.secret_key = "adaptive-coding-dashboard"


# --------------------------------------------------------------------------
# Helpers / data loading
# --------------------------------------------------------------------------

def load_results():
    """Load all result CSVs."""
    eval_dir = RESULTS_DIR / "evaluation"
    train_dir = RESULTS_DIR / "training"

    data = {}

    data["summary"] = pd.read_csv(eval_dir / "summary_by_strategy.csv")
    data["cell_means"] = pd.read_csv(eval_dir / "cell_means.csv")
    data["per_run"] = pd.read_csv(eval_dir / "per_run.csv")

    se_path = eval_dir / "reward_sensitivity.csv"
    data["reward_sensitivity"] = pd.read_csv(se_path) if se_path.exists() else pd.DataFrame()

    # Training curves
    training_curves = {}
    for name in ["dqn", "double_dqn", "sarsa", "q_learning"]:
        runs = []
        for seed in TRAIN_SEEDS:
            p = train_dir / f"{name}_seed{seed}.csv"
            if p.exists():
                s = pd.read_csv(p)["reward"]
                runs.append(s)
        if runs:
            training_curves[name] = pd.concat(runs, axis=1)
    data["training_curves"] = training_curves

    return data


@app.context_processor
def inject_globals():
    """Make data and strategy globals available in all templates."""
    from visualization.plots import STRATEGY_COLORS
    data = load_results()
    return {
        "data": data,
        "STRATEGY_LABELS": STRATEGY_LABELS,
        "STRATEGY_ORDER": STRATEGY_ORDER,
        "STRATEGY_TYPE": STRATEGY_TYPE,
        "STRATEGY_COLORS": STRATEGY_COLORS,
        "zip": zip,
        "enumerate": enumerate,
        "range": range,
        "hasattr": hasattr,
        "nav_items": NAV_ITEMS,
        "NAV_ROUTES": NAV_ROUTES,
    }


# --------------------------------------------------------------------------
# API endpoints for data exploration
# --------------------------------------------------------------------------

@app.route("/api/strategies")
def api_strategies():
    """Return strategy info as JSON."""
    return jsonify({
        "strategies": STRATEGY_ORDER,
        "labels": STRATEGY_LABELS,
        "types": STRATEGY_TYPE,
    })


@app.route("/api/summary")
def api_summary():
    """Return summary data as JSON."""
    df = load_results()["summary"]
    return jsonify({
        "rows": [
            {
                "strategy": row["strategy"],
                "type": row["type"],
                "skill_gain_mean": float(row["skill_gain_mean"]),
                "total_reward_mean": float(row["total_reward_mean"]),
                "completion_rate_mean": float(row["completion_rate_mean"]),
                "calibration_mean": float(row["calibration_mean"]),
                "frustration_end_mean": float(row["frustration_end_mean"]),
            }
            for _, row in df.iterrows()
        ]
    })


@app.route("/api/cell_means")
def api_cell_means():
    """Return cell means data as JSON."""
    df = load_results()["cell_means"]
    return jsonify({
        "rows": [
            {
                "strategy": row["strategy"],
                "student_id": row["student_id"],
                "skill_gain": float(row["skill_gain"]),
                "total_reward": float(row["total_reward"]),
                "completion_rate": float(row["completion_rate"]),
                "calibration": float(row["calibration"]),
                "frustration_end": float(row["frustration_end"]),
            }
            for _, row in df.iterrows()
        ]
    })


# -------------------------------------------------------------------------
# Main dashboard page
# -------------------------------------------------------------------------

NAV_ITEMS = [
    "Overview",
    "Strategy Comparison",
    "Evaluation",
    "Training Curves",
    "Figures",
]

# Nav label -> endpoint name (used with url_for in templates)
NAV_ROUTES = {
    "Overview": "overview",
    "Strategy Comparison": "comparison",
    "Evaluation": "evaluation",
    "Training Curves": "training_dashboard",
    "Figures": "figures_dashboard",
}


@app.route("/")
def dashboard():
    """Main dashboard page — entry point of the Flask app."""
    return render_template(
        "dashboard.html",
        nav_items=NAV_ITEMS,
        selected="overview",
    )


# -------------------------------------------------------------------------
# Overview page — high-level metrics summary
# -------------------------------------------------------------------------

@app.route("/overview")
def overview():
    """Overview page — key metrics at a glance."""
    data = load_results()
    summary = data["summary"]

    # Compute totals
    total_students = summary["n"].sum() if "n" in summary.columns else 0

    # Best strategy by skill gain
    best_idx = summary["skill_gain_mean"].idxmax()
    best_strategy = STRATEGY_LABELS.get(summary.loc[best_idx, "strategy"], "Unknown")

    # Aggregate metrics
    avg_reward = summary["total_reward_mean"].mean()
    avg_skill_gain = summary["skill_gain_mean"].mean()
    avg_calibration = summary["calibration_mean"].mean()

    return render_template(
        "overview.html",
        nav_items=NAV_ITEMS,
        selected="overview",
        total_students=total_students,
        best_strategy=best_strategy,
        avg_reward=f"{avg_reward:.2f}",
        avg_skill_gain=f"{avg_skill_gain:.4f}",
        avg_calibration=f"{avg_calibration:.2f}",
        strategy_count=len(summary),
    )


# -------------------------------------------------------------------------
# Strategy Comparison page — interactive comparison
# -------------------------------------------------------------------------

@app.route("/comparison")
def comparison():
    """Strategy Comparison page — detailed strategy metrics."""
    data = load_results()
    summary = data["summary"]

    return render_template(
        "comparison.html",
        nav_items=NAV_ITEMS,
        selected="comparison",
        strategies=STRATEGY_ORDER,
        data=data,
    )


# -------------------------------------------------------------------------
# Evaluation page — per-student, per-strategy details
# -------------------------------------------------------------------------

@app.route("/evaluation")
def evaluation():
    """Evaluation page — per-run metrics and filtering."""
    data = load_results()
    strategy_order = STRATEGY_ORDER

    # Default to first strategy
    strategy = request.args.get("strategy", strategy_order[0] if strategy_order else "")

    if request.method == "POST":
        strategy = request.form.get("strategy", strategy)
    else:
        strategy = request.args.get("strategy", strategy)

    with_tab = request.args.get("tab", "summary")
    unique_strategies = list(set(
        row.strategy for _, row in data["per_run"].iterrows()
    )) if hasattr(data, "per_run") else strategy_order

    return render_template(
        "evaluation.html",
        nav_items=NAV_ITEMS,
        selected="evaluation",
        data=data,
        strategy_order=strategy_order,
        strategy=strategy,
        with_tab=with_tab,
        unique_strategies=unique_strategies,
    )


# -------------------------------------------------------------------------
# Training Curves page — learning curves
# -------------------------------------------------------------------------

@app.route("/training")
def training_dashboard():
    """Training Curves page — learning curves across seeds."""
    data = load_results()
    # Pre-compute curve stats for template
    curve_stats = {}
    for name, df in data["training_curves"].items():
        if df is not None and not df.empty:
            curve_stats[name] = {
                "mean_reward": float(df.mean().mean()) if not df.mean().isna().all() else 0.0,
                "std_reward": float(df.mean().std()) if not df.mean().isna().all() else 0.0,
                "episodes": len(df.mean()),
                "seeds": df.shape[1] if df.shape else 0,
            }
    else:
        curve_stats = {}
    return render_template(
        "training_dashboard.html",
        nav_items=NAV_ITEMS,
        selected="training",
        data=data,
        curve_stats=curve_stats,
    )


# -------------------------------------------------------------------------
# Figures page — interactive figure viewer
# -------------------------------------------------------------------------

@app.route("/figures")
def figures_dashboard():
    """Figures page — list and view evaluation figures."""
    fig_dir = Path(__file__).resolve().parent / "visualization" / "figures"
    codes = sorted({p.stem[:2] for p in fig_dir.glob("*.png")})
    return render_template(
        "figures_dashboard.html",
        nav_items=NAV_ITEMS,
        selected="figures",
        codes=codes,
    )


@app.route("/figure/<code>")
def figure_dashboard(code):
    """Serve a single figure by code."""
    fig_dir = Path(__file__).resolve().parent / "visualization" / "figures"
    pattern = f"{code}_*.png"
    paths = sorted(fig_dir.glob(pattern))
    if paths:
        return send_from_directory(str(fig_dir), paths[0].name)
    return "Figure not found", 404


# -------------------------------------------------------------------------
# Static figure serving
# -------------------------------------------------------------------------

@app.route("/static/figures/<path:filename>")
def static_figures(filename):
    """Serve figures from visualization/figures/."""
    fig_dir = Path(__file__).resolve().parent / "visualization" / "figures"
    return send_from_directory(str(fig_dir), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502, debug=True)