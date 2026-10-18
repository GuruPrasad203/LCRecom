"""Flask frontend for Adaptive Coding-Practice Recommendation project.

Replaces streamlit_app.py — same functionality, Flask routes + Jinja2 templates.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from flask import Flask, redirect, render_template, request, send_from_directory, jsonify, url_for

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

from evaluation.compare_models import cell_means, load_runs
from evaluation.strategies import STRATEGY_LABELS, STRATEGY_ORDER, STRATEGY_TYPE
from visualization.plots import FIGURES, render
from leetcode_integration import build_student_from_leetcode
from leetcode_client import LeetCodeAPIError

app = Flask(__name__)
app.secret_key = "adaptive-coding-rl-secret"


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
    """Make loaded data and strategy globals available in all templates."""
    data = load_results()
    from visualization.plots import STRATEGY_COLORS
    return {
        "data": data,
        "STRATEGY_LABELS": STRATEGY_LABELS,
        "STRATEGY_ORDER": STRATEGY_ORDER,
        "STRATEGY_TYPE": STRATEGY_TYPE,
        "STRATEGY_COLORS": STRATEGY_COLORS,
        "zip": zip,
        "enumerate": enumerate,
        "range": range,
        "nav_items": NAV_ITEMS,
        "NAV_ROUTES": NAV_ROUTES,
        "selected": SELECTED_BY_ENDPOINT.get(request.endpoint, ""),
    }


# --------------------------------------------------------------------------
# Routes for figures
# --------------------------------------------------------------------------

@app.route("/figure/<code>")
def figure(code):
    """Serve a single figure image by code (e.g., 01, 02, ..., 10)."""
    fig_dir = Path(__file__).resolve().parent / "visualization" / "figures"
    pattern = f"{code}_*.png"
    paths = sorted(fig_dir.glob(pattern))
    if paths:
        return send_from_directory(str(fig_dir), paths[0].name)
    return "Figure not found", 404


@app.route("/figures")
def all_figures():
    """List all available figure codes."""
    fig_dir = Path(__file__).resolve().parent / "visualization" / "figures"
    codes = sorted({p.stem[:2] for p in fig_dir.glob("*.png")})
    return jsonify({"codes": codes})


# --------------------------------------------------------------------------
# Main page — navigation
# --------------------------------------------------------------------------

NAV_ITEMS = [
    "Report Figures",
    "Evaluation Metrics",
    "Demo / Trace",
    "Training Curves",
    "LeetCode Integration",
]

# Nav label -> endpoint name (used with url_for in templates)
NAV_ROUTES = {
    "Report Figures": "report",
    "Evaluation Metrics": "metrics",
    "Demo / Trace": "demo",
    "Training Curves": "training",
    "LeetCode Integration": "leetcode",
}

SELECTED_BY_ENDPOINT = {
    "index": "",
    "report": "Report Figures",
    "metrics": "Evaluation Metrics",
    "demo": "Demo / Trace",
    "training": "Training Curves",
    "leetcode": "LeetCode Integration",
    "results": "",
    "figure": "Report Figures",
    "all_figures": "Report Figures",
    "static_figures": "Report Figures",
}


@app.route("/")
def index():
    """Home page with navigation menu."""
    page = request.args.get("page")
    if page in NAV_ROUTES:
        return redirect(url_for(NAV_ROUTES[page]))
    data = load_results()
    return render_template("index.html", data=data)


# --------------------------------------------------------------------------
# Report Figures page
# --------------------------------------------------------------------------

@app.route("/report")
def report():
    """Report Figures page — displays the 10 figures."""
    data = load_results()
    fig_names = [
        ("01", "Learning Curves"),
        ("02", "Skill Gain Bars"),
        ("03", "Reward Bars"),
        ("04", "Metrics Panel 2x2"),
        ("05", "Skill Gain Boxplots"),
        ("06", "Reward Sensitivity"),
        ("07", "DQN vs Q-Learning Heatmaps"),
        ("08", "Calibration vs Frustration"),
        ("09", "Significance Heatmap"),
        ("10", "Per-Topic Mastery Gain"),
    ]
    return render_template(
        "report.html",
        fig_names=fig_names,
        data=data,
    )


# --------------------------------------------------------------------------
# Evaluation Metrics page
# --------------------------------------------------------------------------

@app.route("/metrics", methods=["GET", "POST"])
def metrics():
    """Evaluation Metrics page — Summary, Cell Means, Per-Run tabs."""
    data = load_results()
    strategy_order = STRATEGY_ORDER

    if request.method == "POST":
        strategy = request.form.get("strategy", strategy_order[0])
    else:
        strategy = request.args.get("strategy", strategy_order[0])

    with_tab = request.args.get("tab", "summary")

    return render_template(
        "metrics.html",
        data=data,
        strategy_order=strategy_order,
        strategy=strategy,
        with_tab=with_tab,
    )


# --------------------------------------------------------------------------
# Demo / Trace page — simplified: shows form, no full trace execution
# --------------------------------------------------------------------------

@app.route("/demo", methods=["GET", "POST"])
def demo():
    """Demo / Trace page — form to select student and strategy.

    Full per-step trace execution is complex in Flask (requires environment
    reset, step-by-step rendering). The form pre-selects and validates choices;
    trace generation would need a dedicated backend endpoint.
    """
    data = load_results()

    # Default values
    strategy_name = request.form.get(
        "strategy",
        [n for n in STRATEGY_ORDER if STRATEGY_TYPE[n] != "Baseline"][0],
    )
    student_id = request.form.get(
        "student",
        sorted(data["per_run"]["student_id"].unique())[0]
        if "student_id" in data["per_run"].columns
        else "",
    )
    eval_seed = request.form.get("eval_seed", "100")

    if request.method == "POST" and strategy_name and student_id:
        # Form was submitted — validate and re-render with the selection
        # For now just re-show the form with the selected values;
        # a full trace would require aJAX backend or separate route.
        strategy_options = [n for n in STRATEGY_ORDER if STRATEGY_TYPE[n] != "Baseline"]
        student_options = (
            sorted(data["per_run"]["student_id"].unique())
            if "student_id" in data["per_run"].columns
            else []
        )

        return render_template(
            "demo.html",
            data=data,
            strategy_options=strategy_options,
            student_options=student_options,
            eval_seeds=[100, 200, 300],
            selected_strategy=strategy_name,
            selected_student=student_id,
            selected_seed=eval_seed,
            form_submitted=True,
        )

    # GET request — show the form
    strategy_options = [n for n in STRATEGY_ORDER if STRATEGY_TYPE[n] != "Baseline"]
    student_options = (
        sorted(data["per_run"]["student_id"].unique())
        if "student_id" in data["per_run"].columns
        else []
    )

    return render_template(
        "demo.html",
        data=data,
        strategy_options=strategy_options,
        student_options=student_options,
        eval_seeds=[100, 200, 300],
        selected_strategy=strategy_name,
        selected_student=student_id,
        selected_seed=eval_seed,
    )


# --------------------------------------------------------------------------
# Training Curves page
# --------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb color to an rgba() string."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(108, 117, 125, {alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def build_training_charts(data: dict, strategy: str = "") -> list[dict]:
    """Prepare Chart.js payloads for the training curves page.

    Each entry carries a 100-episode moving-average mean across seeds plus a
    +/-1 SD band, downsampled so the JSON payload stays small.
    """
    from visualization.plots import STRATEGY_COLORS

    window = 100
    max_points = 500
    charts = []
    for name, df in data["training_curves"].items():
        if strategy and name != strategy:
            continue
        n = len(df)
        if not n:
            continue
        step = max(1, -(-n // max_points))
        mean = df.mean(axis=1).rolling(window, min_periods=1).mean()
        sd = df.std(axis=1).fillna(0.0).rolling(window, min_periods=1).mean()
        label = STRATEGY_LABELS.get(name, name)
        color = STRATEGY_COLORS.get(label, "#6c757d")
        charts.append({
            "id": name,
            "label": label,
            "color": color,
            "band": _hex_to_rgba(color, 0.18),
            "n_seeds": int(df.shape[1]),
            "n_episodes": int(n),
            "episodes": [int(i) for i in df.index[::step]],
            "mean": [round(float(v), 3) for v in mean.iloc[::step]],
            "lower": [round(float(v), 3) for v in (mean - sd).iloc[::step]],
            "upper": [round(float(v), 3) for v in (mean + sd).iloc[::step]],
        })
    return charts


@app.route("/training")
def training():
    """Training Curves page — learning curves for each strategy."""
    data = load_results()
    strategy = request.args.get("strategy", "").strip()
    charts = build_training_charts(data, strategy=strategy)
    return render_template(
        "training.html",
        data=data,
        strategy=strategy,
        charts=charts,
    )


# --------------------------------------------------------------------------
# Evaluation Results page
# --------------------------------------------------------------------------

@app.route("/results", methods=["GET", "POST"])
def results():
    """Evaluation Results page — structured display of all key metrics.

    Shows the main comparison table, research questions, and summary statistics
    from the locked evaluation results (450 per-run rows, 30 cells per strategy).
    """
    import pandas as pd

    # Load data
    per_run = pd.read_csv("results/evaluation/per_run.csv")
    summary = per_run.groupby("strategy").mean(numeric_only=True)

    # Research questions data from REPORT.md §4.3
    rqs = [
        {
            "rq": "RQ1 — Does RL beat difficulty-based recommendation?",
            "question": "No. Q-Learning vs Difficulty-Based: Δ skill gain −0.0006, p = 0.208 / 0.903 (not significant). RL worse than best baseline (Prerequisite-Based).",
            "answer": "RL matches the heuristic difficulty baseline and beats Random, but the prerequisite ordering is the strongest method on skill gain.",
        },
        {
            "rq": "RQ2 — Does Q-Learning outperform SARSA?",
            "question": "No. SARSA has the higher skill gain (0.0145 vs 0.0139, Δ −0.0006, p = 0.031 / 0.043, significant in SARSA's favour). Reward and calibration indistinguishable.",
            "answer": "SARSA wins on skill gain; Q-Learning wins on reward (15.19 vs 15.29 — essentially tied).",
        },
        {
            "rq": "RQ3 — Does DQN beat tabular Q-Learning on the 26-dim continuous state?",
            "question": "No. Skill gain Δ −0.0004 (n.s.), calibration Δ +0.003 (n.s.), but total reward far lower: 2.37 vs 15.19 (p = 2.85e-08). 37/90 DQN runs end with negative total reward.",
            "answer": "Double DQN recovers the reward (12.06) but not the skill gain (0.0135). Tabular agents strictly positive on same cells.",
        },
        {
            "rq": "RQ4 — Does Double DQN help with calibration and frustration?",
            "question": "Calibration: DQN 0.887 vs Prerequisite 0.963 (worse, Δ −0.076, p = 1.50e-04). Frustration: DQN better than Random (p = 0.0187). Double DQN vs DQN: identical skill gain (Δ +0.0000) but reward significantly higher (12.06 vs 2.37, p = 8.57e-08).",
            "answer": "Double DQN fixes the reward collapse but not the skill-gain ranking. DQN is unstable with high variance.",
        },
    ]

    # Main evaluation table data
    strategy_order = ["Prerequisite-Based", "Difficulty-Based", "Random",
                      "SARSA", "Q-Learning", "Double DQN", "DQN"]

    # Build row data from summary
    rows = []
    for s in strategy_order:
        if s in summary.index:
            row = summary.loc[s]
            # Determine type from strategy_order position
            is_baseline = s in ["Prerequisite-Based", "Difficulty-Based", "Random"]
            # Compute std from per_run for this strategy
            strat_data = per_run[per_run["strategy"] == s]
            std_sg = strat_data["skill_gain"].std() if len(strat_data) > 1 else 0.0
            rows.append({
                "strategy": s,
                "type": "Baseline" if is_baseline else "RL",
                "skill_gain": f"{row['skill_gain']:.4f} ± {std_sg:.4f}",
                "completion": f"{row['completion_rate']:.3f}",
                "hints": f"{row['avg_hints']:.3f}",
                "time": f"{row['mean_solve_time']:.1f}",
                "calibration": f"{row['calibration']:.3f}",
                "frustration": f"{row['frustration_end']:.3f}",
                "reward": f"{row['total_reward']:.2f}",
            })

    if request.method == "POST":
        # Just re-render with same data; no form actions needed for this page
        pass

    return render_template(
        "results.html",
        rows=rows,
        rqs=rqs,
        strategy_order=strategy_order,
    )


# --------------------------------------------------------------------------
# LeetCode Integration page
# --------------------------------------------------------------------------

def _default_form_cookie() -> str:
    """Cookie pasted into the form, else the LEETCODE_SESSION env var."""
    cookie = request.form.get("session_cookie", "").strip()
    return cookie or (os.environ.get("LEETCODE_SESSION") or "").strip()


def run_leetcode_trace(
    username: str,
    strategy: str,
    eval_seed,
    session_cookie: str | None = None,
    model_seed: int = 0,
) -> tuple[object, object, list[dict], dict]:
    """Run one demo episode for a LeetCode-imported student.

    Returns (student_profile, mapped, steps, summary) where ``steps`` is the
    per-problem trace rendered by templates/leetcode.html.
    """
    from data.generate_data import load_problems, load_students
    from environment.coding_env import CodingPracticeEnv
    from evaluation.strategies import build_strategy
    from leetcode_client import load_problem_index, resolve_leetcode_problem

    student_profile, mapped = build_student_from_leetcode(
        username, session_cookie=session_cookie, split="train"
    )

    problems = load_problems()
    students = load_students(split="train")
    env = CodingPracticeEnv(problems=problems, students=students, mode="train")
    recommender = build_strategy(strategy, model_seed=model_seed, random_seed=model_seed)
    recommender.reset()

    state, _info = env.reset(seed=int(eval_seed), options={"student": student_profile})
    mastery_start = env.mastery.copy()

    # Title -> real LeetCode number/URL (cached; one HTTP call per week).
    problem_index = load_problem_index()

    steps: list[dict] = []
    total_reward = 0.0
    for i in range(env.episode_len):
        action = recommender.recommend(state)
        state, reward, terminated, truncated, si = env.step(action)
        total_reward += float(reward)
        steps.append({
            "step": i + 1,
            "topic": si["topic"],
            "difficulty_label": si["difficulty_label"],
            "outcome": "PASS" if si["success"] else "FAIL",
            "reward": float(reward),
            "mastery_after": float(si["mastery_after"]),
            "problem": resolve_leetcode_problem(
                si["problem_title"], index=problem_index
            ),
        })
        if terminated or truncated:
            break

    n_steps = max(len(env.history), 1)
    summary = {
        "strategy": strategy,
        "eval_seed": int(eval_seed),
        "total_reward": total_reward,
        "completion_rate": sum(1 for h in env.history if h["success"]) / n_steps,
        "skill_gain_total": float((env.mastery - mastery_start).sum()),
        "n_steps": len(steps),
    }
    return student_profile, mapped, steps, summary


@app.route("/leetcode", methods=["GET", "POST"])
def leetcode():
    """LeetCode Integration page — fetch user profile and initialize student.

    Accepts a LeetCode username and uses the GraphQL API to map the user's
    solved counts, acceptance rate, and recent submissions to LCRecom student
    proficiency metrics. The resulting student can then be used with any
    strategy in the demo or evaluation pages.
    """
    form_submitted = False
    selected_strategy = None
    selected_student = None
    eval_seed = None
    student_profile = None
    recommended_action = None
    steps_data = None
    trace_summary = None
    error = None

    strategy_options = [n for n in STRATEGY_ORDER if STRATEGY_TYPE[n] != "Baseline"]
    lc_username = ""

    if request.method == "POST":
        fetch_username = request.form.get("username", "").strip()
        trace_username = request.form.get("leetcode_username", "").strip()
        session_cookie = _default_form_cookie() or None

        if fetch_username:
            # ---- "Fetch Student Profile" -------------------------------
            lc_username = fetch_username
            try:
                student_profile, mapped = build_student_from_leetcode(
                    fetch_username, session_cookie=session_cookie, split="train"
                )
                recommended_action = mapped.recommended_action
                form_submitted = True
                selected_strategy = strategy_options[0]
                selected_student = student_profile.student_id
                eval_seed = "100"
            except LeetCodeAPIError as e:
                error = f"LeetCode API error: {str(e)[:400]}"
            except Exception as e:
                app.logger.exception("LeetCode import failed")
                error = f"Failed to initialize student: {type(e).__name__}: {str(e)[:400]}"

        elif trace_username:
            # ---- "Run Demo Trace" -------------------------------------
            lc_username = trace_username
            selected_strategy = request.form.get("strategy") or strategy_options[0]
            if selected_strategy not in strategy_options:
                error = f"Unknown strategy: {selected_strategy}"
            eval_seed = request.form.get("eval_seed") or "100"
            if not error:
                try:
                    student_profile, mapped, steps_data, trace_summary = run_leetcode_trace(
                        trace_username, selected_strategy, eval_seed,
                        session_cookie=session_cookie,
                    )
                    recommended_action = mapped.recommended_action
                    form_submitted = True
                    selected_student = student_profile.student_id
                except LeetCodeAPIError as e:
                    error = f"LeetCode API error: {str(e)[:400]}"
                except Exception as e:
                    app.logger.exception("LeetCode trace failed")
                    error = f"Failed to run trace: {type(e).__name__}: {str(e)[:400]}"
        else:
            error = "Please enter a LeetCode username."

    # Student dropdown: imported student first, then the training students.
    from data.generate_data import load_students
    train_students = load_students(split="train")
    if student_profile is not None:
        student_options = [student_profile.student_id] + [
            s.student_id for s in train_students if s.student_id != student_profile.student_id
        ]
        selected_student = selected_student or student_profile.student_id
    else:
        student_options = [s.student_id for s in train_students]

    return render_template(
        "leetcode.html",
        form_submitted=form_submitted,
        selected_strategy=selected_strategy,
        selected_student=selected_student,
        eval_seed=eval_seed,
        eval_seeds=[100, 200, 300],
        strategy_options=strategy_options,
        student_options=student_options,
        student_profile=student_profile,
        recommended_action=recommended_action,
        steps_data=steps_data,
        trace_summary=trace_summary,
        lc_username=lc_username,
        error=error,
    )


# --------------------------------------------------------------------------
# Static figure serving
# --------------------------------------------------------------------------

@app.route("/static/figures/<path:filename>")
def static_figures(filename):
    """Serve figures from visualization/figures/."""
    fig_dir = Path(__file__).resolve().parent / "visualization" / "figures"
    return send_from_directory(str(fig_dir), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)