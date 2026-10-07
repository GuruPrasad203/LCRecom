"""Phase 7 — printable demo: one held-out student, per-step traces.

Runs the **best RL strategy** and the **best baseline** (selected with the
same rule the report uses: highest mean skill gain over the 30 held-out
cells of ``results/evaluation/cell_means.csv``) on a single test student and
prints, for every one of the 20 recommendations:

    state   -> the compact observation handed to the policy
    action  -> (topic, difficulty) that the policy chose
    problem -> the concrete problem the environment sampled
    outcome -> solved? solve time, hints, wrong submissions, error type
    reward  -> every reward component and the clipped total

The demo only *reads* frozen artifacts (Phase 4 evaluation, Phase 3 lock).

Usage::

    python demo.py                       # best RL vs best baseline, S019
    python demo.py --student S024 --eval-seed 300
    python demo.py --strategies q_learning prerequisite
    python demo.py --out results/demo_trace.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EPISODE_LEN, EVAL_SEEDS  # noqa: E402
from data.generate_data import history_lookup, load_problems, load_students  # noqa: E402
from environment.coding_env import CodingPracticeEnv  # noqa: E402
from environment.state import (  # noqa: E402
    IDX_CONSEC_FAIL,
    IDX_DIFFICULTY,
    IDX_FRUSTRATION,
    IDX_RECENT_ACCURACY,
    IDX_STREAK,
    MASTERY_SLICE,
)
from evaluation.compare_models import best_strategy, cell_means, load_runs  # noqa: E402
from evaluation.strategies import (  # noqa: E402
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    STRATEGY_TYPE,
    build_strategy,
)

DIFF_NAMES = {0: "Easy", 1: "Medium", 2: "Hard"}


def best_pair() -> tuple[str, str]:
    """(best RL strategy, best baseline) by mean skill gain on the 30 cells."""
    cells = cell_means(load_runs())
    rl = [n for n in STRATEGY_ORDER if STRATEGY_TYPE[n] != "Baseline"]
    base = [n for n in STRATEGY_ORDER if STRATEGY_TYPE[n] == "Baseline"]
    return (
        best_strategy(cells, "skill_gain", rl),
        best_strategy(cells, "skill_gain", base),
    )


def _state_line(state, topic_idx: int) -> str:
    return (
        f"acc={float(state[IDX_RECENT_ACCURACY]):.2f} "
        f"frust={float(state[IDX_FRUSTRATION]):.2f} "
        f"last_diff={DIFF_NAMES[int(round(float(state[IDX_DIFFICULTY]) * 2))]} "
        f"consec_fail={int(round(float(state[IDX_CONSEC_FAIL]) * 5))} "
        f"streak={int(round(float(state[IDX_STREAK]) * 5))} "
        f"mastery={float(state[MASTERY_SLICE][topic_idx]):.3f}"
    )


def _reward_line(comps: dict) -> str:
    keys = [
        ("gain", "skill_gain"), ("success", "success"),
        ("challenge", "challenge"), ("hint", "hint"), ("time", "time"),
        ("frust", "frustration"), ("rep_topic", "repetition_topic"),
        ("rep_problem", "repetition_problem"), ("streak", "repetition_streak"),
    ]
    body = "  ".join(f"{lbl}={comps[k]:+.2f}" for lbl, k in keys)
    return f"{body}  | raw={comps['raw_total']:+.2f} clipped={comps['clipped']:+.2f}"


def run_trace(
    strategy: str,
    env: CodingPracticeEnv,
    *,
    student,
    history: list[dict],
    env_seed: int,
    model_seed: int = 0,
    out: TextIO,
) -> dict:
    rec = build_strategy(strategy, model_seed=model_seed, random_seed=model_seed)
    rec.reset()
    state, info = env.reset(
        seed=env_seed, options={"student_id": student.student_id, "history": history}
    )
    initial = env.mastery.copy()

    label = STRATEGY_LABELS[strategy]
    print(f"\n{'=' * 100}", file=out)
    print(f"  {label}   (model seed {model_seed}, env_seed {env_seed})", file=out)
    print(f"  student {student.student_id} / {student.profile} / "
          f"{student.name}, {EPISODE_LEN} recommendations", file=out)
    print(f"{'=' * 100}", file=out)

    total = 0.0
    for step in range(EPISODE_LEN):
        action = rec.recommend(state)
        topic_idx, difficulty = action // 3, action % 3 + 1
        state_snapshot = _state_line(state, topic_idx)
        next_state, reward, terminated, truncated, si = env.step(action)
        total += reward

        print(f"\n  step {step + 1:>2}/{EPISODE_LEN}", file=out)
        print(f"    state    {state_snapshot}", file=out)
        print(f"    action   [{action:>2}] {si['action_label']}", file=out)
        print(f"    problem  #{si['problem_id']} \"{si['problem_title']}\" "
              f"(expected {si['expected_time']:.0f} min, "
              f"mastery {si['mastery_before']:.3f}, challenge {si['challenge']})",
              file=out)
        outcome = "PASS" if si["success"] else "FAIL"
        err = si["error_type"] or "-"
        print(f"    outcome  {outcome} in {si['solve_time']:.1f} min "
              f"({si['solve_time'] / max(si['expected_time'], 1e-6):.2f}x), "
              f"hints={si['hints_used']}, wrong={si['wrong_submissions']}, "
              f"error={err}, skill_gain={si['skill_gain']:+.4f}", file=out)
        print(f"    reward   {reward:+.2f}   {_reward_line(si['reward_components'])}",
              file=out)
        state = next_state
        if terminated or truncated:
            break

    final = env.mastery.copy()
    diff = final - initial
    summary = {
        "strategy": label,
        "total_reward": float(total),
        "skill_gain_total": float(diff.sum()),
        "skill_gain_mean": float(diff.mean()),
        "completion_rate": float(
            sum(bool(h["success"]) for h in env.history) / max(len(env.history), 1)
        ),
        "calibration": float(
            sum(h["challenge"] == "appropriate" for h in env.history)
            / max(len(env.history), 1)
        ),
        "unique_topics": len({h["topic"] for h in env.history}),
        "frustration_end": float(env.frustration),
    }
    print(f"\n  {'-' * 96}", file=out)
    print(f"  {label:<22} reward={summary['total_reward']:+8.2f}  "
          f"skill_gain={summary['skill_gain_total']:.4f}  "
          f"completion={summary['completion_rate']:.2f}  "
          f"calibration={summary['calibration']:.2f}  "
          f"unique_topics={summary['unique_topics']:>2}  "
          f"frustration={summary['frustration_end']:.2f}", file=out)
    return summary


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Per-step demo trace: best RL vs best baseline on one student"
    )
    p.add_argument("--student", default=None,
                   help="held-out student id (default: first test student)")
    p.add_argument("--eval-seed", type=int, default=EVAL_SEEDS[0])
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--strategies", nargs="+", default=None,
                   help="override the auto-selected best pair")
    p.add_argument("--out", default=None, help="also write the trace to this file")
    args = p.parse_args()

    problems = load_problems()
    students = load_students(split="test")
    by_id = {s.student_id: s for s in students}
    if args.student is None:
        student = students[0]
    elif args.student in by_id:
        student = by_id[args.student]
    else:
        print(f"unknown test student {args.student!r}; "
              f"available: {sorted(by_id)}", file=sys.stderr)
        return 2

    s_idx = [s.student_id for s in students].index(student.student_id)
    env_seed = int(args.eval_seed * 100_003 + s_idx)   # same rule as evaluate.py
    history = history_lookup(problems).get(student.student_id, [])
    env = CodingPracticeEnv(problems=problems, students=students, mode="eval")

    if args.strategies:
        strategies = args.strategies
        pair_desc = "explicitly chosen"
    else:
        rl, base = best_pair()
        strategies = [rl, base]
        pair_desc = ("selected as best RL / best baseline by mean skill gain "
                     "over the 30 held-out cells")

    handle = open(args.out, "w") if args.out else None
    out: TextIO = handle or sys.stdout

    print("=" * 100, file=out)
    print("ADAPTIVE CODING-PRACTICE RECOMMENDATION — per-step demo", file=out)
    print(f"student {student.student_id} ({student.profile}, {student.name}), "
          f"eval_seed={args.eval_seed}, env_seed={env_seed}", file=out)
    print(f"strategies: {' vs '.join(STRATEGY_LABELS[s] for s in strategies)} "
          f"— {pair_desc}", file=out)
    print("=" * 100, file=out)

    summaries = [
        run_trace(s, env, student=student, history=history, env_seed=env_seed,
                  model_seed=args.model_seed, out=out)
        for s in strategies
    ]

    print(f"\n{'=' * 100}", file=out)
    print("  SUMMARY", file=out)
    for s in summaries:
        print(f"    {s['strategy']:<22} reward={s['total_reward']:+8.2f}  "
              f"skill_gain={s['skill_gain_total']:.4f}  "
              f"completion={s['completion_rate']:.2f}  "
              f"calibration={s['calibration']:.2f}  "
              f"unique_topics={s['unique_topics']:>2}", file=out)
    if len(summaries) == 2:
        a, b = summaries
        print(f"    difference ({a['strategy']} - {b['strategy']}): "
              f"reward {a['total_reward'] - b['total_reward']:+.2f}, "
              f"skill_gain {a['skill_gain_total'] - b['skill_gain_total']:+.4f}",
              file=out)
    print("=" * 100, file=out)

    if handle:
        handle.close()
        print(f"\ntrace written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
