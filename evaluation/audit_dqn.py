"""Contract audit of the deep RL agents (DQN / Double DQN) — Phase 1 gate.

The audit codifies **ten checks**.  Checks 1-8 and 10 are *contracts*: they
must hold for any checkpoint, collapsed or not, and a failure means the
training / checkpointing pipeline itself is broken.  Check 9 (and the
``greedy_train_reward`` readout attached to it) *describes* the learned
policy: it reports whether the greedy policy has collapsed onto a handful of
actions, and is the evidence used for the "collapse gone / collapse persists"
verdict — it never fails the audit on its own.

=====  #  name                     what is asserted ===========================
   1  preprocessing_parity       the network sees exactly the observation the
                                 env produces (object identity + recomputed
                                 equality), in BOTH the training path
                                 (``training.trainer.run_episode``) and the
                                 evaluation path (``evaluate.run_session``);
                                 dtype float32, range [0,1], input dim STATE_DIM
   2  action_roundtrip           topic/difficulty <-> action id is a bijection
                                 on all 36 actions and the env agrees
   3  reward_decomposition       components sum to raw_total, clipped = clip(raw),
                                 env.step() reward == info[reward_components][clipped]
   4  terminal_invariants        terminated is never True, truncated only on the
                                 EPISODE_LEN-th step, and the ``done`` flag pushed
                                 into replay is True on exactly that last step
   5  target_invariants          DQN:  r + gamma*max Q_target(s')  (0 when done)
                                 Double: r + gamma*Q_target(s', argmax Q_online)
                                 target network receives no gradient and is not
                                 touched by the optimizer; hard update copies online
   6  replay_invariants          shapes / dtypes / index bounds / size growth /
                                 capacity wrap-around keeps the newest transitions
   7  recommend_greediness       monkeypatched *hostile* rng: recommend() still
                                 returns argmax_a Q(s,a) while act(explore=True)
                                 does go through the rng
   8  checkpoint_identity        the shipped models/<name>_seed*.pt reproduces the
                                 recorded ``best_greedy_val_reward`` of its own
                                 selection record bit-for-bit, and its config
                                 equals the current DQNConfig (no hyper-parameter
                                 drift)
   9  action_distribution        greedy rollout counts over 6 fixed TRAINING
                                 students x 2 eval seeds -> results/audit/*.csv;
                                 reports unique actions / top-action share /
                                 entropy and a collapse verdict (informational)
  10 deterministic_step_trace    the same (agent, student, env_seed) trace is
                                 reproducible and a different env_seed diverges

Usage::

    python evaluation/audit_dqn.py                 # all deep models, all seeds
    python evaluation/audit_dqn.py --models dqn    # subset

Artifacts (``results/audit/``)::

    audit_report.json            status + metrics of every check
    action_distribution.csv      per (model, seed) 36-row action-count table
    greedy_train_reward.csv      per (model, seed, student, eval_seed) readout
    step_trace.json              deterministic traces of check 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    DIFFICULTY_LABELS,
    DQNConfig,
    EPISODE_LEN,
    MODELS_DIR,
    N_ACTIONS,
    RESULTS_DIR,
    TOPICS,
)

from data.generate_data import history_lookup, load_problems, load_students  # noqa: E402
from environment.action_space import (  # noqa: E402
    ACTION_LABELS,
    action_to_topic_difficulty,
    topic_difficulty_to_action,
)
from environment.coding_env import CodingPracticeEnv  # noqa: E402
from environment.reward import RewardConfig, compute_reward  # noqa: E402
from environment.state import STATE_DIM  # noqa: E402

AUDIT_DIR = RESULTS_DIR / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed readout geometry, defined once in config.py and shared with the
# reward-sensitivity study: 6 TRAINING students x 2 evaluation seeds
# (never the held-out test split).
from config import READOUT_EVAL_SEEDS as AUDIT_EVAL_SEEDS  # noqa: E402
from config import READOUT_N_STUDENTS as AUDIT_N_STUDENTS  # noqa: E402

DEEP_MODELS = ("dqn", "double_dqn")


@dataclass
class CheckResult:
    name: str
    status: str                      # "pass" | "fail" | "skip"
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "skip")


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _audit_env() -> CodingPracticeEnv:
    return CodingPracticeEnv(
        problems=load_problems(),
        students=load_students(),
        mode="train",
    )


def train_students() -> list:
    return [s for s in load_students() if s.split == "train"]


def test_students() -> list:
    """Held-out split — must never appear in any audit readout."""
    return [s for s in load_students() if s.split == "test"]


def audit_students() -> list:
    """The 6 fixed TRAINING students used by the greedy readouts."""
    return train_students()[:AUDIT_N_STUDENTS]


def env_seed_for(eval_seed: int, student_idx: int) -> int:
    """Same (eval_seed, student) -> seed family as evaluation/evaluate.py."""
    return int(eval_seed * 100_003 + student_idx)


def _greedy_rollout(
    env: CodingPracticeEnv,
    recommender,
    *,
    student,
    history: Optional[list[dict]],
    env_seed: int,
) -> tuple[float, list[int], list[dict]]:
    """Deterministic greedy session; returns (total_reward, actions, log)."""
    recommender.reset()
    options = {"student_id": student.student_id}
    if history is not None:
        options["history"] = history
    state, _ = env.reset(seed=env_seed, options=options)
    actions: list[int] = []
    log: list[dict] = []
    total = 0.0
    while True:
        action = recommender.recommend(state)
        state, reward, terminated, truncated, info = env.step(action)
        actions.append(int(action))
        log.append(info)
        total += reward
        if terminated or truncated:
            break
    return float(total), actions, log


def load_deep_agent(name: str, seed: int, device: str | None = None):
    from agents.dqn_agent import DQNAgent
    from agents.double_dqn_agent import DoubleDQNAgent

    cls = DQNAgent if name == "dqn" else DoubleDQNAgent
    path = MODELS_DIR / f"{name}_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return cls.load(path, device=device)


# --------------------------------------------------------------------- #
# 1. preprocessing parity
# --------------------------------------------------------------------- #
def check_preprocessing_parity() -> CheckResult:
    from training.trainer import run_episode

    problems, students = load_problems(), load_students()
    history = history_lookup(problems)
    env = CodingPracticeEnv(problems=problems, students=students, mode="train")
    agent = load_deep_agent("dqn", 0) if (MODELS_DIR / "dqn_seed0.pt").exists() \
        else _fresh_agent("dqn")

    records: list[dict] = []
    last: dict[str, Any] = {"obs": None}
    orig_reset, orig_step, orig_greedy = env.reset, env.step, agent._greedy

    def reset(*a, **k):
        obs, info = orig_reset(*a, **k)
        last["obs"] = obs
        return obs, info

    def step(a):
        obs, r, te, tr, info = orig_step(a)
        last["obs"] = obs
        return obs, r, te, tr, info

    def spy(state):
        s = np.asarray(state)
        records.append(
            {
                "is_env_object": state is last["obs"],
                "matches_rebuild": bool(np.array_equal(s, env._build_state())),
                "dtype": str(s.dtype),
                "in_range": bool(s.min() >= 0.0 and s.max() <= 1.0),
                "shape": tuple(s.shape),
            }
        )
        return orig_greedy(state)

    env.reset, env.step = reset, step
    agent._greedy = spy
    try:
        # (a) evaluation path: exactly what evaluate.run_session does
        st = audit_students()[0]
        _greedy_rollout(
            env, agent, student=st, history=history.get(st.student_id),
            env_seed=env_seed_for(AUDIT_EVAL_SEEDS[0], 0),
        )
        n_eval_path = len(records)

        # (b) training path: trainer.run_episode with epsilon forced to 0 so
        #     every behaviour action goes through _greedy (still the same
        #     preprocessing pipeline as a real training step).  The DQN
        #     recomputes epsilon from cfg on every observe(), so the config
        #     itself has to be pinned for the duration of the episode.
        eps, start, end = agent.epsilon, agent.cfg.epsilon_start, agent.cfg.epsilon_end
        agent.epsilon = 0.0
        agent.cfg.epsilon_start = 0.0
        agent.cfg.epsilon_end = 0.0
        try:
            run_episode(
                env, agent, student=st,
                history=history.get(st.student_id),
                env_seed=env_seed_for(AUDIT_EVAL_SEEDS[1], 1),
                explore=True,
            )
        finally:
            agent.epsilon = eps
            agent.cfg.epsilon_start = start
            agent.cfg.epsilon_end = end
    finally:
        env.reset, env.step = orig_reset, orig_step
        agent._greedy = orig_greedy

    problems_seen = [i for i, r in enumerate(records) if not (
        r["is_env_object"] and r["matches_rebuild"] and r["in_range"]
        and r["dtype"] == "float32" and r["shape"] == (STATE_DIM,)
    )]
    in_features = agent.online.net[0].in_features
    bad = problems_seen or (in_features != STATE_DIM)
    return CheckResult(
        name="preprocessing_parity",
        status="fail" if bad else "pass",
        detail=(
            f"{len(records)} policy inputs "
            f"({n_eval_path} eval-path, {len(records) - n_eval_path} train-path); "
            f"net input dim {in_features}"
        ),
        metrics={
            "n_policy_inputs": len(records),
            "n_eval_path": n_eval_path,
            "n_train_path": len(records) - n_eval_path,
            "net_input_dim": int(in_features),
            "state_dim": int(STATE_DIM),
            "bad_inputs": [int(i) for i in problems_seen[:10]],
        },
    )


def _fresh_agent(name: str, seed: int = 0):
    from agents.dqn_agent import DQNAgent
    from agents.double_dqn_agent import DoubleDQNAgent

    cls = DQNAgent if name == "dqn" else DoubleDQNAgent
    return cls(config=DQNConfig(), seed=seed)


# --------------------------------------------------------------------- #
# 2. action roundtrip
# --------------------------------------------------------------------- #
def check_action_roundtrip() -> CheckResult:
    bad: list[int] = []
    for a in range(N_ACTIONS):
        t, d = action_to_topic_difficulty(a)
        if topic_difficulty_to_action(t, d) != a:
            bad.append(a)
        if not (0 <= t < len(TOPICS) and d in (1, 2, 3)):
            bad.append(a)
        if ACTION_LABELS[a] != f"{DIFFICULTY_LABELS[d]} {TOPICS[t]}":
            bad.append(a)

    env = _audit_env()
    obs, _ = env.reset(seed=0)
    seen: dict[tuple[str, int], int] = {}
    for a in range(N_ACTIONS):
        _, _, _, _, info = env.step(a)
        seen[(info["topic"], info["difficulty"])] = a
        if info["topic_idx"] * 3 + (info["difficulty"] - 1) != a:
            bad.append(a)
    ok = not bad and len(seen) == N_ACTIONS
    return CheckResult(
        name="action_roundtrip",
        status="pass" if ok else "fail",
        detail=f"{len(seen)}/{N_ACTIONS} (topic, difficulty) pairs reached the env",
        metrics={"n_pairs": len(seen), "bad_actions": sorted(set(bad))},
    )


# --------------------------------------------------------------------- #
# 3. reward decomposition
# --------------------------------------------------------------------- #
def check_reward_decomposition() -> CheckResult:
    cfg = RewardConfig()
    bad: list[str] = []

    for kwargs in (
        dict(skill_gain=0.013, success=True, challenge="appropriate", hints=1,
             solve_time=12.0, expected_time=10.0, frustration=0.2,
             repeated_topic=False, repeated_problem=False, same_topic_streak=1),
        dict(skill_gain=0.0, success=False, challenge="too_hard", hints=3,
             solve_time=60.0, expected_time=10.0, frustration=1.0,
             repeated_topic=True, repeated_problem=True, same_topic_streak=5),
        dict(skill_gain=0.05, success=True, challenge="too_easy", hints=0,
             solve_time=5.0, expected_time=10.0, frustration=0.0,
             repeated_topic=False, repeated_problem=False, same_topic_streak=2),
    ):
        reward, comps = compute_reward(cfg, **kwargs)
        keys = [k for k in comps if k not in ("raw_total", "clipped")]
        if abs(sum(comps[k] for k in keys) - comps["raw_total"]) > 1e-12:
            bad.append("components != raw_total")
        if abs(comps["clipped"] - min(cfg.clip_max, max(cfg.clip_min, comps["raw_total"]))) > 1e-12:
            bad.append("clipped != clip(raw)")
        if abs(reward - comps["clipped"]) > 1e-12:
            bad.append("returned != clipped")
        if not cfg.clip_min <= reward <= cfg.clip_max:
            bad.append("reward out of bounds")

    env = _audit_env()
    env.reset(seed=3)
    n_steps = 0
    for a in range(N_ACTIONS):
        _, r, _, _, info = env.step(a)
        c = info["reward_components"]
        keys = [k for k in c if k not in ("raw_total", "clipped")]
        if abs(sum(c[k] for k in keys) - c["raw_total"]) > 1e-12:
            bad.append("env: components != raw_total")
        if abs(r - c["clipped"]) > 1e-12:
            bad.append("env: reward != clipped")
        if not cfg.clip_min <= r <= cfg.clip_max:
            bad.append("env: reward out of bounds")
        n_steps += 1

    return CheckResult(
        name="reward_decomposition",
        status="fail" if bad else "pass",
        detail=f"3 unit cases + {n_steps} env steps checked",
        metrics={"n_unit_cases": 3, "n_env_steps": n_steps, "errors": sorted(set(bad))},
    )


# --------------------------------------------------------------------- #
# 4. terminal invariants
# --------------------------------------------------------------------- #
def check_terminal_invariants() -> CheckResult:
    from training.trainer import run_episode

    problems, students = load_problems(), load_students()
    history = history_lookup(problems)
    env = CodingPracticeEnv(problems=problems, students=students, mode="train")
    agent = load_deep_agent("dqn", 0) if (MODELS_DIR / "dqn_seed0.pt").exists() \
        else _fresh_agent("dqn")
    agent.buffer.pos, agent.buffer.size = 0, 0     # start from an empty buffer

    term_flags: list[bool] = []
    trunc_flags: list[bool] = []
    orig_step = env.step

    def step(a):
        obs, r, te, tr, info = orig_step(a)
        term_flags.append(bool(te))
        trunc_flags.append(bool(tr))
        return obs, r, te, tr, info

    env.step = step
    st = audit_students()[0]
    try:
        run_episode(
            env, agent, student=st, history=history.get(st.student_id),
            env_seed=env_seed_for(AUDIT_EVAL_SEEDS[0], 0), explore=True,
        )
    finally:
        env.step = orig_step

    dones = agent.buffer.d[: agent.buffer.size].astype(int).tolist()
    bad: list[str] = []
    if any(term_flags):
        bad.append("terminated must never be True")
    if trunc_flags != [False] * (EPISODE_LEN - 1) + [True]:
        bad.append("truncated only on the last step")
    if len(dones) != EPISODE_LEN:
        bad.append(f"buffer grew by {len(dones)} != EPISODE_LEN")
    if dones != [0] * (EPISODE_LEN - 1) + [1]:
        bad.append(f"done flags {dones} != 0...01")

    return CheckResult(
        name="terminal_invariants",
        status="fail" if bad else "pass",
        detail=f"{len(term_flags)} steps; done flags = {dones}",
        metrics={"n_steps": len(term_flags), "done_flags": dones,
                 "n_terminated": int(sum(term_flags)),
                 "n_truncated": int(sum(trunc_flags))},
    )


# --------------------------------------------------------------------- #
# 5. target invariants
# --------------------------------------------------------------------- #
def check_target_invariants() -> CheckResult:
    import torch

    bad: list[str] = []
    metrics: dict[str, Any] = {}
    batch = 8
    for name in DEEP_MODELS:
        if not (MODELS_DIR / f"{name}_seed0.pt").exists():
            continue
        agent = load_deep_agent(name, 0)
        cfg = agent.cfg
        g = torch.Generator().manual_seed(0)
        r = torch.randn(batch, generator=g)
        s2 = torch.randn(batch, STATE_DIM, generator=g)
        d0 = torch.zeros(batch)
        d1 = torch.ones(batch)

        with torch.no_grad():
            # --- not done: full bootstrap -------------------------------
            got = agent._targets(r, s2, d0)
            if name == "dqn":
                exp = r + cfg.gamma * agent.target(s2).max(dim=1).values
            else:
                sel = agent.online(s2).argmax(dim=1)
                exp = r + cfg.gamma * agent.target(s2).gather(
                    1, sel.unsqueeze(1)).squeeze(1)
            if not torch.allclose(got, exp, atol=1e-6):
                bad.append(f"{name}: bootstrap target mismatch")
            # --- done: no bootstrap at all ------------------------------
            got_done = agent._targets(r, s2, d1)
            if not torch.allclose(got_done, r, atol=1e-6):
                bad.append(f"{name}: done target must equal r")
            if not torch.allclose(got_done - r, torch.zeros_like(r), atol=1e-6):
                bad.append(f"{name}: bootstrap leaked into terminal target")

        # --- gradients / optimizer must not touch the target network ----
        target_before = {k: v.detach().clone() for k, v in agent.target.state_dict().items()}
        for p in agent.target.parameters():
            p.grad = None
        agent.buffer.pos, agent.buffer.size = 0, 0
        for i in range(cfg.batch_size):
            agent.buffer.push(
                np.random.default_rng(i).random(STATE_DIM).astype(np.float32),
                i % N_ACTIONS, 0.5,
                np.random.default_rng(1000 + i).random(STATE_DIM).astype(np.float32),
                False,
            )
        agent.train_step()
        grads = [p.grad for p in agent.target.parameters()]
        if any(g is not None for g in grads):
            bad.append(f"{name}: target network received gradients")
        for k, v in agent.target.state_dict().items():
            if not torch.equal(v.detach(), target_before[k]):
                bad.append(f"{name}: optimizer step changed the target network")
        online_grad = any(p.grad is not None for p in agent.online.parameters())
        if not online_grad:
            bad.append(f"{name}: online network got no gradient")

        # --- hard update copies the online network ----------------------
        agent.update_target()
        same = all(
            torch.equal(v.detach(), agent.online.state_dict()[k].detach())
            for k, v in agent.target.state_dict().items()
        )
        if not same:
            bad.append(f"{name}: update_target() does not copy online -> target")
        metrics[f"{name}_target_updates_after_train_step"] = int(agent.updates)

    if not metrics:
        bad.append("no deep checkpoints found on disk")
    return CheckResult(
        name="target_invariants",
        status="fail" if bad else "pass",
        detail="DQN/Double-DQN targets, target-network isolation, hard update",
        metrics={**metrics, "errors": sorted(set(bad))},
    )


# --------------------------------------------------------------------- #
# 6. replay invariants
# --------------------------------------------------------------------- #
def check_replay_invariants() -> CheckResult:
    from agents.dqn_agent import ReplayBuffer

    bad: list[str] = []
    cap = 50
    buf = ReplayBuffer(cap, STATE_DIM, seed=0)
    if len(buf) != 0:
        bad.append("fresh buffer must be empty")

    for i in range(cap):
        s = np.full(STATE_DIM, i / cap, dtype=np.float32)
        buf.push(s, i % N_ACTIONS, float(i), s + 1.0, i % 7 == 0)
    if len(buf) != cap:
        bad.append(f"len={len(buf)} after {cap} pushes")
    if buf.pos != 0:
        bad.append(f"pos={buf.pos} expected wrap to 0")

    bs, ba, br, bs2, bd = buf.sample(16)
    for arr, shape in ((bs, (16, STATE_DIM)), (ba, (16,)), (br, (16,)),
                       (bs2, (16, STATE_DIM)), (bd, (16,))):
        if arr.shape != shape:
            bad.append(f"sampled shape {arr.shape} != {shape}")
    if ba.min() < 0 or ba.max() >= N_ACTIONS:
        bad.append("sampled actions out of range")
    if not (bs.dtype == np.float32 and ba.dtype == np.int64 and bd.dtype == np.float32):
        bad.append("unexpected sampled dtypes")

    # wrap-around must keep the NEWEST transitions (they overwrite the oldest,
    # i.e. after pos wrapped to 0 the fresh writes land at positions 0..9)
    for i in range(cap, cap + 10):
        s = np.full(STATE_DIM, i / cap, dtype=np.float32)
        buf.push(s, i % N_ACTIONS, float(i), s, False)
    if len(buf) != cap:
        bad.append("size must saturate at capacity")
    newest = np.arange(cap, cap + 10, dtype=np.float32)
    if not np.allclose(buf.r[:10], newest):
        bad.append("wrap-around dropped the newest transitions")
    if not np.allclose(buf.r[10:], np.arange(10, cap, dtype=np.float32)):
        bad.append("wrap-around corrupted the surviving older transitions")
    if not np.array_equal(buf.a[:10], newest.astype(np.int64) % N_ACTIONS):
        bad.append("wrap-around dropped the newest actions")
    if buf.pos != 10:
        bad.append(f"pos={buf.pos} expected 10 after overflow")

    # done flag stored as float in {0,1}
    dvals = set(np.unique(buf.d[: len(buf)]).tolist())
    if not dvals <= {0.0, 1.0}:
        bad.append(f"done flags not binary: {dvals}")

    return CheckResult(
        name="replay_invariants",
        status="fail" if bad else "pass",
        detail=f"capacity={cap}, wrap-around + dtype + index checks",
        metrics={"capacity": cap, "errors": sorted(set(bad))},
    )


# --------------------------------------------------------------------- #
# 7. recommend() greediness under a hostile rng
# --------------------------------------------------------------------- #
def check_recommend_greediness() -> CheckResult:
    class HostileRNG:
        """Any use of the exploration rng is fatal (or returns 0 -> explore)."""

        def random(self):
            raise AssertionError("recommend() must not consult the exploration rng")

        integers = random
        choice = random

    bad: list[str] = []
    n = 0
    for name in DEEP_MODELS:
        path = MODELS_DIR / f"{name}_seed0.pt"
        if not path.exists():
            continue
        agent = load_deep_agent(name, 0)
        real_rng = agent.rng
        agent.rng = HostileRNG()
        try:
            agent.epsilon = 1.0          # maximally exploratory behaviour policy
            for i in range(12):
                state = np.random.default_rng(i).random(STATE_DIM).astype(np.float32)
                expected = int(np.argmax(agent.q_values(state)))
                got = agent.recommend(state)
                if got != expected:
                    bad.append(f"{name}: recommend != argmax (i={i})")
                # the monkeypatch is live: act(explore=True) must blow up
                try:
                    agent.act(state, explore=True)
                    bad.append(f"{name}: act(explore=True) ignored the hostile rng")
                except AssertionError:
                    pass
                n += 1
        finally:
            agent.rng = real_rng
            agent.epsilon = DQNConfig().epsilon_start

    if not n:
        bad.append("no deep checkpoints found on disk")
    return CheckResult(
        name="recommend_greediness",
        status="fail" if bad else "pass",
        detail=f"{n} states checked under a hostile rng (epsilon forced to 1.0)",
        metrics={"n_states": n, "errors": sorted(set(bad))},
    )


# --------------------------------------------------------------------- #
# 8. checkpoint identity
# --------------------------------------------------------------------- #
def _selection_record(name: str, seed: int) -> Optional[dict]:
    p = RESULTS_DIR / "training" / f"{name}_seed{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def check_checkpoint_identity(name: str, seed: int) -> CheckResult:
    rec = _selection_record(name, seed)
    if rec is None or "best_greedy_val_reward" not in rec:
        return CheckResult(
            name=f"checkpoint_identity[{name}_seed{seed}]",
            status="skip",
            detail="selection record has no greedy-val entry (retrain required)",
            metrics={"record": rec or {}},
        )

    agent = load_deep_agent(name, seed)
    problems, students = load_problems(), load_students()
    history = history_lookup(problems)
    env = CodingPracticeEnv(problems=problems, students=students, mode="train")
    by_id = {s.student_id: s for s in students}
    val_seed_base = seed * 7919 + 13

    rewards = []
    traces = []
    for vi, sid in enumerate(rec["val_students"]):
        rw, actions, _ = _greedy_rollout(
            env, agent, student=by_id[sid], history=history.get(sid),
            env_seed=val_seed_base + vi,
        )
        rewards.append(rw)
        traces.append(actions)
    got = float(np.mean(rewards))
    want = float(rec["best_greedy_val_reward"])
    tol = 5e-4  # the record is rounded to 4 decimals

    # determinism: the same checkpoint + seeds must replay the same trace
    replay = [
        _greedy_rollout(
            env, agent, student=by_id[sid], history=history.get(sid),
            env_seed=val_seed_base + vi,
        )[1]
        for vi, sid in enumerate(rec["val_students"])
    ]
    deterministic = replay == traces

    cfg_payload = agent.cfg.__dict__
    cfg_match = cfg_payload == DQNConfig().__dict__
    config_drift = {k: (cfg_payload.get(k), v)
                    for k, v in DQNConfig().__dict__.items()
                    if cfg_payload.get(k) != v}

    bad: list[str] = []
    if abs(got - want) > tol:
        bad.append(f"recomputed {got:.4f} != recorded {want:.4f}")
    if not deterministic:
        bad.append("greedy trace not reproducible")
    if not cfg_match:
        bad.append(f"config drift: {config_drift}")

    return CheckResult(
        name=f"checkpoint_identity[{name}_seed{seed}]",
        status="fail" if bad else "pass",
        detail=(f"greedy val reward {got:.4f} vs recorded {want:.4f}; "
                f"config {'matches' if cfg_match else 'DRIFTED'}"),
        metrics={
            "recomputed_greedy_val_reward": round(got, 6),
            "recorded_greedy_val_reward": want,
            "best_episode": rec.get("best_episode"),
            "config_match": bool(cfg_match),
            "config_drift": config_drift,
            "deterministic_trace": bool(deterministic),
            "val_students": rec.get("val_students"),
        },
    )


# --------------------------------------------------------------------- #
# 9. action distribution (readout, not a contract)
# --------------------------------------------------------------------- #
def action_distribution_rows(name: str, seed: int) -> tuple[list[dict], list[dict]]:
    """Greedy rollouts -> 36 count rows + one reward row per (student, seed)."""
    agent = load_deep_agent(name, seed)
    problems, students = load_problems(), load_students()
    history = history_lookup(problems)
    env = CodingPracticeEnv(problems=problems, students=students, mode="train")
    studs = audit_students()

    counts = np.zeros(N_ACTIONS, dtype=np.int64)
    reward_rows: list[dict] = []
    for s_idx, st in enumerate(studs):
        for eval_seed in AUDIT_EVAL_SEEDS:
            rw, actions, log = _greedy_rollout(
                env, agent, student=st, history=history.get(st.student_id),
                env_seed=env_seed_for(eval_seed, s_idx),
            )
            counts += np.bincount(actions, minlength=N_ACTIONS)
            reward_rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "student_id": st.student_id,
                    "eval_seed": eval_seed,
                    "env_seed": env_seed_for(eval_seed, s_idx),
                    "reward": rw,
                    "unique_actions": int(len(set(actions))),
                    "unique_topics": len({a // 3 for a in actions}),
                }
            )

    total = int(counts.sum())
    p = counts / max(total, 1)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    max_entropy = float(np.log(N_ACTIONS))
    rows = []
    for a in range(N_ACTIONS):
        rows.append(
            {
                "model": name,
                "seed": seed,
                "action": a,
                "label": ACTION_LABELS[a],
                "topic": TOPICS[a // 3],
                "difficulty": a % 3 + 1,
                "count": int(counts[a]),
                "prob": float(p[a]),
                "total_steps": total,
                "unique_actions": int((counts > 0).sum()),
                "top_action_share": float(p.max()),
                "entropy": entropy,
                "max_entropy": max_entropy,
            }
        )
    return rows, reward_rows


def check_action_distribution(name: str, seed: int) -> CheckResult:
    rows, reward_rows = action_distribution_rows(name, seed)
    df = pd.DataFrame(rows)
    total = int(df["count"].sum())
    expected = AUDIT_N_STUDENTS * len(AUDIT_EVAL_SEEDS) * EPISODE_LEN
    bad: list[str] = []
    if len(df) != N_ACTIONS:
        bad.append(f"{len(df)} rows != {N_ACTIONS}")
    if total != expected:
        bad.append(f"counts sum {total} != {expected}")
    if abs(float(df["prob"].sum()) - 1.0) > 1e-9:
        bad.append("probabilities do not sum to 1")

    uniq = int(df["unique_actions"].iloc[0])
    top_share = float(df["prob"].max())
    entropy = float(df["entropy"].iloc[0])
    mean_reward = float(np.mean([r["reward"] for r in reward_rows]))
    min_reward = float(np.min([r["reward"] for r in reward_rows]))
    rewards = [r["reward"] for r in reward_rows]

    # Two independent properties of the greedy policy:
    #   * diversity  - how many of the 36 (topic, difficulty) pairs it uses
    #   * quality    - the reward it actually collects on TRAINING students
    # "collapsed" requires BOTH a narrow action set and a non-positive reward;
    # a narrow but profitable policy is reported as "narrow" (still worth
    # reporting: the reference tabular policies use 15-20 of the 36 actions).
    narrow = bool(uniq < 10 or top_share > 0.50)
    reward_positive = bool(mean_reward > 0.0)
    verdict = (
        "collapsed" if (narrow and not reward_positive)
        else "narrow" if narrow
        else "healthy"
    )

    return CheckResult(
        name=f"action_distribution[{name}_seed{seed}]",
        status="fail" if bad else "pass",
        detail=(f"{uniq}/36 actions, top share {top_share:.2f}, "
                f"entropy {entropy:.2f}/{np.log(N_ACTIONS):.2f}, "
                f"greedy reward {mean_reward:+.2f} (min {min_reward:+.2f}) "
                f"-> {verdict.upper()}"),
        metrics={
            "total_steps": total,
            "n_distinct_actions": uniq,
            "top_action": str(df.loc[df["count"].idxmax(), "label"]),
            "top_action_share": round(top_share, 6),
            "entropy": round(entropy, 6),
            "max_entropy": round(float(np.log(N_ACTIONS)), 6),
            "entropy_ratio": round(entropy / float(np.log(N_ACTIONS)), 6),
            "mean_greedy_train_reward": round(mean_reward, 6),
            "min_greedy_train_reward": round(min_reward, 6),
            "n_positive_reward_cells": int(sum(r > 0 for r in rewards)),
            "n_cells": len(rewards),
            "narrow": narrow,
            "reward_positive": reward_positive,
            "verdict": verdict,
            "collapsed": verdict == "collapsed",
            "errors": sorted(set(bad)),
        },
        # the CSV itself is written by run_audit()
    )


def reference_readout_rows() -> list[dict]:
    """Same greedy readout for ALL seven strategies (context for the audit).

    Every strategy is scored on the identical (student, eval_seed) cells, so
    the DQN's action diversity and reward can be compared against the
    tabular RL agents and the baselines without any extra protocol.
    """
    from evaluation.strategies import STRATEGY_ORDER, build_strategy

    problems, students = load_problems(), load_students()
    history = history_lookup(problems)
    env = CodingPracticeEnv(problems=problems, students=students, mode="train")
    studs = audit_students()
    rows: list[dict] = []
    for name in STRATEGY_ORDER:
        seeds = (0, 1, 2) if name in ("sarsa", "q_learning", "dqn", "double_dqn") else (0,)
        for seed in seeds:
            rec = build_strategy(name, model_seed=seed, random_seed=seed)
            counts = np.zeros(N_ACTIONS, dtype=np.int64)
            rewards: list[float] = []
            for s_idx, st in enumerate(studs):
                for eval_seed in AUDIT_EVAL_SEEDS:
                    rw, actions, _ = _greedy_rollout(
                        env, rec, student=st,
                        history=history.get(st.student_id),
                        env_seed=env_seed_for(eval_seed, s_idx),
                    )
                    counts += np.bincount(actions, minlength=N_ACTIONS)
                    rewards.append(rw)
            p = counts / max(counts.sum(), 1)
            nz = p[p > 0]
            rows.append(
                {
                    "strategy": name,
                    "seed": seed,
                    "mean_reward": float(np.mean(rewards)),
                    "min_reward": float(np.min(rewards)),
                    "max_reward": float(np.max(rewards)),
                    "n_distinct_actions": int((counts > 0).sum()),
                    "n_distinct_topics": int(len({a // 3 for a in np.flatnonzero(counts)})),
                    "top_action_share": float(p.max()),
                    "entropy": float(-(nz * np.log(nz)).sum()),
                    "max_entropy": float(np.log(N_ACTIONS)),
                }
            )
    return rows


# --------------------------------------------------------------------- #
# 10. deterministic step trace
# --------------------------------------------------------------------- #
def _trace(env, agent, student, history, env_seed: int) -> list[dict]:
    _, actions, log = _greedy_rollout(
        env, agent, student=student, history=history, env_seed=env_seed
    )
    return [
        {
            "step": i,
            "action": a,
            "label": info["action_label"],
            "problem_id": info["problem_id"],
            "success": bool(info["success"]),
            "reward": round(float(info["reward_components"]["clipped"]), 10),
            "skill_gain": round(float(info["skill_gain"]), 10),
            "mastery_after": round(float(info["mastery_after"]), 10),
        }
        for i, (a, info) in enumerate(zip(actions, log))
    ]


def check_deterministic_step_trace(name: str, seed: int) -> CheckResult:
    agent = load_deep_agent(name, seed)
    problems, students = load_problems(), load_students()
    history = history_lookup(problems)
    env = CodingPracticeEnv(problems=problems, students=students, mode="train")
    st = audit_students()[0]
    h = history.get(st.student_id)
    seed_a = env_seed_for(AUDIT_EVAL_SEEDS[0], 0)
    seed_b = env_seed_for(AUDIT_EVAL_SEEDS[1], 0)

    t1 = _trace(env, agent, st, h, seed_a)
    t2 = _trace(env, agent, st, h, seed_a)
    t3 = _trace(env, agent, st, h, seed_b)

    bad: list[str] = []
    if t1 != t2:
        bad.append("same seed produced a different trace")
    if t1 == t3:
        bad.append("different env seeds produced identical traces")
    if len(t1) != EPISODE_LEN:
        bad.append(f"trace length {len(t1)} != {EPISODE_LEN}")

    return CheckResult(
        name=f"deterministic_step_trace[{name}_seed{seed}]",
        status="fail" if bad else "pass",
        detail=f"{len(t1)} steps, reproducible under env_seed={seed_a}",
        metrics={
            "env_seed": seed_a,
            "n_steps": len(t1),
            "reproducible": t1 == t2,
            "diverges_on_other_seed": t1 != t3,
            "first_actions": [t["action"] for t in t1[:5]],
            "trace": t1,
            "errors": sorted(set(bad)),
        },
    )


# --------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------- #
def run_audit(
    models: tuple[str, ...] = DEEP_MODELS,
    seeds: tuple[int, ...] = (0, 1, 2),
    out_dir: Path = AUDIT_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[CheckResult] = []

    # shared, model-independent contracts
    results.append(check_preprocessing_parity())
    results.append(check_action_roundtrip())
    results.append(check_reward_decomposition())
    results.append(check_terminal_invariants())
    results.append(check_target_invariants())
    results.append(check_replay_invariants())
    results.append(check_recommend_greediness())

    # per-checkpoint contracts + readouts
    dist_rows: list[dict] = []
    reward_rows: list[dict] = []
    traces: dict[str, Any] = {}
    for name in models:
        for seed in seeds:
            if not (MODELS_DIR / f"{name}_seed{seed}.pt").exists():
                results.append(CheckResult(
                    name=f"checkpoint_identity[{name}_seed{seed}]",
                    status="skip", detail="checkpoint missing on disk",
                ))
                continue
            results.append(check_checkpoint_identity(name, seed))
            d = check_action_distribution(name, seed)
            results.append(d)
            rows, rrows = action_distribution_rows(name, seed)
            dist_rows.extend(rows)
            reward_rows.extend(rrows)
            t = check_deterministic_step_trace(name, seed)
            results.append(t)
            traces[f"{name}_seed{seed}"] = t.metrics

    pd.DataFrame(dist_rows).to_csv(out_dir / "action_distribution.csv", index=False)
    pd.DataFrame(reward_rows).to_csv(out_dir / "greedy_train_reward.csv", index=False)
    (out_dir / "step_trace.json").write_text(json.dumps(traces, indent=2))

    reference = reference_readout_rows()
    pd.DataFrame(reference).to_csv(out_dir / "reference_readout.csv", index=False)
    report = {
        "reference_readout": reference,
        "checks": [asdict(r) for r in results],
        "n_pass": sum(r.status == "pass" for r in results),
        "n_fail": sum(r.status == "fail" for r in results),
        "n_skip": sum(r.status == "skip" for r in results),
        "ok": all(r.ok for r in results),
        "outcome_readouts": {
            f"{name}_seed{seed}": next(
                (r.metrics for r in results
                 if r.name == f"action_distribution[{name}_seed{seed}]"),
                {},
            )
            for name in models for seed in seeds
        },
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="DQN contract audit (Phase 1)")
    p.add_argument("--models", nargs="+", default=list(DEEP_MODELS),
                   choices=list(DEEP_MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    report = run_audit(models=tuple(args.models), seeds=tuple(args.seeds))

    width = max(len(c["name"]) for c in report["checks"])
    for c in report["checks"]:
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[c["status"]]
        print(f"  [{mark}] {c['name']:<{width}}  {c['detail']}")
    print(f"\n  {report['n_pass']} passed, {report['n_fail']} failed, "
          f"{report['n_skip']} skipped -> {AUDIT_DIR / 'audit_report.json'}")

    print("\n  outcome readouts (greedy policy on 6 TRAINING students x 2 eval seeds):")
    for key, m in report["outcome_readouts"].items():
        if not m:
            continue
        print(f"    {key:<20} actions={m['n_distinct_actions']:>2}/36  "
              f"top_share={m['top_action_share']:.2f}  "
              f"reward={m['mean_greedy_train_reward']:+.2f} "
              f"(min {m['min_greedy_train_reward']:+.2f})  "
              f"{m['verdict'].upper()}")
    ref = pd.DataFrame(report.get("reference_readout", []))
    if not ref.empty:
        print("\n  reference (same cells, same harness):")
        for _, r in ref.iterrows():
            print(f"    {r['strategy']:<14} s{int(r['seed'])}  "
                  f"reward={r['mean_reward']:+6.2f}  "
                  f"actions={int(r['n_distinct_actions']):>2}/36  "
                  f"top_share={r['top_action_share']:.2f}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
