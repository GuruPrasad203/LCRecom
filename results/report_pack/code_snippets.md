# Code snippets (verbatim source excerpts)

Each excerpt is copied at report-build time from the file at the line range shown in its header comment.

## Q-Learning update (`agents/q_learning_agent.py`)

```python
# agents/q_learning_agent.py:36-39
    def update(self, s: int, action: int, reward: float, s2: int, done: bool) -> None:
        """Q-Learning update on *discrete* state indices (unit-testable core)."""
        target = float(reward) if done else float(reward) + self.cfg.gamma * float(self.Q[s2].max())
        self.Q[s, action] += self.cfg.alpha * (target - self.Q[s, action])
```


## SARSA update (`agents/sarsa_agent.py`)

```python
# agents/sarsa_agent.py:36-46
    def update(self, s: int, action: int, reward: float, s2: int, done: bool) -> None:
        """SARSA update on *discrete* state indices (unit-testable core)."""
        if done:
            target = float(reward)
            self._cached_action = None
        else:
            # on-policy: next action from the same epsilon-greedy behaviour policy
            a2 = self.select_action(s2, explore=True)
            target = float(reward) + self.cfg.gamma * float(self.Q[s2, a2])
            self._cached_action = a2   # must be the action executed next
        self.Q[s, action] += self.cfg.alpha * (target - self.Q[s, action])
```


## DQN targets + gradient step (`agents/dqn_agent.py`)

```python
# agents/dqn_agent.py:167-171
    def _targets(self, rewards, next_states, dones) -> torch.Tensor:
        """DQN target:  r + gamma * max_a' Q_target(s', a')."""
        max_next = self.target(next_states).max(dim=1).values
        return rewards + self.cfg.gamma * max_next * (1.0 - dones)
```

```python
# agents/dqn_agent.py:172-194
    def train_step(self) -> float:
        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = self._targets(r, s2, d)
        loss = self.loss_fn(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.updates += 1
        self.last_loss = float(loss.item())
        self.last_max_q = float(q.max().item())
        return self.last_loss
```


## Double DQN target (`agents/double_dqn_agent.py`)

```python
# agents/double_dqn_agent.py:27-33
    def _targets(self, rewards, next_states, dones) -> torch.Tensor:
        """Double-DQN target:  r + gamma * Q_target(s', argmax_a Q_online(s', a))."""
        next_actions = self.online(next_states).argmax(dim=1)          # selection
        q_next = self.target(next_states).gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)                                                   # evaluation
        return rewards + self.cfg.gamma * q_next * (1.0 - dones)
```


## Reward (`environment/reward.py::compute_reward`)

```python
# environment/reward.py:81-115
def compute_reward(
    cfg: RewardConfig,
    *,
    skill_gain: float,
    success: bool,
    challenge: str,
    hints: int,
    solve_time: float,
    expected_time: float,
    frustration: float,
    repeated_topic: bool,
    repeated_problem: bool,
    same_topic_streak: int,
) -> tuple[float, dict[str, float]]:
    """Return ``(reward, components)`` for one recommendation step."""
    time_ratio = solve_time / max(expected_time, 1e-6)

    components: dict[str, float] = {
        "skill_gain": cfg.skill_gain_weight * skill_gain,
        "success": cfg.success_bonus if success else 0.0,
        "challenge": cfg.challenge_appropriate if challenge == "appropriate"
        else cfg.challenge_mismatch,
        "hint": -cfg.hint_penalty * hints,
        "time": -cfg.time_penalty * max(0.0, time_ratio - cfg.time_tolerance),
        "frustration": -cfg.frustration_penalty * frustration,
        "repetition_topic": -cfg.repetition_topic_penalty if repeated_topic else 0.0,
        "repetition_problem": -cfg.repetition_problem_penalty if repeated_problem else 0.0,
        "repetition_streak": -cfg.repetition_streak_penalty
        * max(0, same_topic_streak - 2),
    }
    reward = float(sum(components.values()))
    clipped = float(min(cfg.clip_max, max(cfg.clip_min, reward)))
    components["raw_total"] = reward
    components["clipped"] = clipped
    return clipped, components
```


## Environment step (`environment/coding_env.py::step`)

```python
# environment/coding_env.py:228-346
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.student is None:
            raise RuntimeError("call reset() before step()")

        action = int(action)
        topic_idx, difficulty = action_to_topic_difficulty(action)
        topic = TOPICS[topic_idx]
        problem = self._select_problem(topic_idx, difficulty)
        expected_time = float(problem["expected_time"])

        mastery_before = float(self.mastery[topic_idx])
        challenge = classify_challenge(mastery_before, difficulty)

        attempt = simulate_attempt(
            profile=self.student,
            topic=topic,
            mastery=mastery_before,
            difficulty=difficulty,
            expected_time=expected_time,
            frustration=self.frustration,
            consecutive_failures=self.consec_failures,
            rng=self.np_random,
        )

        skill_gain = compute_skill_gain(
            success=attempt.success,
            mastery=mastery_before,
            difficulty=difficulty,
            ability=self.student.ability,
            learning_rate=self.student.learning_rate,
            hints=attempt.hints_used,
            frustration=self.frustration,
            challenge=challenge,
        )
        self.mastery[topic_idx] = min(1.0, mastery_before + skill_gain)

        self.frustration = update_frustration(
            self.frustration,
            attempt.success,
            difficulty,
            mastery_before,
            self.student.frustration_sensitivity,
        )
        self.consec_failures = 0 if attempt.success else self.consec_failures + 1
        self.current_difficulty = difficulty

        # ---- sliding windows -------------------------------------------
        time_ratio = attempt.solve_time / max(expected_time, 1e-6)
        self.recent.append(
            {
                "success": attempt.success,
                "time_ratio": time_ratio,
                "hints": attempt.hints_used,
                "errors": attempt.wrong_submissions,
            }
        )
        if attempt.error_type is not None:
            for _ in range(attempt.wrong_submissions):
                self.recent_errors.append(attempt.error_type)

        # ---- repetition bookkeeping ------------------------------------
        repeated_topic = self.last_topic_idx == topic_idx
        repeated_problem = self.last_problem_id == int(problem["problem_id"])
        self.same_topic_streak = self.same_topic_streak + 1 if repeated_topic else 1
        self.last_topic_idx = topic_idx
        self.last_problem_id = int(problem["problem_id"])
        self.attempted.append(int(problem["problem_id"]))
        self.steps += 1

        reward, reward_components = compute_reward(
            self.reward_config,
            skill_gain=skill_gain,
            success=attempt.success,
            challenge=challenge,
            hints=attempt.hints_used,
            solve_time=attempt.solve_time,
            expected_time=expected_time,
            frustration=self.frustration,
            repeated_topic=repeated_topic and self.steps > 1,
            repeated_problem=repeated_problem and self.steps > 1,
            same_topic_streak=self.same_topic_streak,
        )

        info: dict[str, Any] = {
            "action_label": ACTION_LABELS[action],
            "topic": topic,
            "topic_idx": topic_idx,
            "difficulty": difficulty,
            "difficulty_label": DIFFICULTY_LABELS[difficulty],
            "problem_id": int(problem["problem_id"]),
            "problem_title": problem["title"],
            "expected_time": expected_time,
            "success": attempt.success,
            "solve_time": attempt.solve_time,
            "hints_used": attempt.hints_used,
            "error_type": attempt.error_type,
            "wrong_submissions": attempt.wrong_submissions,
            "p_base": attempt.p_base,
            "skill_gain": skill_gain,
            "challenge": challenge,
            "mastery_before": mastery_before,
            "mastery_after": float(self.mastery[topic_idx]),
            "frustration": self.frustration,
            "consecutive_failures": self.consec_failures,
            "repeated_topic": repeated_topic and self.steps > 1,
            "repeated_problem": repeated_problem and self.steps > 1,
            "reward_components": reward_components,
        }
        self.history.append(info)
        self._episode_log.append(info)

        terminated = False  # a session always lasts EPISODE_LEN problems
        truncated = self.steps >= self.episode_len
        obs = self._build_state()
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
```


## State vector (`environment/state.py`)

```python
# environment/state.py:42-91
FEATURE_NAMES: list[str] = (
    [f"mastery/{t}" for t in TOPICS]
    + [
        "recent_accuracy",
        "avg_solve_time_norm",
        "recent_hint_usage",
        "recent_error_rate",
        "current_difficulty_norm",
        "consecutive_failures_norm",
        "frustration",
        "same_topic_streak_norm",
    ]
    + [f"error/{e}" for e in ERROR_TYPES]
)

STATE_DIM = len(FEATURE_NAMES)  # 12 + 8 + 6 = 26

MASTERY_SLICE = slice(0, N_TOPICS)
IDX_RECENT_ACCURACY = 12
IDX_AVG_TIME = 13
IDX_HINT_USAGE = 14
IDX_ERROR_RATE = 15
IDX_DIFFICULTY = 16
IDX_CONSEC_FAIL = 17
IDX_FRUSTRATION = 18
IDX_STREAK = 19
ERROR_SLICE = slice(20, 20 + N_ERROR_TYPES)

# --------------------------------------------------------------------------
# Discrete (tabular) state
# --------------------------------------------------------------------------
# bins: skill(3) x accuracy(3) x consecutive failures(3) x difficulty(3)
#       x weak-topic fraction(2) x hint usage(2) x same-topic streak(3)
#       = 972 states   (the streak bin keeps the tabular agents on equal
#                       footing with the deep agents, at the cost of a 3x
#                       larger table -> exactly the state-space growth that
#                       motivates DQN, see RQ3)
TABULAR_SHAPE: tuple[int, ...] = (3, 3, 3, 3, 2, 2, 3)
TABULAR_N_STATES = int(np.prod(TABULAR_SHAPE))
TABULAR_FEATURES = (
    "skill_level",
    "recent_accuracy",
    "consecutive_failures",
    "current_difficulty",
    "weak_topic_fraction",
    "recent_hint_usage",
    "same_topic_streak",
)
```


## Prerequisite-violation readout (`evaluation/evaluate.py`)

```python
# evaluation/evaluate.py:77-84
        sinfo["prereq_violated"] = violated
        total_reward += reward
        log.append(sinfo)
        if terminated or truncated:
            break
    return log, initial, env.mastery.copy(), total_reward
```


## Overestimation sampling (`evaluation/overestimation.py`)

```python
# evaluation/overestimation.py:166-203
def sample_targets(env: CodingPracticeEnv, snap: dict, action: int,
                   agent, base_seed: int, pair_idx: int) -> dict:
    y_max, y_double, ys = [], [], []
    agree = 0
    for i in range(N_SAMPLES):
        restore(env, snap)
        # independent stream per sample: dynamics are identical, only the
        # RNG differs -> 32 distinct draws from the transition distribution
        env.np_random = np.random.default_rng(base_seed + pair_idx * 1009 + i)
        _s2, r, term, trunc, _info = env.step(action)
        done = float(term or trunc)
        s2 = _s2.astype(np.float32)

        with torch.no_grad():
            x = torch.as_tensor(s2, device=agent.device).unsqueeze(0)
            q_target = agent.target(x).squeeze(0)
            q_online = agent.online(x).squeeze(0)
            v_max = float(q_target.max().item())                       # DQN rule
            a_star = int(q_online.argmax().item())
            a_targ = int(q_target.argmax().item())
            v_double = float(q_target[a_star].item())                  # DDQN rule
            agree += int(a_star == a_targ)

        g = agent.cfg.gamma
        y_max.append(float(r) + g * (1.0 - done) * v_max)
        y_double.append(float(r) + g * (1.0 - done) * v_double)
        ys.append(float(r))

    return {
        "E_y_dqn_rule": float(np.mean(y_max)),
        "E_y_ddqn_rule": float(np.mean(y_double)),
        "sd_y_dqn_rule": float(np.std(y_max, ddof=1)),
        "sd_y_ddqn_rule": float(np.std(y_double, ddof=1)),
        "E_reward": float(np.mean(ys)),
        "argmax_agreement": agree / N_SAMPLES,
    }
```


## Pseudocode

```
Q-Learning (tabular, epsilon-greedy):
  for episode e = 1..2000:
    s <- state_to_index(observation); a ~ pi(.|s; eps_e)
    (s', r, done) <- env.step(a)
    Q[s,a] <- Q[s,a] + alpha * ( r + gamma * max_a' Q[s',a'] - Q[s,a] )
    eps_e linear 1.0 -> 0.05 over 1200 episodes

SARSA (same, on-policy):
    Q[s,a] <- Q[s,a] + alpha * ( r + gamma * Q[s',a'] - Q[s,a] ),  a' ~ pi(.|s')

DQN (26-dim observation, MLP 128-128, replay 50k, batch 64):
  for each env step t:
    a ~ eps_t-greedy(Q_online(.|s));  (s', r) <- env.step(a)
    store (s,a,r,s') in D;  eps_t linear 1.0 -> 0.05 over 30 000 steps
    if t >= warmup(1000) and t mod 4 == 0:
      y <- r + gamma * max_a' Q_target(s',a')     # DDQN: a* = argmax_a' Q_online(s',a'),  y <- r + gamma * Q_target(s',a*)
      minimize Huber( y - Q_online(s,a) ),  grad clip 10
    every 250 steps:  Q_target <- Q_online

PPO (implemented, excluded from the main comparison):
    clipped surrogate over GAE advantages, same env/geometry
```
