"""Hand-calculated RL update tests — the Stage 4 gate.

Every expected number below is worked out by hand from the textbook update
equations, *before* any training run:

    SARSA       Q(s,a) += alpha * [ r + gamma*Q(s',a') - Q(s,a) ]       (on-policy)
    Q-Learning  Q(s,a) += alpha * [ r + gamma*max_a' Q(s',a') - Q(s,a) ] (off-policy)
"""

import numpy as np
import pytest

from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent
from config import N_ACTIONS, TabularConfig
from environment.state import TABULAR_N_STATES, state_to_index

ALPHA, GAMMA = 0.5, 0.9


def make(cls, epsilon: float, seed: int = 0):
    cfg = TabularConfig(
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=epsilon,
        epsilon_end=epsilon,
        epsilon_decay_episodes=1,
    )
    return cls(config=cfg, n_states=3, n_actions=2, seed=seed)


# --------------------------------------------------------------------- #
# 1. SARSA: worked example
#    Q[0] = [1.0, 2.0],  Q[1] = [0.5, 2.0],  epsilon = 0 (greedy behaviour)
#    transition: s=0, a=1, r=1.0 -> s'=1
#    greedy a' in s' = argmax([0.5, 2.0]) = 1
#    target  = 1.0 + 0.9 * Q[1,1] = 1.0 + 0.9*2.0 = 2.8
#    Q[0,1]  = 2.0 + 0.5*(2.8 - 2.0) = 2.4
# --------------------------------------------------------------------- #
def test_sarsa_update_hand_calculated():
    agent = make(SarsaAgent, epsilon=0.0)
    agent.Q[0] = [1.0, 2.0]
    agent.Q[1] = [0.5, 2.0]

    agent.update(s=0, action=1, reward=1.0, s2=1, done=False)

    assert agent.Q[0, 1] == pytest.approx(2.4, abs=1e-12)   # 2.0 + 0.5*(2.8-2.0)
    assert agent.Q[0, 0] == pytest.approx(1.0)               # untouched
    assert agent.Q[1, 0] == pytest.approx(0.5)               # untouched
    assert agent._cached_action == 1                         # on-policy a' cached


def test_sarsa_terminal_transition_uses_raw_reward():
    # done -> no bootstrapping: Q[0,1] = 2.0 + 0.5*(0.5 - 2.0) = 1.25
    agent = make(SarsaAgent, epsilon=0.0)
    agent.Q[0] = [1.0, 2.0]
    agent.update(s=0, action=1, reward=0.5, s2=1, done=True)
    assert agent.Q[0, 1] == pytest.approx(1.25, abs=1e-12)
    assert agent._cached_action is None


# --------------------------------------------------------------------- #
# 2. Q-Learning: worked example (same numbers, greedy behaviour)
#    target = 1.0 + 0.9 * max([0.5, 2.0]) = 2.8
#    Q[0,1] = 2.0 + 0.5*(2.8 - 2.0) = 2.4
# --------------------------------------------------------------------- #
def test_qlearning_update_hand_calculated():
    agent = make(QLearningAgent, epsilon=0.0)
    agent.Q[0] = [1.0, 2.0]
    agent.Q[1] = [0.5, 2.0]

    agent.update(s=0, action=1, reward=1.0, s2=1, done=False)

    assert agent.Q[0, 1] == pytest.approx(2.4, abs=1e-12)
    assert agent.Q[0, 0] == pytest.approx(1.0)


def test_qlearning_terminal_transition_uses_raw_reward():
    agent = make(QLearningAgent, epsilon=0.0)
    agent.Q[0] = [1.0, 2.0]
    agent.update(s=0, action=1, reward=0.5, s2=1, done=True)
    assert agent.Q[0, 1] == pytest.approx(1.25, abs=1e-12)


# --------------------------------------------------------------------- #
# 3. On-policy vs off-policy — the algorithms must *differ* when the
#    behaviour policy samples a sub-optimal action (epsilon = 1.0).
#    SARSA bootstraps with Q[s', a'_sampled],
#    Q-Learning bootstraps with max_a' Q[s', a'].
# --------------------------------------------------------------------- #
def test_sarsa_and_qlearning_differ_under_exploration():
    seed = 123
    # reproduce the exact draw made by select_action with epsilon = 1.0:
    #   rng.random() < 1.0  -> True (draw consumed), then rng.integers(2)
    probe = np.random.default_rng(seed)
    probe.random()
    sampled_a2 = int(probe.integers(2))

    # arrange Q[s'] so that the sampled action is strictly sub-optimal
    q_next = [0.5, 2.0] if sampled_a2 == 0 else [2.0, 0.5]

    sarsa = make(SarsaAgent, epsilon=1.0, seed=seed)
    ql = make(QLearningAgent, epsilon=1.0, seed=seed)
    for ag in (sarsa, ql):
        ag.Q[0] = [1.0, 2.0]
        ag.Q[1] = q_next

    sarsa.update(s=0, action=1, reward=1.0, s2=1, done=False)
    ql.update(s=0, action=1, reward=1.0, s2=1, done=False)

    # SARSA:  target = 1 + 0.9 * 0.5 = 1.45 -> Q = 2.0 + 0.5*(1.45-2.0) = 1.725
    assert sarsa.Q[0, 1] == pytest.approx(1.725, abs=1e-12)
    assert sarsa._cached_action == sampled_a2
    # Q-Learning: target = 1 + 0.9 * 2.0 = 2.8 -> Q = 2.4 (independent of behaviour)
    assert ql.Q[0, 1] == pytest.approx(2.4, abs=1e-12)
    assert ql.Q[0, 1] != sarsa.Q[0, 1], "on-policy and off-policy must differ here"


def test_sarsa_executes_the_bootstrapped_action():
    seed = 7
    agent = make(SarsaAgent, epsilon=1.0, seed=seed)
    agent.Q[1] = [0.0, 1.0]
    agent.update(s=0, action=0, reward=0.0, s2=1, done=False)
    cached = agent._cached_action
    assert cached is not None
    zeros = np.zeros(26, dtype=np.float32)
    assert agent.act(zeros, explore=True) == cached          # a' is executed
    assert agent._cached_action is None                      # cache consumed


# --------------------------------------------------------------------- #
# 4. Value iteration convergence: single self-looping state, r=1 forever
#    -> V* = r / (1 - gamma) = 1 / (1 - 0.9) = 10
# --------------------------------------------------------------------- #
def test_qlearning_converges_to_analytic_value():
    cfg = TabularConfig(alpha=0.1, gamma=0.9, epsilon_start=0.0,
                        epsilon_end=0.0, epsilon_decay_episodes=1)
    agent = QLearningAgent(config=cfg, n_states=1, n_actions=1, seed=0)
    for _ in range(2000):
        agent.update(s=0, action=0, reward=1.0, s2=0, done=False)
    assert agent.Q[0, 0] == pytest.approx(10.0, abs=0.5)   # 1/(1-0.9)


def test_sarsa_converges_to_analytic_value():
    cfg = TabularConfig(alpha=0.1, gamma=0.9, epsilon_start=0.0,
                        epsilon_end=0.0, epsilon_decay_episodes=1)
    agent = SarsaAgent(config=cfg, n_states=1, n_actions=1, seed=0)
    for _ in range(2000):
        agent.update(s=0, action=0, reward=1.0, s2=0, done=False)
    assert agent.Q[0, 0] == pytest.approx(10.0, abs=0.5)


# --------------------------------------------------------------------- #
# 5. Behaviour / lifecycle / I-O
# --------------------------------------------------------------------- #
def test_epsilon_decays_linearly_to_end():
    cfg = TabularConfig(alpha=0.1, gamma=0.9, epsilon_start=1.0,
                        epsilon_end=0.05, epsilon_decay_episodes=10)
    agent = QLearningAgent(config=cfg, n_states=4, n_actions=2, seed=0)
    eps = [agent.epsilon]
    for _ in range(10):
        agent.end_episode()
        eps.append(agent.epsilon)
    assert all(b < a for a, b in zip(eps, eps[1:])), "epsilon must decay strictly"
    assert eps[5] == pytest.approx(1.0 + 5 * (0.05 - 1.0) / 10)   # midpoint
    assert agent.epsilon == pytest.approx(0.05)                    # fully decayed


def test_recommend_is_greedy_even_with_high_epsilon():
    cfg = TabularConfig(alpha=0.1, gamma=0.9, epsilon_start=1.0,
                        epsilon_end=1.0, epsilon_decay_episodes=1)
    agent = QLearningAgent(config=cfg)      # full 972 x 36 table
    obs = np.zeros(26, dtype=np.float32)
    idx = state_to_index(obs)
    agent.Q[idx] = 0.0
    agent.Q[idx, 0] = 0.1
    agent.Q[idx, 1] = 0.9
    assert agent.recommend(obs) == 1               # greedy despite epsilon = 1
    assert agent.select_action(idx, explore=False) == 1


def test_greedy_tie_breaks_to_lowest_action():
    cfg = TabularConfig(alpha=0.1, gamma=0.9, epsilon_start=0.0,
                        epsilon_end=0.0, epsilon_decay_episodes=1)
    agent = QLearningAgent(config=cfg)      # full 972 x 36 table
    agent.Q[:] = 0.0
    obs = np.zeros(26, dtype=np.float32)
    assert agent.recommend(obs) == 0


def test_qtable_shape_is_972x36():
    agent = QLearningAgent()   # default full-size table
    assert TABULAR_N_STATES == 972           # 3*3*3*3*2*2*3
    assert N_ACTIONS == 36
    assert agent.table_size == (972, 36)
    assert agent.Q.size == 34_992            # 972 * 36


def test_save_load_roundtrip(tmp_path):
    agent = make(QLearningAgent, epsilon=0.3)
    agent.Q[:] = np.arange(agent.Q.size, dtype=np.float64).reshape(agent.Q.shape)
    path = tmp_path / "q.npz"
    agent.save(path)
    clone = QLearningAgent.load(path)
    assert np.allclose(clone.Q, agent.Q)
    assert clone.epsilon == pytest.approx(0.3)
