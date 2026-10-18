"""Map LeetCode user data to internal student proficiency metrics.

Translates LeetCode's public metrics (acceptance rate, topic solved counts,
recent submission success) into the LCRecom student profile parameters:

- ability         -> derived from overall rating/acceptance rate
- learning_rate   -> inferred from recent success trend
- frustration_sensitivity -> inferred from failure streak pattern
- hint_inclination -> inferred from hint-needed pattern
- time_factor     -> inferred from solve-time vs expected-time ratio
- per-topic mastery -> mapped from LeetCode topic-wise solved counts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from leetcode_client import LeetCodeClient, LeetCodeAPIError
from config import TOPICS


# ---------------------------------------------------------------------------
# Normalisation constants (tuned so output stays in [0, 1])
# ---------------------------------------------------------------------------

# Rating -> ability [0, 1]. LC ratings go up to ~1600+; we cap at 1600.
RATING_MAX = 1600.0

# Acceptance rate -> ability adjustment factor [0.8, 1.2]
ACC_MIN, ACC_MAX = 0.10, 0.95

# Topic solved count -> mastery [0, 1]. We normalise by the max problems
# per topic in the LCRecom action space (5 per difficulty x 3 = 15).
TOPIC_MAX = 15

# Solved problems that count as "full volume" for the ability estimate.
SOLVED_FULL_CREDIT = 250

# LeetCode tag names (lower-case, as returned by tagProblemCounts) -> our
# TOPICS vocabulary. StudentProfile.mastery is keyed by the capitalised
# TOPICS names (see environment/simulator.py:mastery_array), so the values
# here are capitalised too; tags with no counterpart are ignored.
LC_TAG_TO_TOPIC = {
    "array": "Arrays",
    "string": "Strings",
    "linked list": "Linked Lists",
    "stack": "Stack",
    "queue": "Queue",
    "recursion": "Recursion",
    "tree": "Trees",
    "graph": "Graphs",
    "dynamic programming": "Dynamic Programming",
    "greedy": "Greedy",
    "binary search": "Binary Search",
    "sorting": "Sorting",
}


# Internal topic -> LeetCode tag slug (used for outbound links).
TOPIC_TO_LC_SLUG = {
    "Arrays": "array",
    "Strings": "string",
    "Linked Lists": "linked-list",
    "Stack": "stack",
    "Queue": "queue",
    "Recursion": "recursion",
    "Trees": "tree",
    "Graphs": "graph",
    "Dynamic Programming": "dynamic-programming",
    "Greedy": "greedy",
    "Binary Search": "binary-search",
    "Sorting": "sorting",
}

LC_DIFFICULTY_PARAM = {1: "EASY", 2: "MEDIUM", 3: "HARD"}


@dataclass
class MappedProficiency:
    """Mapped proficiency metrics from LeetCode data."""

    ability: float              # [0, 1] baseline problem-solving strength
    learning_rate: float        # [0.5, 1.5] how fast mastery grows
    frustration_sensitivity: float  # [0.2, 1.6] how quickly frustration builds
    hint_inclination: float     # [0, 1] propensity to request hints
    time_factor: float          # [0.6, 1.8] relative slowness/fastness
    mastery: Dict[str, float]   # per-topic mastery [0, 1], keys = TOPICS
    raw: Dict[str, Any] = field(default_factory=dict)  # original LeetCode data
    recommended_action: Dict[str, Any] = field(default_factory=dict)


def map_proiciency_from_leetcode(client: LeetCodeClient) -> MappedProficiency:
    """Map a LeetCodeClient's fetched data to internal proficiency metrics."""
    profile = client.fetch_profile()
    topic_stats = client.fetch_topic_stats()
    recent = client.fetch_recent_submissions(limit=30)

    # --- acceptance rate (accepted submissions / total submissions) ---
    submit_stats = profile.get("submitStats", {})
    accepted = sum(s["submissions"] for s in submit_stats.get("acSubmissionNum", []))
    total_submissions = submit_stats.get("totalSubmitNum", {}).get("submissions", 0)
    acceptance_rate = (
        min(1.0, accepted / total_submissions) if total_submissions else 0.0
    )

    # --- ability: contest rating when available, else acceptance + volume ---
    rating = profile.get("rating")
    if rating:
        ability = min(1.0, float(rating) / RATING_MAX)
        if acceptance_rate > 0:
            ability = min(1.0, ability * (0.8 + 0.4 * acceptance_rate / ACC_MAX))
    else:
        volume = min(1.0, profile.get("num_solved", 0) / SOLVED_FULL_CREDIT)
        ability = 0.15 + 0.45 * acceptance_rate + 0.40 * volume
    ability = min(1.0, max(0.0, ability))

    # --- per-topic mastery from LeetCode tag problem counts ---
    mastery: Dict[str, float] = {t: 0.0 for t in TOPICS}
    for lc_tag, solved in topic_stats.items():
        topic_key = LC_TAG_TO_TOPIC.get(lc_tag.lower())
        if topic_key is None:
            continue
        bounded = min(TOPIC_MAX, max(0, int(solved)))
        mastery[topic_key] = max(mastery[topic_key], bounded / TOPIC_MAX)

    # --- learning_rate / frustration / hints from recent submission history ---
    if recent:
        recent_total = len(recent)
        recent_success_rate = sum(
            1 for s in recent if s["status"] in ("Accepted", "Correct")
        ) / recent_total
        recent_failures = sum(
            1 for s in recent
            if "Failed" in s["status"] or "TLE" in s["status"] or "CE" in s["status"]
        )
        failure_rate = recent_failures / recent_total
        learning_rate = 0.5 + 1.0 * recent_success_rate
        frustration_sensitivity = 1.6 - 1.4 * failure_rate
        hint_inclination = min(1.0, 0.5 + 0.5 * (1 - recent_success_rate))
    else:
        # Submission history is private/empty: fall back to neutral defaults
        # instead of punishing the user with worst-case values.
        recent_success_rate = acceptance_rate
        learning_rate = 1.0
        frustration_sensitivity = 1.0
        hint_inclination = 0.5

    # --- time_factor: no timing signal from the public API ---
    time_factor = 1.0  # neutral

    # --- next-problem recommendation ---
    recommended_action = recommend_next_problem(mastery, recent_success_rate)

    return MappedProficiency(
        ability=round(ability, 3),
        learning_rate=round(learning_rate, 3),
        frustration_sensitivity=round(frustration_sensitivity, 3),
        hint_inclination=round(hint_inclination, 3),
        time_factor=round(time_factor, 3),
        mastery={k: round(v, 4) for k, v in mastery.items()},
        raw=profile,
        recommended_action=recommended_action,
    )


def recommend_next_problem(mastery: Dict[str, float],
                           success_rate: float) -> dict[str, Any]:
    """Recommend the next (topic, difficulty) for a LeetCode user.

    Strategy:
    1. Topics with mastery < 0.5 (solved < 8 out of 15) are candidates.
    2. Pick the lowest-mastery topic.
    3. Difficulty: Easy if success_rate < 0.3, Medium if 0.3-0.7, Hard if > 0.7.
    3. If all topics have mastery >= 0.5, suggest a review of the weakest topic.
    """
    # Find candidate topics (mastery < 0.5 = solved < 8 out of 15)
    candidates = [(t, mastery.get(t, 0.0)) for t in TOPICS
                  if mastery.get(t, 0.0) < 0.5]

    if not candidates:
        # All topics have mastery >= 0.5; suggest reviewing the weakest
        weakest = min(TOPICS, key=lambda t: mastery.get(t, 0.0))
        return {
            "topic": weakest,
            "difficulty": 1,  # Review with Easy
            "diff_label": "Easy",
            "lc_tag_slug": TOPIC_TO_LC_SLUG.get(weakest, ""),
            "lc_difficulty": LC_DIFFICULTY_PARAM[1],
            "reason": f"All topics mastered! Review weakest: {weakest} (mastery={mastery.get(weakest, 0.0):.2f})",
        }

    # Pick the topic with lowest mastery
    candidates.sort(key=lambda x: x[1])  # sort by mastery ascending
    topic = candidates[0][0]
    mastery_val = candidates[0][1]

    # Determine difficulty based on success rate
    if success_rate < 0.3:
        difficulty = 1  # Easy
        diff_label = "Easy"
    elif success_rate > 0.7:
        difficulty = 3  # Hard
        diff_label = "Hard"
    else:
        difficulty = 2  # Medium
        diff_label = "Medium"

    reason = f"Practice {diff_label} {topic} (mastery={mastery_val:.2f}; " \
             f"success rate={success_rate:.0%})"

    return {
        "topic": topic,
        "difficulty": difficulty,
        "diff_label": diff_label,
        "lc_tag_slug": TOPIC_TO_LC_SLUG.get(topic, ""),
        "lc_difficulty": LC_DIFFICULTY_PARAM[difficulty],
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Convenience: build a StudentProfile directly from a LeetCode username
# ---------------------------------------------------------------------------

def build_student_from_leetcode(
    username: str,
    session_cookie: str | None = None,
    split: str = "train",
) -> tuple["StudentProfile", "MappedProficiency"]:
    """Fetch LeetCode data and return (StudentProfile, MappedProficiency).

    This is a *initialisation* step only — the student then participates
    in the RL environment normally.
    """
    from environment.simulator import build_profile_from_metrics  # local import to avoid circularity

    client = LeetCodeClient(username, session_cookie=session_cookie)
    mapped = map_proiciency_from_leetcode(client)

    profile = build_profile_from_metrics(
        student_id=f"LC-{username}",
        name=f"LC-{username}",
        split=split,
        ability=mapped.ability,
        learning_rate=mapped.learning_rate,
        frustration_sensitivity=mapped.frustration_sensitivity,
        hint_inclination=mapped.hint_inclination,
        time_factor=mapped.time_factor,
        mastery=mapped.mastery,
    )

    return profile, mapped