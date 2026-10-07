"""Synthetic dataset generation (fully reproducible, seed = DATA_SEED).

Produces three files in ``data/``:

``problems.csv``       180 realistic coding problems (12 topics x 15, i.e. 5 per
                       difficulty level -> every (topic, difficulty) category
                       used by the RL action space is populated).
``students.csv``        28 profiled learners (18 train / 10 held-out test) with
                       ability, learning rate, frustration sensitivity, hint
                       inclination, time factor and *current* per-topic mastery.
``interactions.csv``   historical practice log used to seed the "recent
                       performance" part of the RL state at reset.

The student masteries in ``students.csv`` are the state **after** the logged
history, so the three files are mutually consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, DATA_SEED, TOPIC_PREREQS, TOPICS  # noqa: E402
from environment.simulator import (  # noqa: E402
    PROFILE_TEMPLATES,
    StudentProfile,
    build_profile,
    compute_skill_gain,
    simulate_attempt,
    update_frustration,
)
from environment.reward import classify_challenge  # noqa: E402

# --------------------------------------------------------------------------
# Problem titles: 5 Easy + 5 Medium + 5 Hard per topic = 15 -> 180 problems
# --------------------------------------------------------------------------
PROBLEM_TITLES: dict[str, dict[int, list[str]]] = {
    "Arrays": {
        1: ["Two Sum", "Best Time to Buy and Sell Stock", "Merge Sorted Array",
            "Move Zeroes", "Rotate Array"],
        2: ["Product of Array Except Self", "Next Permutation",
            "Longest Consecutive Sequence", "Find All Duplicates in an Array",
            "Subarray Sum Equals K"],
        3: ["Trapping Rain Water", "First Missing Positive", "Merge K Sorted Arrays",
            "Sliding Window Maximum", "Count of Range Sum"],
    },
    "Strings": {
        1: ["Valid Anagram", "Reverse String", "Valid Palindrome",
            "First Unique Character in a String", "Implement strStr"],
        2: ["Longest Substring Without Repeating Characters", "Group Anagrams",
            "Longest Palindromic Substring", "String to Integer (atoi)",
            "Encode and Decode Strings"],
        3: ["Regular Expression Matching", "Distinct Subsequences",
            "Wildcard Matching", "Minimum Window Substring", "Text Justification"],
    },
    "Linked Lists": {
        1: ["Reverse Linked List", "Merge Two Sorted Lists", "Linked List Cycle",
            "Middle of the Linked List", "Remove Duplicates from Sorted List"],
        2: ["Add Two Numbers", "Odd Even Linked List", "Remove Nth Node From End",
            "Sort List", "Rotate List"],
        3: ["Merge K Sorted Lists", "Reverse Nodes in k-Group",
            "Copy List with Random Pointer", "Flatten Multilevel Doubly Linked List",
            "Add Two Numbers II"],
    },
    "Stack": {
        1: ["Valid Parentheses", "Min Stack", "Baseball Game",
            "Next Greater Element I", "Remove All Adjacent Duplicates In String"],
        2: ["Daily Temperatures", "Next Greater Element II", "Decode String",
            "Evaluate Reverse Polish Notation", "Asteroid Collision"],
        3: ["Largest Rectangle in Histogram", "Maximal Rectangle", "Basic Calculator",
            "Basic Calculator II", "Score of Parentheses"],
    },
    "Queue": {
        1: ["Implement Queue using Stacks", "Moving Average from Data Stream",
            "Number of Students Unable to Eat Lunch", "Recent Counter",
            "Design Circular Queue (Easy)"],
        2: ["Queue Reconstruction by Height", "Design Circular Queue",
            "Design Front Middle Back Queue", "Rotting Oranges", "Open the Lock"],
        3: ["Shortest Path in a Grid with Obstacles Elimination",
            "Find Median from Data Stream", "Design Skiplist", "Sliding Window Median",
            "Design Hit Counter"],
    },
    "Recursion": {
        1: ["Fibonacci Number", "Climbing Stairs", "Sum of Digits", "Power of Four",
            "Number of Steps to Reduce a Number to Zero"],
        2: ["Generate Parentheses", "Permutations", "Combinations", "Subset Sum",
            "Decode Ways"],
        3: ["N-Queens", "Sudoku Solver", "Word Break II", "Expression Add Operators",
            "Rat in a Maze"],
    },
    "Trees": {
        1: ["Invert Binary Tree", "Maximum Depth of Binary Tree", "Same Tree",
            "Symmetric Tree", "Path Sum"],
        2: ["Binary Tree Level Order Traversal", "Validate Binary Search Tree",
            "Lowest Common Ancestor of a BST", "Binary Tree Zigzag Level Order Traversal",
            "Construct Binary Tree from Preorder and Inorder Traversal"],
        3: ["Binary Tree Maximum Path Sum", "Recover Binary Search Tree",
            "Count of Smaller Numbers After Self",
            "Serialize and Deserialize Binary Tree", "Populating Next Right Pointers"],
    },
    "Graphs": {
        1: ["Find the Town Judge", "Find if Path Exists in Graph",
            "Number of Provinces", "Detect Cycle in an Undirected Graph",
            "Find Center of Star Graph"],
        2: ["Number of Islands", "Course Schedule", "Clone Graph", "Word Ladder",
            "Is Graph Bipartite"],
        3: ["Alien Dictionary", "Word Ladder II", "Reconstruct Itinerary",
            "Cheapest Flights Within K Stops", "Minimum Height Trees"],
    },
    "Dynamic Programming": {
        1: ["Climbing Stairs (DP)", "House Robber", "Min Cost Climbing Stairs",
            "Counting Bits", "Nth Tribonacci Number"],
        2: ["Coin Change", "Longest Increasing Subsequence", "Unique Paths",
            "Partition Equal Subset Sum", "Word Break (DP)"],
        3: ["Edit Distance", "Longest Valid Parentheses", "Burst Balloons",
            "Minimum Window Subsequence", "Stone Game II"],
    },
    "Greedy": {
        1: ["Assign Cookies", "Lemonade Change", "Best Time to Buy and Sell Stock II",
            "Maximum Units on a Truck", "Can Place Flowers"],
        2: ["Jump Game", "Gas Station", "Partition Labels", "Task Scheduler",
            "Minimum Number of Arrows to Burst Balloons"],
        3: ["Candy", "IPO", "Remove K Digits", "Meeting Rooms III",
            "Minimum Incompatibility"],
    },
    "Binary Search": {
        1: ["Binary Search", "Search Insert Position", "First Bad Version", "Sqrt(x)",
            "Guess Number Higher or Lower"],
        2: ["Search in Rotated Sorted Array", "Find Peak Element", "Search a 2D Matrix",
            "Capacity to Ship Packages Days", "Find Smallest Letter Greater Than Target"],
        3: ["Median of Two Sorted Arrays", "Split Array Largest Sum",
            "Find in Mountain Array", "Kth Smallest Element in a Sorted Matrix",
            "Smallest Range Covering Elements from K Lists"],
    },
    "Sorting": {
        1: ["Sort Array by Parity", "Relative Sort Array", "Height Checker",
            "Third Maximum Number", "Missing Number"],
        2: ["Sort Colors", "Kth Largest Element in an Array", "Sort an Array",
            "Merge Intervals", "Reorder Data in Log Files"],
        3: ["Maximum Gap", "Sort Items by Groups Respected Dependencies",
            "Count Inversions", "Meeting Rooms II", "Skyline Problem"],
    },
}

EXPECTED_TIME_RANGES = {1: (8, 18), 2: (20, 40), 3: (45, 85)}


# --------------------------------------------------------------------------
def generate_problems(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 100
    for topic in TOPICS:
        for difficulty in (1, 2, 3):
            lo, hi = EXPECTED_TIME_RANGES[difficulty]
            for title in PROBLEM_TITLES[topic][difficulty]:
                pid += 1
                rows.append(
                    {
                        "problem_id": pid,
                        "title": title,
                        "topic": topic,
                        "difficulty": difficulty,
                        "prerequisite_topics": ";".join(
                            __import__("config").TOPIC_PREREQS[topic]
                        ),
                        "expected_time": int(rng.integers(lo, hi + 1)),
                    }
                )
    df = pd.DataFrame(rows).sort_values("problem_id").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
FIRST_NAMES = [
    "Aarav", "Bianca", "Chen", "Dmitri", "Elena", "Farhan", "Grace", "Hana",
    "Ibrahim", "Julia", "Kofi", "Lucia", "Mei", "Nikita", "Olivia", "Priya",
    "Quentin", "Rosa", "Samir", "Tara", "Umar", "Vera", "Wei", "Ximena",
    "Yusuf", "Zoe", "Amir", "Nora",
]


def generate_students(rng: np.random.Generator) -> list[StudentProfile]:
    students: list[StudentProfile] = []
    counter = 0
    profile_names = list(PROFILE_TEMPLATES)

    def add(profile_name: str, split: str, name_suffix: str = "") -> None:
        nonlocal counter
        counter += 1
        sid = f"S{counter:03d}"
        name = f"{FIRST_NAMES[(counter - 1) % len(FIRST_NAMES)]}{name_suffix}"
        students.append(
            build_profile(profile_name, sid, name, split, rng, jitter=0.05)
        )

    # 18 training students (2 per profile)
    for p in profile_names:
        add(p, "train")
        add(p, "train")
    # 10 held-out test students (1 per profile + one extra intermediate)
    for p in profile_names:
        add(p, "test")
    add("intermediate", "test")
    return students


# --------------------------------------------------------------------------
def generate_history(
    students: list[StudentProfile], problems: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[StudentProfile]]:
    """Simulate a historical practice log and return the *updated* profiles."""
    pools: dict[tuple[str, int], pd.DataFrame] = {
        (t, d): problems[(problems.topic == t) & (problems.difficulty == d)]
        for t in TOPICS
        for d in (1, 2, 3)
    }
    rows = []
    updated: list[StudentProfile] = []
    iid = 0
    day = pd.Timestamp("2025-01-06")

    for student in students:
        mastery = dict(student.mastery)
        frustration = 0.0
        consec = 0
        for k in range(40):
            topic = TOPICS[int(rng.integers(len(TOPICS)))]
            m = mastery[topic]
            # historical difficulty was picked with a mild personal heuristic
            r = rng.random()
            if m < 0.40:
                difficulty = 1 if r < 0.75 else 2
            elif m < 0.70:
                difficulty = 2 if r < 0.70 else (1 if r < 0.85 else 3)
            else:
                difficulty = 3 if r < 0.45 else (2 if r < 0.9 else 1)

            pool = pools[(topic, difficulty)]
            row = pool.iloc[int(rng.integers(len(pool)))]
            challenge = classify_challenge(m, difficulty)
            att = simulate_attempt(
                profile=student,
                topic=topic,
                mastery=m,
                difficulty=difficulty,
                expected_time=float(row["expected_time"]),
                frustration=frustration,
                consecutive_failures=consec,
                rng=rng,
            )
            gain = compute_skill_gain(
                success=att.success,
                mastery=m,
                difficulty=difficulty,
                ability=student.ability,
                learning_rate=student.learning_rate,
                hints=att.hints_used,
                frustration=frustration,
                challenge=challenge,
            )
            before = m
            m = min(1.0, m + gain)
            mastery[topic] = m
            frustration = update_frustration(
                frustration, att.success, difficulty, before,
                student.frustration_sensitivity,
            )
            consec = 0 if att.success else consec + 1

            iid += 1
            rows.append(
                {
                    "interaction_id": iid,
                    "student_id": student.student_id,
                    "problem_id": int(row["problem_id"]),
                    "topic": topic,
                    "difficulty": difficulty,
                    "success": int(att.success),
                    "solve_time": round(att.solve_time, 2),
                    "hints_used": att.hints_used,
                    "error_type": att.error_type or "",
                    "wrong_submissions": att.wrong_submissions,
                    "skill_gain": round(gain, 5),
                    "timestamp": (day - pd.Timedelta(days=int(rng.integers(0, 45)))).isoformat(),
                }
            )
        updated.append(student.with_mastery(mastery))

    return pd.DataFrame(rows), updated


# --------------------------------------------------------------------------
def students_to_frame(students: list[StudentProfile]) -> pd.DataFrame:
    rows = []
    for s in students:
        row = {
            "student_id": s.student_id,
            "name": s.name,
            "profile": s.profile,
            "split": s.split,
            "ability": round(s.ability, 3),
            "learning_rate": round(s.learning_rate, 3),
            "frustration_sensitivity": round(s.frustration_sensitivity, 3),
            "hint_inclination": round(s.hint_inclination, 3),
            "time_factor": round(s.time_factor, 3),
            "error_bias": ";".join(f"{k}:{v}" for k, v in s.error_bias.items()),
        }
        for t in TOPICS:
            row[f"mastery_{t}"] = round(s.mastery[t], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def frame_to_students(df: pd.DataFrame) -> list[StudentProfile]:
    students = []
    for _, r in df.iterrows():
        error_bias = {}
        for part in str(r["error_bias"]).split(";"):
            if ":" in part:
                k, v = part.split(":")
                error_bias[k] = float(v)
        mastery = {t: float(r[f"mastery_{t}"]) for t in TOPICS}
        students.append(
            StudentProfile(
                student_id=r["student_id"],
                name=r["name"],
                profile=r["profile"],
                split=r["split"],
                ability=float(r["ability"]),
                learning_rate=float(r["learning_rate"]),
                frustration_sensitivity=float(r["frustration_sensitivity"]),
                hint_inclination=float(r["hint_inclination"]),
                time_factor=float(r["time_factor"]),
                error_bias=error_bias,
                mastery=mastery,
            )
        )
    return students


def generate_all(out_dir: Path = DATA_DIR, seed: int = DATA_SEED) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = generate_problems(rng)
    students = generate_students(rng)
    interactions, updated = generate_history(students, problems, rng)

    students_df = students_to_frame(updated)
    problems_path = out_dir / "problems.csv"
    students_path = out_dir / "students.csv"
    interactions_path = out_dir / "interactions.csv"
    problems.to_csv(problems_path, index=False)
    students_df.to_csv(students_path, index=False)
    interactions.to_csv(interactions_path, index=False)

    return {
        "problems": problems_path,
        "students": students_path,
        "interactions": interactions_path,
    }


def load_problems(path: Path = DATA_DIR / "problems.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_students(split: str | None = None, path: Path = DATA_DIR / "students.csv") -> list[StudentProfile]:
    df = pd.read_csv(path)
    if split is not None:
        df = df[df["split"] == split]
    return frame_to_students(df.reset_index(drop=True))


def load_interactions(path: Path = DATA_DIR / "interactions.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["error_type"] = df["error_type"].fillna("")
    return df


def state_history_for(interactions: pd.DataFrame, student_id: str,
                      problems: pd.DataFrame) -> list[dict]:
    """History entries (last 10 attempts) with correct time ratios."""
    etime = problems.set_index("problem_id")["expected_time"].to_dict()
    sub = interactions[interactions.student_id == student_id]
    sub = sub.sort_values("interaction_id").tail(10)
    hist = []
    for _, r in sub.iterrows():
        exp = float(etime.get(int(r["problem_id"]), 10.0))
        et = r["error_type"]
        hist.append(
            {
                "success": bool(r["success"]),
                "time_ratio": float(r["solve_time"]) / max(exp, 1e-6),
                "hints": int(r["hints_used"]),
                "errors": int(r["wrong_submissions"]),
                "error_types": [et] if isinstance(et, str) and et else [],
                "problem_id": int(r["problem_id"]),
                "topic": r["topic"],
                "difficulty": int(r["difficulty"]),
            }
        )
    return hist


def history_lookup(
    problems: pd.DataFrame | None = None,
    interactions: pd.DataFrame | None = None,
) -> dict[str, list[dict]]:
    """{student_id: last-10-attempt history} used to seed the RL state."""
    problems = problems if problems is not None else load_problems()
    interactions = interactions if interactions is not None else load_interactions()
    return {
        sid: state_history_for(interactions, sid, problems)
        for sid in interactions["student_id"].unique()
    }


if __name__ == "__main__":
    paths = generate_all()
    for k, v in paths.items():
        print(f"{k:14s} -> {v}")
