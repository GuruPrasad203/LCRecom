"""LeetCode API client for fetching user proficiency data.

Fetches user profile, topic-wise solved counts, acceptance rate,
and estimated skill metrics from LeetCode's public GraphQL API.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from typing import Any
from urllib.parse import quote

import requests

from config import DATA_DIR

LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
LEETCODE_PROFILE_URL = "https://leetcode.com/{username}/"
LEETCODE_ALL_PROBLEMS_URL = "https://leetcode.com/api/problems/all/"

# Cached title -> {id, slug} index so traces can link to real LeetCode problems.
PROBLEM_INDEX_PATH = DATA_DIR / "leetcode_index.json"
PROBLEM_INDEX_MAX_AGE = 7 * 24 * 3600  # refresh weekly

# LeetCode's WAF rejects default `python-requests` clients.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": "https://leetcode.com",
    "Referer": "https://leetcode.com/",
}


def slugify_title(title: str) -> str:
    """Best-guess LeetCode URL slug from a problem title ('Pow(x, n)' -> 'powx-n')."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _norm_title(title: str) -> str:
    """Normalise a title: lowercase, drop trailing annotations like '(DP)'/'()'."""
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", title or "").strip()
    return re.sub(r"\s+", " ", cleaned.lower())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text)


# Dataset phrasings -> LeetCode phrasings (token-level, for matching only).
_SYNONYMS = {"bst": ["binary", "search", "tree"]}


def _expand(tokens: list[str]) -> list[str]:
    out = []
    for tok in tokens:
        out.extend(_SYNONYMS.get(tok, [tok]))
    return out


def _token_covered(query: list[str], candidate: list[str]) -> bool:
    """True when every query token appears in the candidate title.

    Guards against misleading fuzzy hits ('Merge K Sorted Arrays' must not
    resolve to LeetCode #88 'Merge Sorted Array'): every token of the dataset
    title has to be found inside a candidate token, and only long tokens may
    match on a shared 5-character stem ('respected' ~ 'respecting').
    """
    if not query or not candidate:
        return False
    for q in query:
        if not any(
            q in c or c in q or (len(q) >= 5 and len(c) >= 5 and q[:5] == c[:5])
            for c in candidate
        ):
            return False
    return True


def _fetch_problem_index() -> dict[str, dict[str, Any]]:
    """Download LeetCode's full problem list once (single unauthenticated call)."""
    resp = requests.get(
        LEETCODE_ALL_PROBLEMS_URL, headers=DEFAULT_HEADERS, timeout=30
    )
    resp.raise_for_status()
    index: dict[str, dict[str, Any]] = {}
    for pair in resp.json().get("stat_status_pairs", []):
        stat = pair.get("stat") or {}
        title = stat.get("question__title")
        slug = stat.get("question__title_slug")
        if not title or not slug:
            continue
        index[_norm_title(title)] = {
            "id": stat.get("frontend_question_id"),
            "slug": slug,
            "title": title,
        }
    if not index:
        raise LeetCodeAPIError("LeetCode problem list came back empty")
    return index


def load_problem_index() -> dict[str, dict[str, Any]]:
    """Return the cached LeetCode title -> problem index (refreshed weekly).

    Falls back to a stale cache when offline; returns {} only if we have
    never managed to download it.
    """
    cached: dict[str, dict[str, Any]] = {}
    if PROBLEM_INDEX_PATH.exists():
        try:
            payload = json.loads(PROBLEM_INDEX_PATH.read_text())
            # Re-key with the same normalisation used at build time.
            cached = {
                _norm_title(k): v
                for k, v in (payload.get("problems") or {}).items()
            }
            if time.time() - float(payload.get("fetched_at", 0)) < PROBLEM_INDEX_MAX_AGE:
                return cached
        except (ValueError, OSError):
            cached = {}
    try:
        index = _fetch_problem_index()
    except Exception:
        return cached
    try:
        PROBLEM_INDEX_PATH.write_text(
            json.dumps({"fetched_at": time.time(), "problems": index})
        )
    except OSError:
        pass
    return index


def resolve_leetcode_problem(
    title: str, index: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Resolve a problem title to its LeetCode number and URL.

    Returns {"title", "id", "slug", "url", "matched"} where ``matched`` is:
      * "exact"  - title found verbatim (after normalisation)
      * "fuzzy"  - close title match that also passes a token-coverage check
      * "search" - no confident match: link goes to LeetCode's problem search
    """
    index = load_problem_index() if index is None else index
    key = _norm_title(title)
    query_tokens = _expand(_tokens(key))

    entry = index.get(key) or index.get(" ".join(query_tokens))
    matched = "exact"
    if entry is None:
        ranked = difflib.get_close_matches(key, index, n=5, cutoff=0.75)
        entry = next(
            (index[c] for c in ranked
             if _token_covered(query_tokens, _expand(_tokens(c)))),
            None,
        )
        matched = "fuzzy" if entry is not None else "search"

    if entry is not None:
        slug = entry.get("slug") or slugify_title(title)
        return {
            "title": title,
            "id": entry.get("id"),
            "slug": slug,
            "url": f"https://leetcode.com/problems/{slug}/",
            "matched": matched,
        }

    # No confident LeetCode counterpart (dataset paraphrase): link to search.
    return {
        "title": title,
        "id": None,
        "slug": None,
        "url": f"https://leetcode.com/problemset/?search={quote(key)}",
        "matched": "search",
    }


class LeetCodeClient:
    """Client for fetching LeetCode user data."""

    def __init__(self, username: str, session_cookie: str | None = None):
        self.username = username
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        if session_cookie:
            self.session.cookies.set(
                "LEETCODE_SESSION", session_cookie.strip(), domain=".leetcode.com"
            )
        self._last_request = 0
        self.min_delay = 0.5  # be polite to the API

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self._last_request = time.time()

    def _gql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a GraphQL request and return the parsed data dict.

        LeetCode answers schema/authorisation problems with HTTP 400 plus a
        GraphQL error body, so the body is inspected *before* raise_for_status
        — otherwise the real message ("Cannot query field ...") is masked by a
        generic "400 Bad Request".
        """
        self._rate_limit()
        payload = {"query": query, "variables": variables or {}}
        resp = self.session.post(LEETCODE_GRAPHQL_URL, json=payload, timeout=30)
        try:
            data = resp.json()
        except ValueError:
            raise LeetCodeAPIError(
                f"LeetCode returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        if isinstance(data, dict) and data.get("errors"):
            messages = "; ".join(
                str(e.get("message", e)) for e in data["errors"]
            )
            raise LeetCodeAPIError(
                f"LeetCode GraphQL error (HTTP {resp.status_code}): {messages}"
            )
        if not resp.ok:
            raise LeetCodeAPIError(f"LeetCode API error: HTTP {resp.status_code}")
        if not isinstance(data, dict) or data.get("data") is None:
            raise LeetCodeAPIError("LeetCode API returned an empty response")
        return data["data"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_profile(self) -> dict[str, Any]:
        """Fetch basic user profile (ranking, avatar, submission stats).

        LeetCode retired `matchedUser.ranking`/`reputation`/`profile2` and
        renamed `totalSubmitNum` -> `totalSubmissionNum` (now a list), so the
        query below follows the current schema and normalises the result into
        the shape the mapping layer expects.
        """
        query = """
        query getUserProfile($username: String!) {
          matchedUser(username: $username) {
            username
            submitStats {
              acSubmissionNum { difficulty count submissions }
              totalSubmissionNum { difficulty count submissions }
            }
            profile {
              ranking
              reputation
              starRating
              userAvatar
              location
              aboutMe
            }
          }
        }
        """
        data = self._gql(query, {"username": self.username})
        user = data["matchedUser"]
        if user is None:
            raise LeetCodeAPIError(f"No LeetCode user named '{self.username}'")

        ac_list = user["submitStats"]["acSubmissionNum"]
        total_list = user["submitStats"]["totalSubmissionNum"]
        by_diff = {s["difficulty"]: s for s in ac_list}
        total_by_diff = {s["difficulty"]: s for s in total_list}
        overall = total_by_diff.get("All", {})

        profile = user["profile"] or {}
        return {
            "username": user["username"],
            # This schema no longer exposes a contest rating; the mapping
            # layer falls back to acceptance/solved counts when it is None.
            "rating": None,
            "ranking": profile.get("ranking"),
            "max_rank": profile.get("starRating"),
            "reputation": profile.get("reputation"),
            "user_avatar": profile.get("userAvatar"),
            "num_solved": by_diff.get("All", {}).get("count", 0),
            "solved_by_difficulty": {
                d: by_diff.get(d, {}).get("count", 0)
                for d in ("Easy", "Medium", "Hard")
            },
            "total_submissions": overall.get("count", 0),
            # Legacy shape consumed by map_proiciency_from_leetcode().
            # Only the "All" row is kept: summing per-difficulty rows would
            # double count every submission.
            "submitStats": {
                "acSubmissionNum": [s for s in ac_list if s["difficulty"] == "All"],
                "totalSubmitNum": {
                    "count": overall.get("count", 0),
                    "submissions": overall.get("submissions", 0),
                },
            },
        }

    def fetch_topic_stats(self) -> dict[str, int]:
        """Fetch per-topic solved counts (topic -> number solved).

        `matchedUser.topicTags` was replaced by `tagProblemCounts`, which
        buckets tags as fundamental / intermediate / advanced.
        """
        query = """
        query getTagProblemCounts($username: String!) {
          matchedUser(username: $username) {
            tagProblemCounts {
              fundamental { tagName problemsSolved }
              intermediate { tagName problemsSolved }
              advanced { tagName problemsSolved }
            }
          }
        }
        """
        data = self._gql(query, {"username": self.username})
        counts = (data["matchedUser"] or {}).get("tagProblemCounts") or {}
        topic_counts: dict[str, int] = {}
        for bucket in ("fundamental", "intermediate", "advanced"):
            for tag in counts.get(bucket) or []:
                name = tag["tagName"].lower()
                solved = int(tag["problemsSolved"])
                # A tag can show up in more than one bucket: keep the max.
                topic_counts[name] = max(topic_counts.get(name, 0), solved)
        return topic_counts

    def fetch_recent_submissions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch recent submission status (success/failure + timestamps)."""
        query = """
        query recentSubmissions($username: String!, $limit: Int!) {
          recentSubmissionList(username: $username, limit: $limit) {
            id
            title
            titleSlug
            timestamp
            statusDisplay
            __typename
          }
        }
        """
        data = self._gql(query, {"username": self.username, "limit": limit})
        subs = data["recentSubmissionList"]
        return [
            {
                "title": s["title"],
                "slug": s["titleSlug"],
                "status": s["statusDisplay"],
                "timestamp": s["timestamp"],
            }
            for s in subs
        ]

    def fetch_all(self) -> dict[str, Any]:
        """Fetch profile + topic stats + recent submissions."""
        return {
            "profile": self.fetch_profile(),
            "topic_stats": self.fetch_topic_stats(),
            "recent_submissions": self.fetch_recent_submissions(),
        }


class LeetCodeAPIError(Exception):
    """Raised when LeetCode API returns an error."""

    pass