"""Official split plus subject-grouped folds inside train.

Streams are concatenations of source recordings, and one source recording (and one
person) is reused across many training streams. Grouping CV folds by stream would
therefore leak people between folds, so folds are built from ``subject_group``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .validation import LeakageError, check_disjoint

LOG = logging.getLogger(__name__)


def official_assignment(manifest: pd.DataFrame, split_field: str = "official_split") -> dict[str, str]:
    return dict(zip(manifest["stream_id"], manifest[split_field]))


def assign_group_folds(groups: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """Shuffle each corpus (``fall_*``, ``adl_*``) separately, then deal groups round-robin.

    Dealing per corpus keeps falls and daily activities in every fold when there are
    enough groups. The result only depends on the set of groups and the seed.
    """
    rng = np.random.default_rng(seed)
    by_corpus: dict[str, list[str]] = {}
    for group in sorted(set(groups)):
        by_corpus.setdefault(group.split("_", 1)[0], []).append(group)
    folds: dict[str, int] = {}
    offset = 0
    for corpus in sorted(by_corpus):
        members = list(rng.permutation(by_corpus[corpus]))
        if len(members) < n_folds:
            LOG.warning("corpus %s has %d groups for %d folds: some folds get none", corpus, len(members), n_folds)
        for position, group in enumerate(members):
            folds[str(group)] = (position + offset) % n_folds
        offset += len(members)
    return folds


def check_fold_assignment(group_folds: dict[str, int], n_folds: int) -> None:
    sets = {f"fold_{k}": {g for g, f in group_folds.items() if f == k} for k in range(n_folds)}
    check_disjoint(sets, "subject groups")
    unknown = {f for f in group_folds.values() if f not in range(n_folds)}
    if unknown:
        raise LeakageError(f"invalid fold numbers {sorted(unknown)}")


def window_fold_membership(
    win_start: np.ndarray,
    win_end: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    seg_fold: np.ndarray,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For windows of one stream: majority fold and the folds each window touches.

    Returns ``(fold, touches)``: ``fold`` is -1 for a window that overlaps no segment;
    ``touches[i, k]`` is true when window ``i`` overlaps any segment of fold ``k``.
    Windows must be sorted by start and share one duration.
    """
    overlap = np.zeros((len(win_start), n_folds), dtype=np.float64)
    for start, end, fold in zip(seg_start, seg_end, seg_fold):
        first = int(np.searchsorted(win_end, start, side="right"))
        last = int(np.searchsorted(win_start, end, side="left"))
        if last <= first:
            continue
        amount = np.minimum(win_end[first:last], end) - np.maximum(win_start[first:last], start)
        overlap[first:last, int(fold)] += np.clip(amount, 0.0, None)
    touches = overlap > 0.0
    fold = np.where(touches.any(axis=1), np.argmax(overlap, axis=1), -1)
    return fold.astype(np.int64), touches


def purged_training_mask(touches: np.ndarray, held_out: int) -> np.ndarray:
    """Windows that may train the model for ``held_out``: they touch no held-out segment."""
    return ~touches[:, held_out]


def split_assignment_table(
    stream_split: dict[str, str], group_split: dict[str, str], group_folds: dict[str, int], seed: int, group_key: str
) -> pd.DataFrame:
    rows = [
        {"unit": "stream", "id": stream, "split": split, "fold": "", "grouping_key": "stream_id", "seed": seed}
        for stream, split in sorted(stream_split.items())
    ]
    rows += [
        {
            "unit": "group",
            "id": group,
            "split": split,
            "fold": group_folds.get(group, ""),
            "grouping_key": group_key,
            "seed": seed,
        }
        for group, split in sorted(group_split.items())
    ]
    return pd.DataFrame(rows)
