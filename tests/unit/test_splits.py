"""Unit tests for canonical and random split construction."""

from __future__ import annotations

import pytest

from vsumbench.config import CanonicalSplits, RandomSplits
from vsumbench.data.splits import build_splits, canonical_splits, random_splits

TVSUM_IDS = tuple(f"video_{i}" for i in range(1, 51))
SUMME_IDS = tuple(f"video_{i}" for i in range(1, 26))


@pytest.mark.parametrize(
    ("name", "all_ids", "n_train", "n_test"),
    [
        ("tvsum-canonical-5", TVSUM_IDS, 40, 10),
        ("summe-canonical-5", SUMME_IDS, 20, 5),
    ],
)
def test_canonical_structure(
    name: str, all_ids: tuple[str, ...], n_train: int, n_test: int
) -> None:
    folds = canonical_splits(name)
    assert len(folds) == 5
    seen: set[str] = set()
    for fold in folds:
        assert len(fold.train) == n_train
        assert len(fold.test) == n_test
        assert not set(fold.train) & set(fold.test)
        seen.update(fold.train)
        seen.update(fold.test)
    assert seen == set(all_ids)


def test_canonical_fold0_matches_published_tvsum_split() -> None:
    # Verbatim from PGL-SUM data/splits/tvsum_splits.json, fold 0.
    assert canonical_splits("tvsum-canonical-5")[0].test == (
        "video_14",
        "video_22",
        "video_31",
        "video_34",
        "video_38",
        "video_43",
        "video_45",
        "video_5",
        "video_6",
        "video_8",
    )


def test_canonical_fold4_matches_published_summe_split() -> None:
    assert canonical_splits("summe-canonical-5")[4].test == (
        "video_11",
        "video_15",
        "video_22",
        "video_24",
        "video_25",
    )


def test_unknown_canonical_name() -> None:
    with pytest.raises(ValueError, match="unknown canonical split"):
        canonical_splits("ovp-canonical-5")


def test_random_splits_are_seeded() -> None:
    ids = [f"v{i}" for i in range(30)]
    folds_a = random_splits(ids, 5, seed=42)
    folds_b = random_splits(list(reversed(ids)), 5, seed=42)  # order-insensitive
    folds_c = random_splits(ids, 5, seed=43)
    assert folds_a == folds_b
    assert folds_a != folds_c


def test_random_splits_partition() -> None:
    ids = [f"v{i}" for i in range(10)]
    folds = random_splits(ids, 5, seed=0)
    seen: set[str] = set()
    for fold in folds:
        assert len(fold.test) == 2 and len(fold.train) == 8
        assert not set(fold.train) & set(fold.test)
        seen.update(fold.test)
    assert seen == set(ids)


def test_random_splits_reject_bad_arguments() -> None:
    with pytest.raises(ValueError, match="at least one"):
        random_splits([], 3, seed=0)
    with pytest.raises(ValueError, match=">= 2"):
        random_splits(["a", "b"], 1, seed=0)
    with pytest.raises(ValueError, match="non-empty folds"):
        random_splits(["a", "b"], 3, seed=0)


def test_build_splits_canonical_validates_ids() -> None:
    folds = build_splits(CanonicalSplits(name="tvsum-canonical-5"), TVSUM_IDS)
    assert folds == canonical_splits("tvsum-canonical-5")
    with pytest.raises(ValueError, match="missing ids"):
        build_splits(CanonicalSplits(name="tvsum-canonical-5"), TVSUM_IDS[:-1])
    with pytest.raises(ValueError, match="unexpected ids"):
        build_splits(CanonicalSplits(name="tvsum-canonical-5"), (*TVSUM_IDS, "video_51"))


def test_build_splits_random_dispatch() -> None:
    spec = RandomSplits(n_folds=4, seed=7)
    assert build_splits(spec, TVSUM_IDS) == random_splits(TVSUM_IDS, 4, 7)
