"""Split construction: published canonical splits and seeded random k-folds.

Both variants produce :class:`~vsumbench.types.Split` objects over dataset
video ids. Canonical splits are verbatim transcriptions
(:mod:`vsumbench.data._canonical_splits`); random splits are fully
determined by their seed — same ids + seed, same folds, on every platform.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vsumbench.config import CanonicalSplits, RandomSplits, SplitSpec
from vsumbench.data._canonical_splits import CANONICAL_SPLITS
from vsumbench.types import Split

__all__ = [
    "CANONICAL_SPLIT_NAMES",
    "build_splits",
    "canonical_splits",
    "random_splits",
]

#: Published split collections available for ``CanonicalSplits.name``.
CANONICAL_SPLIT_NAMES: frozenset[str] = frozenset(CANONICAL_SPLITS)


def canonical_splits(name: str) -> tuple[Split, ...]:
    """Return the published folds of ``name`` (e.g. ``"tvsum-canonical-5"``)."""
    try:
        folds = CANONICAL_SPLITS[name]
    except KeyError:
        known = ", ".join(sorted(CANONICAL_SPLIT_NAMES)) or "<none>"
        raise ValueError(f"unknown canonical split {name!r}; available: {known}") from None
    return tuple(Split(train=train, test=test) for train, test in folds)


def random_splits(video_ids: Sequence[str], n_folds: int, seed: int) -> tuple[Split, ...]:
    """Seeded k-fold splits over ``video_ids``; fold ``i`` tests on chunk ``i``.

    Ids are de-duplicated and sorted before shuffling, so the result depends
    only on the id set, the fold count and the seed.
    """
    ids = sorted(set(video_ids))
    if not ids:
        raise ValueError("random_splits needs at least one video id")
    if n_folds < 2:
        raise ValueError(f"random_splits.n_folds must be >= 2, got {n_folds!r}")
    if n_folds > len(ids):
        raise ValueError(f"cannot build {n_folds} non-empty folds from {len(ids)} video ids")
    perm = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in perm]
    return tuple(
        Split(
            train=tuple(sorted(v for j, v in enumerate(shuffled) if j not in fold)),
            test=tuple(sorted(v for j, v in enumerate(shuffled) if j in fold)),
        )
        for fold in map(set, np.array_split(range(len(ids)), n_folds))
    )


def build_splits(spec: SplitSpec, video_ids: Sequence[str]) -> tuple[Split, ...]:
    """Dispatch on a config split spec, validating it against loaded ids.

    Canonical splits must match the loaded id set exactly: a canonical
    protocol is only defined for the dataset it was published on.
    """
    if isinstance(spec, CanonicalSplits):
        folds = canonical_splits(spec.name)
        expected = {vid for fold in folds for vid in fold.train + fold.test}
        loaded = set(video_ids)
        missing = sorted(expected - loaded)
        extra = sorted(loaded - expected)
        if missing or extra:
            raise ValueError(
                f"canonical split {spec.name!r} does not match the loaded dataset: "
                f"missing ids {missing}, unexpected ids {extra}"
            )
        return folds
    if isinstance(spec, RandomSplits):
        return random_splits(video_ids, spec.n_folds, spec.seed)
    raise TypeError(f"unsupported split spec {type(spec).__name__}")
