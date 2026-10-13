"""End-to-end Bench runs on the real canonical datasets (uniform reference).

Opt-in via ``-m integration``; skipped when the canonical file is not in the
cache root. This is the M3 acceptance shape: canonical 5-fold protocol, a
registry model, verified data checksums and a run manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vsumbench.bench import Bench
from vsumbench.data import download as download_mod
from vsumbench.data.sources import SOURCES
from vsumbench.data.splits import canonical_splits
from vsumbench.reporting.manifest import load_manifest

pytestmark = pytest.mark.integration

#: (canonical split name, test-video count per fold)
EXPECTED = {"tvsum": ("tvsum-canonical-5", 10), "summe": ("summe-canonical-5", 5)}


@pytest.fixture(params=sorted(EXPECTED))
def dataset(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    path = download_mod.cache_dir() / SOURCES[name].files[0].filename
    if not path.is_file():
        pytest.skip(f"{path} not present; run `vsumm download {name}`")
    return name


def test_uniform_canonical_protocol_end_to_end(dataset: str, tmp_path: Path) -> None:
    split_name, per_fold = EXPECTED[dataset]
    result = Bench(
        dataset=dataset,
        splits=split_name,
        budgets=(0.15,),
        metrics=("f1",),
        data_root=download_mod.cache_dir(),
    ).run(output_dir=tmp_path / "runs")

    # 5 folds with the canonical sizes; the folds' test sets overlap (a
    # verbatim property of the published splits), so distinct tested videos
    # can be fewer than the dataset — each (fold, video) has its own record.
    assert [len(fold["test"]) for fold in result.results["per_fold"]] == [per_fold] * 5
    tested = {
        vid
        for fold in result.results["per_fold"]
        for vid in (fold["videos"] if isinstance(fold["videos"], dict) else {})
    }
    canonical = canonical_splits(split_name)
    assert tested == {vid for fold in canonical for vid in fold.test}

    # the uniform reference is a low bar: F1@15% lands in a sane band
    f1 = result.summary["f1"]["0.15"]
    assert f1["n_folds"] == 5
    assert 0.0 <= f1["mean"] <= 1.0

    # manifest traceability: data checksum verified against the spec
    data = result.manifest["data"]
    assert isinstance(data, dict)
    files = data["files"]
    assert isinstance(files, list) and len(files) == 1
    first = files[0]
    assert isinstance(first, dict)
    assert first["matches_spec"] is True
    assert first["sha256"] == SOURCES[dataset].files[0].sha256

    # round-trips through the file the CLI/`vsumm report` consume
    reloaded = load_manifest(result.manifest_path)
    results_block = reloaded["results"]
    assert isinstance(results_block, dict)
    assert results_block["summary"] == result.summary
