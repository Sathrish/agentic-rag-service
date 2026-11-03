import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import CORPUS, FakeCrossEncoder, FakeEncoder  # noqa: E402
from ragagent.search import SearchEngine  # noqa: E402


@pytest.fixture
def engine():
    ids = [f"D{i}" for i in range(len(CORPUS))]
    return SearchEngine(ids, [t for t, _ in CORPUS], [x for _, x in CORPUS],
                        FakeEncoder(), FakeCrossEncoder())
