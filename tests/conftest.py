import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cost_model import CostModel          # noqa: E402
from src.engine import EngineConfig           # noqa: E402
from src.workload import load_workload_config # noqa: E402


@pytest.fixture(scope="session")
def slo():
    return load_workload_config()["slo"]


@pytest.fixture(scope="session")
def cost_model():
    return CostModel.load()


@pytest.fixture
def config():
    return EngineConfig.load()


@pytest.fixture
def tight_config():
    """A deliberately cramped engine, so preemption and queueing actually happen."""
    return EngineConfig.load(max_batch_size=4, kv_blocks_total=64, block_size=16)
