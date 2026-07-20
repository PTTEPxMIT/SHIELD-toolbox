from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the shared test-fixture directory (synthetic data only)."""
    return FIXTURES_DIR
