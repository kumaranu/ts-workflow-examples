import pytest
import logging
from ase.io import read
from quacc import get_settings
from pathlib import Path
from ts_workflow_examples.neb_ts_with_hessian.using_mace import neb_ts_hess_irc_mace


# Define the paths relative to the project root
project_root = Path(__file__).resolve().parents[2]


@pytest.fixture()
def setup_test_environment(tmp_path):
    reactant = read(project_root / "tests" / '000_R.xyz')
    product = read(project_root / "tests" / '000_P.xyz')

    return reactant, product


def test_neb_ts_hess_irc_mace(setup_test_environment):
    reactant, product = setup_test_environment
    calc_kwargs = {}

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    settings = get_settings()
    settings.CHECK_CONVERGENCE = False

    jobs = neb_ts_hess_irc_mace(reactant, product, calc_kwargs, logger)

    # Assertions
    assert len(jobs) == 4
