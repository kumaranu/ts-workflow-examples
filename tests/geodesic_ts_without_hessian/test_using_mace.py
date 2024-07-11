import pytest
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    np.random.seed(42)  # noqa: NPY002


import logging
from ase.io import read
from quacc import get_settings
from pathlib import Path
from ts_workflow_examples.geodesic_ts_without_hessian.using_mace import geodesic_ts_no_hess_irc_mace


# Define the paths relative to the project root
project_root = Path(__file__).resolve().parents[2]


@pytest.fixture()
def setup_test_environment(tmp_path):
    reactant = read(project_root / "tests" / '000_R.xyz')
    product = read(project_root / "tests" / '000_P.xyz')

    return reactant, product


def test_geodesic_ts_hess_irc_mace(setup_test_environment):
    reactant, product = setup_test_environment
    calc_kwargs = {}

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    settings = get_settings()
    settings.CHECK_CONVERGENCE = False

    jobs = geodesic_ts_no_hess_irc_mace(reactant, product, calc_kwargs, logger)

    # Assertions
    assert jobs[0]['relax_reactant']['results']['energy'] == pytest.approx(-7400.109474432057, 1e-6)
    assert jobs[0]['relax_reactant']['results']['forces'][0, 1] == pytest.approx(2.66840467e-04, 1e-2)

    assert jobs[0]['relax_product']['results']['energy'] == pytest.approx(-7395.830068714384, 1e-6)
    assert jobs[0]['relax_product']['results']['forces'][0, 1] == pytest.approx(-0.00025790767076738774, 1e-2)

    # geodesic output
    assert jobs[0]['highest_e_atoms'].get_potential_energy() == pytest.approx(-7392.679151885447, 1e-6)

    # transition state optimization output
    assert jobs[1]['trajectory_results'][-1]['energy'] == pytest.approx(-7395.245196822939, 1e-6)

    # IRC forward output
    assert jobs[2]['trajectory_results'][-1]['energy'] == pytest.approx(-7395.808953241075, 1e-6)

    # IRC reverse output
    assert jobs[3]['trajectory_results'][-1]['energy'] == pytest.approx(-7396.074596733643, 1e-6)
