import pytest

import numpy as np


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    np.random.seed(42)  # noqa: NPY002


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

    print(jobs[0]['relax_reactant']['results']['energy'])
    print(jobs[0]['relax_reactant']['results']['forces'][0, 1])
    print(jobs[0]['relax_product']['results']['energy'])
    print(jobs[0]['relax_product']['results']['forces'][0, 1])
    print(jobs[0]['neb_results']['highest_e_atoms'].get_potential_energy())
    print(jobs[1]['trajectory_results'][-1]['energy'])
    print(jobs[2]['trajectory_results'][-1]['energy'])
    print(jobs[3]['trajectory_results'][-1]['energy'])

    # Assertions
    assert jobs[0]['relax_reactant']['results']['energy'] == pytest.approx(-68.26889038085938, 1e-6)
    assert jobs[0]['relax_reactant']['results']['forces'][0, 1] == pytest.approx(0.0006616527098231, 1e-2)

    assert jobs[0]['relax_product']['results']['energy'] == pytest.approx(-63.780540466308594, 1e-6)
    assert jobs[0]['relax_product']['results']['forces'][0, 1] == pytest.approx(-0.0018400131957605481, 1e-2)

    # neb output
    assert jobs[0]['neb_results']['highest_e_atoms'].get_potential_energy() == pytest.approx(-61.5327682495, 1e-6)

    # transition state optimization output
    assert jobs[1]['trajectory_results'][-1]['energy'] == pytest.approx(-63.742549896240234, 1e-6)

    # IRC forward output
    assert jobs[2]['trajectory_results'][-1]['energy'] == pytest.approx(-67.372802734375, 1e-6)

    # IRC reverse output
    assert jobs[3]['trajectory_results'][-1]['energy'] == pytest.approx(-67.37286376953125, 1e-6)
