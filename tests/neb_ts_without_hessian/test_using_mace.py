import pytest


import numpy as np


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    np.random.seed(42)  # noqa: NPY002


import logging
from ase.io import read
from pathlib import Path
from quacc import get_settings
from ts_workflow_examples.neb_ts_without_hessian.using_mace import neb_ts_no_hess_irc_mace


# Define the paths relative to the project root
project_root = Path(__file__).resolve().parents[2]


@pytest.fixture()
def setup_test_environment(tmp_path):
    reactant = read(project_root / "tests" / '000_R.xyz')
    product = read(project_root / "tests" / '000_P.xyz')

    return reactant, product


def test_neb_ts_no_hess_irc_mace(setup_test_environment):
    reactant, product = setup_test_environment

    calc_kwargs = {}
    run_neb_kwargs = {
        'max_steps': 2,
    }

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    settings = get_settings()
    settings.CHECK_CONVERGENCE = False

    jobs = neb_ts_no_hess_irc_mace(
        reactant,
        product,
        run_neb_kwargs,
        calc_kwargs,
        logger,
    )

    # Assertions
    assert len(jobs) == 4

    print(jobs[0]['relax_reactant']['results']['energy'])
    print(jobs[0]['relax_reactant']['results']['forces'][0, 1])
    print(jobs[0]['relax_product']['results']['energy'])
    print(jobs[0]['relax_product']['results']['forces'][0, 1])
    # print(jobs[0]['neb_results']['highest_e_atoms'].get_potential_energy())
    print(jobs[1]['trajectory_results'][-1]['energy'])
    print(jobs[2]['trajectory_results'][-1]['energy'])
    print(jobs[3]['trajectory_results'][-1]['energy'])

    # Assertions
    assert jobs[0]['relax_reactant']['results']['energy'] == pytest.approx(-7400.1083984375, abs=1e-6)
    assert jobs[0]['relax_reactant']['results']['forces'][0, 1] == pytest.approx(0.00045277923, abs=1e-6)

    assert jobs[0]['relax_product']['results']['energy'] == pytest.approx(-7395.8291015625, abs=1e-6)
    assert jobs[0]['relax_product']['results']['forces'][0, 1] == pytest.approx(-0.002765655517578125, abs=1e-4)

    # neb output
    assert jobs[0]['neb_results']['highest_e_atoms'].positions[0, 1] == pytest.approx(1.35378979542, abs=1e-6)

    # transition state optimization output
    assert jobs[1]['trajectory_results'][-1]['energy'] == pytest.approx(-7395.718941743618, abs=1e-4)

    # IRC forward output
    assert jobs[2]['trajectory_results'][-1]['energy'] == pytest.approx(-7395.816609139422, abs=1e-6)

    # IRC reverse output
    assert jobs[3]['trajectory_results'][-1]['energy'] == pytest.approx(-7395.828279829561, abs=1e-6)
