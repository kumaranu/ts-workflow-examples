import torch
import pytest


import numpy as np


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    np.random.seed(42)  # noqa: NPY002


import logging
from ase.io import read
from pathlib import Path
from quacc import get_settings
from ts_workflow_examples.neb_ts_without_hessian.using_newtonnet import neb_ts_no_hess_irc_newtonnet


# Define the paths relative to the project root
project_root = Path(__file__).resolve().parents[2]


@pytest.fixture()
def setup_test_environment(tmp_path):
    reactant = read(project_root / "tests" / '000_R.xyz')
    product = read(project_root / "tests" / '000_P.xyz')

    return reactant, product


@pytest.fixture(autouse=True)
def reset_default_tensor_type():
    torch.set_default_tensor_type(torch.FloatTensor)


def test_neb_ts_no_hess_irc_newtonnet(setup_test_environment):
    reactant, product = setup_test_environment

    # Calculation and optimization keyword arguments
    calc_kwargs1 = {
        'hess_method': None,
    }
    run_neb_kwargs = {
        'max_steps': 2,
    }
    settings = get_settings()
    settings.NEWTONNET_MODEL_PATH = project_root / "tests" / "best_model_state.tar"
    settings.NEWTONNET_CONFIG_PATH = project_root / "tests" / "config0.yml"
    settings.CHECK_CONVERGENCE = False

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    jobs = neb_ts_no_hess_irc_newtonnet(
        reactant,
        product,
        run_neb_kwargs,
        calc_kwargs1,
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
    assert jobs[0]['relax_reactant']['results']['energy'] == pytest.approx(-68.26889038085938, abs=1e-6)
    assert jobs[0]['relax_reactant']['results']['forces'][0, 1] == pytest.approx(0.0006616527098231, abs=1e-2)

    assert jobs[0]['relax_product']['results']['energy'] == pytest.approx(-63.780540466308594, abs=1e-6)
    assert jobs[0]['relax_product']['results']['forces'][0, 1] == pytest.approx(-0.0018400131957605481, abs=1e-2)

    # neb output
    assert jobs[0]['neb_results']['highest_e_atoms'].positions[0, 1] == pytest.approx(1.37073848484, abs=1e-6)

    # transition state optimization output
    assert jobs[1]['trajectory_results'][-1]['energy'] == pytest.approx(-63.50568389892578, abs=1e-6)

    # IRC forward output
    assert jobs[2]['trajectory_results'][-1]['energy'] == pytest.approx(-67.26187133789062, abs=1e-6)

    # IRC reverse output
    assert jobs[3]['trajectory_results'][-1]['energy'] == pytest.approx(-68.01038360595703, abs=1e-6)
