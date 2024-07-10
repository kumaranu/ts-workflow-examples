import os
import torch
import pytest
import logging
from ase.io import read
from quacc import get_settings
from pathlib import Path
from ts_workflow_examples.geodesic_ts_without_hessian.using_newtonnet import geodesic_ts_no_hess_irc_newtonnet


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


def test_geodesic_ts_hess_irc_newtonnet(setup_test_environment):
    reactant, product = setup_test_environment

    # Calculation and optimization keyword arguments
    calc_kwargs1 = {
        'hess_method': None,
    }

    settings = get_settings()
    settings.NEWTONNET_MODEL_PATH = project_root / "tests" / "best_model_state.tar"
    settings.NEWTONNET_CONFIG_PATH = project_root / "tests" / "config0.yml"
    settings.CHECK_CONVERGENCE = False

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    jobs = geodesic_ts_no_hess_irc_newtonnet(reactant, product, calc_kwargs1, logger)

    # Assertions
    assert len(jobs) == 4
