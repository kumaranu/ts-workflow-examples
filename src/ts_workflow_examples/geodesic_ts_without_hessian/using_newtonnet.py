import logging
from ase.io import read
import os, toml, glob, shutil
from quacc import get_settings
from quacc import strip_decorator
from quacc.recipes.newtonnet.ts import ts_job, irc_job, geodesic_job


def geodesic_ts_no_hess_irc_newtonnet(reactant, product, calc_kwargs1, logger, clean_up=True):
    # Create geodesic job
    job1 = strip_decorator(geodesic_job)(reactant, product, calc_kwargs=calc_kwargs1)
    logger.info("Created Geodesic job.")

    # Create TS job with custom Hessian
    job2 = strip_decorator(ts_job)(job1['highest_e_atoms'], use_custom_hessian=False, **calc_kwargs1)
    logger.info("Created TS job without custom Hessian.")

    # Create IRC job in forward direction
    job3 = strip_decorator(irc_job)(job2['atoms'], direction='forward', **calc_kwargs1)
    logger.info("Created IRC job in forward direction.")

    # Create IRC job in reverse direction
    job4 = strip_decorator(irc_job)(job2['atoms'], direction='reverse', **calc_kwargs1)
    logger.info("Created IRC job in reverse direction.")

    logger.info("All jobs executed successfully.")

    if clean_up:
        # Delete the raw files
        directory_patterns = ["quacc-*", "tmp*"]
        
        # Delete directories matching patterns
        for pattern in directory_patterns:
            for dir_path in glob.glob(pattern):
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)

    return [job1, job2, job3, job4]



if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration from TOML file
    config = toml.load('inputs_using_newtonnet.toml')
    
    # Constants from TOML file
    REACTANT_XYZ_FILE = config['paths']['reactant']
    PRODUCT_XYZ_FILE = config['paths']['product']
    MODEL_PATH = config['paths']['model_path']
    SETTINGS_PATH = config['paths']['settings_path']

    # Read reactant and product structures
    reactant = read(REACTANT_XYZ_FILE)
    product = read(PRODUCT_XYZ_FILE)
    logger.info("Successfully read reactant and product structures.")

    settings = get_settings()
    settings.NEWTONNET_MODEL_PATH = os.getcwd() + MODEL_PATH
    settings.NEWTONNET_CONFIG_PATH = os.getcwd() + SETTINGS_PATH
    settings.CHECK_CONVERGENCE = False

    # Calculation and optimization keyword arguments
    calc_kwargs1 = {
        'hess_method': None,
    }

    job1, job2, job3, job4 = geodesic_ts_no_hess_irc_newtonnet(reactant, product, calc_kwargs1, logger)
    print('\n\n', job1)
    print('\n\n', job2)
    print('\n\n', job3)
    print('\n\n', job4)

