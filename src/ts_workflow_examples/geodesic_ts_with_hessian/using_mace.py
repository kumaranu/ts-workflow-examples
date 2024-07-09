import logging
from ase.io import read
import os, toml, glob, shutil
from quacc import get_settings
from quacc import strip_decorator
from quacc.recipes.mace.ts import ts_job, irc_job, geodesic_job


def geodesic_ts_hess_irc_mace(reactant, product, calc_kwargs, logger, clean_up=True):
    # Create NEB job
    job1 = strip_decorator(geodesic_job)(reactant, product, calc_kwargs=calc_kwargs)
    logger.info("Geodesic job done.")
    
    # Create TS job with custom Hessian
    job2 = strip_decorator(ts_job)(job1['highest_e_atoms'], use_custom_hessian=True, **calc_kwargs)
    logger.info("TS job with custom Hessian done.")

    # Create IRC job in forward direction
    job3 = strip_decorator(irc_job)(job2['atoms'], direction='forward', **calc_kwargs)
    logger.info("IRC job in forward direction done.")

    # Create IRC job in reverse direction
    job4 = strip_decorator(irc_job)(job2['atoms'], direction='reverse', **calc_kwargs)
    logger.info("IRC job in reverse direction done.")

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
    config = toml.load('inputs_using_mace.toml')
    
    # Constants from TOML file
    REACTANT_XYZ_FILE = config['paths']['reactant']
    PRODUCT_XYZ_FILE = config['paths']['product']
    
    # Read reactant and product structures
    reactant = read(REACTANT_XYZ_FILE)
    product = read(PRODUCT_XYZ_FILE)
    logger.info("Successfully read reactant and product structures.")

    settings = get_settings()
    settings.CHECK_CONVERGENCE = False
    
    # ASE Calculator keyword arguments
    calc_kwargs = {
    }

    job1, job2, job3, job4 = geodesic_ts_hess_irc_mace(reactant, product, calc_kwargs, logger)
    print('\n\n', job1)
    print('\n\n', job2)
    print('\n\n', job3)
    print('\n\n', job4)
