import toml
import logging
#import jobflow as jf
from ase.io import read
from quacc import get_settings
from quacc.recipes.mace.ts import ts_job, irc_job, geodesic_job
from quacc import strip_decorator


# Load configuration from TOML file
config = toml.load('inputs_using_mace.toml')

# Constants from TOML file
REACTANT_XYZ_FILE = config['paths']['reactant']
PRODUCT_XYZ_FILE = config['paths']['product']

settings = get_settings()
settings.WORKFLOW_ENGINE = 'jobflow'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Calculation and optimization keyword arguments
calc_kwargs = {
}

def main():
    # Read reactant and product structures
    reactant = read(REACTANT_XYZ_FILE)
    product = read(PRODUCT_XYZ_FILE)
    logger.info("Successfully read reactant and product structures.")

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

    return [job1, job2, job3, job4]


if __name__ == "__main__":
    job1, job2, job3, job4 = main()
    print('\n\n', job1)
    print('\n\n', job2)
    print('\n\n', job3)
    print('\n\n', job4)

