import toml
import logging
from ase.io import read
from quacc.recipes.mace.ts import ts_job, irc_job, neb_job
import jobflow as jf

# Load configuration from TOML file
config = toml.load('inputs_using_mace.toml')

# Constants from TOML file
REACTANT_XYZ_FILE = config['paths']['reactant']
PRODUCT_XYZ_FILE = config['paths']['product']
TAG = config['run']['tag']

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Calculation and optimization keyword arguments
calc_kwargs1 = {
    'hess_method': None,
}


def main():
    try:
        # Read reactant and product structures
        reactant = read(REACTANT_XYZ_FILE)
        product = read(PRODUCT_XYZ_FILE)
        logger.info("Successfully read reactant and product structures.")
    except Exception as e:
        logger.error(f"Error reading reactant and product structures: {e}")
        return

    try:
        # Create NEB job
        job1 = neb_job(reactant, product, calc_kwargs=calc_kwargs1)
        job1.update_metadata({"tag": f'neb_{TAG}'})
        logger.info("Created NEB job.")
    except Exception as e:
        logger.error(f"Error creating NEB job: {e}")
        return

    try:
        # Create TS job with custom Hessian
        job2 = ts_job(job1.output['neb_results']['highest_e_atoms'], use_custom_hessian=True)
        job2.update_metadata({"tag": f'ts_hess_{TAG}'})
        logger.info("Created TS job with custom Hessian.")
    except Exception as e:
        logger.error(f"Error creating TS job: {e}")
        return

    try:
        # Create IRC job in forward direction
        job3 = irc_job(job2.output['atoms'], direction='forward', **calc_kwargs1)
        job3.update_metadata({"tag": f'ircf_hess_{TAG}'})
        logger.info("Created IRC job in forward direction.")
    except Exception as e:
        logger.error(f"Error creating IRC job in forward direction: {e}")
        return

    try:
        # Create IRC job in reverse direction
        job4 = irc_job(job2.output['atoms'], direction='reverse', **calc_kwargs1)
        job4.update_metadata({"tag": f'ircr_hess_{TAG}'})
        logger.info("Created IRC job in reverse direction.")
    except Exception as e:
        logger.error(f"Error creating IRC job in reverse direction: {e}")
        return

    try:
        # Combine jobs into a flow
        jobs = [job1, job2, job3, job4]
        flow = jf.Flow(jobs)
        logger.info("Jobs combined into a flow.")
    except Exception as e:
        logger.error(f"Error combining jobs into a flow: {e}")
        return

    try:
        # Execute the workflow locally
        responses = jf.managers.local.run_locally(flow)
        logger.info("Workflow executed successfully.")
        logger.info(f"Responses: {responses}")
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        return


if __name__ == "__main__":
    main()
