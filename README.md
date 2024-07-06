# Transition State Workflow Examples

This repository provides directly usable examples for Transition State (TS) workflows using NewtonNet and MACE. The examples include:

NewtonNet:
1. NEB + TS with Hessian
2. NEB + TS without Hessian
3. Geodesic + TS with Hessian
4. Geodesic + TS without Hessian

MACE:
1. NEB + TS with Hessian
2. NEB + TS without Hessian
3. Geodesic + TS with Hessian
4. Geodesic + TS without Hessian

## Getting Started

### Prerequisites

- Python 3.10
- Conda (for managing environments)

### Installation

1. **Create and Activate Conda Environment**

    Create a new Conda environment with Python 3.10:

    ```bash
    conda create -n test_env python=3.10
    ```

    Activate the newly created environment:

    ```bash
    conda activate test_env
    ```

2. **Clone the Repository and Install Dependencies**

    Clone the repository from GitHub:

    ```bash
    git clone https://github.com/kumaranu/ts-workflow-examples.git
    ```

    Navigate to the cloned repository:

    ```bash
    cd ts-workflow-examples
    ```

    Install the package in editable mode:

    ```bash
    pip install -e .
    ```

    Install the additional requirements for testing:

    ```bash
    pip install -r tests/requirements.txt
    ```

## Running Example Scripts

Navigate to one of the directories that contains example scripts inside the `src/ts-workflow-examples` directory. For example, to run a script from the `geodesic_ts_with_hessian` directory:

```bash
cd src/ts-workflow-examples/geodesic_ts_with_hessian
