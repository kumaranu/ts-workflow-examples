# Installation Guide

Follow these steps to set up the `ts-workflow-examples` package on your local machine.

## 1. Create and Activate Conda Environment

Create a new Conda environment with Python 3.10:

```sh
conda create -n test_env python=3.10
```

Activate the newly created environment:

```sh
conda activate test_env
```

## 2. Clone the Repository and Install Dependencies

Clone the repository from GitHub:

```sh
git clone https://github.com/kumaranu/ts-workflow-examples.git
```
Navigate to the cloned repository:
```sh
cd ts-workflow-examples
```

Install the package in editable mode:
```sh
pip install -e .
```

Install the additional requirements for testing:
```sh
pip install -r tests/requirements.txt
```

## 3. Running Example Scripts

Navigate to one of the directories that contains example scripts inside the src/ts-workflow-examples directory.
For example, to run a script from the geodesic_ts_with_hessian directory:
```sh
cd src/ts-workflow-examples/geodesic_ts_with_hessian
```

Run the desired script (for example, using_newtonnet.py):
```sh
python using_newtonnet.py
```

Note: Ensure that you are in the correct directory as the paths inside the input files are relative.
This will allow the inputs and ML models provided with the repository to run correctly.

## 4. Additional Resources
    [NewtonNet](https://github.com/THGLab/NewtonNet) GitHub Repository
    [MACE](https://github.com/ACEsuit/mace/tree/main) GitHub Repository
    