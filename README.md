# Introduction
In this work we characterize how the mobility changes for lockdown restrictions

# Init
#### Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

#### Create the environment
cd HumMobCov
uv sync            # core + numba + polars
uv sync --group dev  # + jupyterlab, ipykernel, nbdime

source .venv/bin/activate


# Usage
/src/main.ipynb