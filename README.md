Experimental code for vec2gc clustering algorithm

## Running tests that require NetworKit (micromamba)

For contributors who want to run the NetworKit-enabled tests locally, you can use micromamba to create a reproducible environment (recommended):

```bash
# Install micromamba (see https://mamba.readthedocs.io/)
# Conda-based install (if you already have conda):
conda install -n base -c conda-forge micromamba -y

# Create env from provided environment.yml and run tests
micromamba create -f environment.yml -n vec2gc -y
micromamba run -n vec2gc pytest -q
```

The repo provides `environment.yml` (conda-forge) to reproduce the CI environment. The GitHub Actions workflow includes a dedicated `test-networkit` job that uses micromamba to run tests with NetworKit installed.
