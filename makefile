.PHONY: help
.PHONY: conda-env
.PHONY: install
.PHONY: test test-exp
.PHONY: isort black flake8
.PHONY: clean clean-seed9 extract

.DEFAULT: help
help:
	@echo "test"
	@echo "        Run pytest on the project and report coverage"
	@echo "test-exp"
	@echo "        Run pytest on the experiment utilities"
	@echo "black"
	@echo "        Run black on the project"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "install"
	@echo "        Install dependencies for running experiments"
	@echo "conda-env"
	@echo "        Create conda environment 'hbp-experiments' with all dependencies"
	@echo "isort"
	@echo "        Run isort on the project"
	@echo "clean"
	@echo "        Remove data and figures from previous runs"
	@echo "extract"
	@echo "        Extract original data"
	@echo "clean-seed9"
	@echo "        Remove data for random seed 9"
###

###
# Test coverage
test:
	@pytest -vx --cov=bpexts tests
test-exp:
	@pytest -vx exp

###
# Linter and Auto-formatter

# Uses black.toml config instead of pyproject.toml to avoid pip issues. See
# - https://github.com/psf/black/issues/683
# - https://github.com/pypa/pip/pull/6370
# - https://pip.pypa.io/en/stable/reference/pip/#pep-517-and-518-support
black:
	@black . --config=black.toml

flake8:
	@flake8 .

isort:
	@isort --apply

###
# Installation

install:
	@pip install -r requirements.txt
	@pip install -e .

###
# Conda environment
conda-env:
	@conda env create --file .conda_env.yml

###
# Clean up
clean:
	@rm -rf dat/
	@rm -rf fig/

clean-seed9:
	find dat -name "seed9" -type d -exec rm -rvf {} +

extract:
	@bash scripts/extract_data.sh
