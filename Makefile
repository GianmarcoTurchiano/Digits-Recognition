#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = digits_recognition
PYTHON_VERSION = 3.12.7
PYTHON_ENV_NAME = .venv
REQUIREMENTS_FILE_NAME = requirements.in
DOCKER_IMAGE_NAME = digits-image
DOCKER_CONTAINER_NAME = digits-container
EMISSIONS_FILE_NAME = emissions.csv
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create and then activate a new python virtual environment
.PHONY: create_environment
create_environment:
	python -m venv $(PYTHON_ENV_NAME)
	@echo "Virtual environment created! Use '. $(PYTHON_ENV_NAME)/bin/python' to activate."

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	
## Exports dependencies and compiles them into a conflict-free requirements file.
.PHONY: export_requirements
export_requirements:
	$(PYTHON_INTERPRETER) -m pip freeze > $(REQUIREMENTS_FILE_NAME)
	pip-compile $(REQUIREMENTS_FILE_NAME)
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Checks the integrity of the DVC pipeline
.PHONY: pipeline_check
pipeline_check:
	dvc repro --dry

## Reproduces experiments
.PHONY: repro
repro:
	dvc repro
	dvc add $(EMISSIONS_FILE_NAME)

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 $(PROJECT_NAME)
	pylint $(PROJECT_NAME) --fail-under=9

## Launches behavioral tests on the current model.
.PHONY: test_behavior
test_behavior:
	$(PYTHON_INTERPRETER) -m pytest $(PROJECT_NAME)/experimentation/modeling/tests/behavioral_tests/

## Launches functional tests on the APIs
.PHONY: test_api
test_api:
	$(PYTHON_INTERPRETER) -m pytest $(PROJECT_NAME)/api/tests/

## Launches functional tests on the experimentation scripts.
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest $(PROJECT_NAME)/experimentation/modeling/tests/functional_tests/

## Soft resets to the previous git commit
.PHONY: rollback
rollback:
	git reset --soft HEAD~1

## Serves the APIs locally on the current machine.
.PHONY: api
api:
	uvicorn $(PROJECT_NAME).api.endpoints:app --reload

## Build a Docker image of the APIs
.PHONY: build_image
build_image:
	docker build -t $(DOCKER_IMAGE_NAME) .

## Creates a Docker container of the APIs from a previously built image of the project
.PHONY: create_container
build_container:
	docker create -p 8000:8000 --name $(DOCKER_CONTAINER_NAME) $(DOCKER_IMAGE_NAME)

## Starts a previously created Docker container of the APIs.
.PHONY: start_container
start_container:
	docker start -a $(DOCKER_CONTAINER_NAME)

## Displays a web page that summarizes recorded emissions data.
.PHONY: show_emissions
show_emissions:
	carbonboard --filepath=$(EMISSIONS_FILE_NAME) --port=3333

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
