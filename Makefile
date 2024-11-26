#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = digits_recognition
PYTHON_VERSION = 3.12.7
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 digits_recognition
	pylint digits_recognition

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml digits_recognition

.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv

.PHONY: export_requirements
export_requirements:
	$(PYTHON_INTERPRETER) -m pip freeze > requirements.txt

.PHONY: test_behavior
test_behavior:
	$(PYTHON_INTERPRETER) -m pytest digits_recognition/experimentation/modeling/behavioral_tests/

.PHONY: test_api
test_api:
	$(PYTHON_INTERPRETER) -m pytest digits_recognition/api/tests/

.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest digits_recognition/experimentation/modeling/tests/

.PHONY: rollback
rollback:
	git reset --soft HEAD~1

.PHONY: api
api:
	uvicorn digits_recognition.api.endpoints:app --reload

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
