# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Run all tests: `pytest`
- Run a specific test file: `pytest tests/test_file.py`
- Run a specific test function: `pytest tests/test_file.py::test_function`
- Run tests with keyword filtering: `pytest -k "pattern"`
- Run tests verbosely: `pytest -v`
- Run data module tests: `pytest tests/test_data.py`
- Run models module tests: `pytest tests/test_models.py`

## Code Style Guidelines
- Python version: 3.7+ with type hints
- Imports organization:
  1. Standard library (e.g., os, typing)
  2. Third-party packages (numpy, tensorflow, pandas)
  3. Local modules (from deeptimer import...)
- Follow docstrings format with Args, Returns, Raises sections
- Use type annotations for parameters and return values
- Naming: snake_case for functions/variables, CamelCase for classes
- Error handling: Raise explicit exceptions with informative messages
- Organization: Group related functions and keep them focused
- File structure: One class per file for major components