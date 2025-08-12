.PHONY: format

format:
	poetry run ruff check --fix .
	poetry run ruff format .
