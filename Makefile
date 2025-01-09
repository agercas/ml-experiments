.PHONY: check format

# Check code with ruff
check:
	poetry run ruff check .

# Format code with ruff
format:
	poetry run ruff format .
