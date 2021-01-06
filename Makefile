mypy:
	poetry run mypy algorithms

lint:
	poetry run flake8 --max-line-length=100 algorithms

test: mypy lint
	poetry run python -m pytest tests/

lab:
	poetry run jupyter lab
