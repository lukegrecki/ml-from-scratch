mypy:
	poetry run mypy algorithms

lint:
	poetry run flake8 --max-line-length=100 --ignore=E203 algorithms

test: mypy lint
	poetry run python -m pytest tests/

lab:
	poetry run jupyter lab
