mypy:
	poetry run mypy algorithms

pyright:
	pyright

watch:
	pyright --watch

lint:
	poetry run flake8 --max-line-length=100 --ignore=E203,W503 algorithms

test: pyright lint
	poetry run python -m pytest tests/

lab:
	poetry run jupyter lab
