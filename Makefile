mypy:
	poetry run mypy algorithms

pyright:
	poetry run pyright

watch:
	poetry run pyright --watch

lint:
	poetry run flake8 --max-line-length=100 --ignore=E203,W503 algorithms

test:
	poetry run python -m pytest tests/

lab:
	poetry run jupyter lab
