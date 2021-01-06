mypy:
	poetry run mypy algorithms

test: mypy
	poetry run python -m pytest tests/

lab:
	poetry run jupyter lab
