PYTHON ?= python3

.PHONY: install test plots paper checksums check clean

install:
	$(PYTHON) -m pip install -e ".[train,plots,dev]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

plots:
	$(PYTHON) scripts/reproduce_plots.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

checksums:
	$(PYTHON) scripts/make_checksums.py

check: test plots paper checksums
	$(PYTHON) scripts/check_repo.py

clean:
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	cd paper && latexmk -C main.tex 2>/dev/null || true
