.PHONY: help install test lint doctor run publish clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s$$'\t'

install:  ## editable install with dev extras
	pip install -e ".[dev]"

test:  ## run the offline test suite
	pytest

test-network:  ## also hit the live public APIs
	SERR_NETWORK_TESTS=1 pytest -m network

lint:  ## ruff check
	ruff check standarderror experiments tests

doctor:  ## environment and credential check
	standarderror doctor

run:  ## run one experiment: make run EXP=exp001_chaos_horizon
	standarderror run $(EXP)

publish:  ## run + write the Hugo bundle and the Medium crosspost
	standarderror run $(EXP) --publish --medium

serve:  ## preview the site locally (needs hugo)
	cd site && hugo server -D

clean:
	rm -rf build .pytest_cache .ruff_cache site/public site/resources
	find . -name __pycache__ -type d -exec rm -rf {} +
