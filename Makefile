SHELL := /bin/bash
.PHONY: install clean test lint

lint:
	pylint cytofpy

clean_pycache:
	find . | grep -E "(__pycache__|\.pyc)" | xargs rm -rf

clean: clean_pycache
	rm -rf *.egg-info
	rm -rf .eggs
	rm -rf dist

install: test
	python3 -m pip install . --user

test:
	python3 setup.py test
