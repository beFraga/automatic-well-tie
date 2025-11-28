PYTHON := py

ifndef file
	$(error Use: make (run, train) file=file.py)

run:
	$(PYTHON) tests.$(file) run

train:
	$(PYTHON) tests.$(file) train
