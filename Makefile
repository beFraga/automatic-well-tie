PYTHON := python

ifndef file
$(error Use: make (run, train) file=file.py)
endif

run:
	$(PYTHON) -m tests.$(file) run

train:
	$(PYTHON) -m tests.$(file) train
