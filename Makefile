PYTHON := python

ifndef file
$(error Use: make (run, train) file=file.py)
endif

run:
	$(PYTHON) -m models.$(file) run

train:
	$(PYTHON) -m models.$(file) train
