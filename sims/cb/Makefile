SHELL:=/bin/bash

.PHONY: getData test run

RESULTS_DIR = results/
AWS_BUCKET = NONE
TEST_DIR = $(RESULTS_DIR)/test/

getData:
	aws s3 cp s3://cytof-cb-data/cb.txt data/cb.txt

run:
	bash run.sh $(RESULTS_DIR)/sim1/ $(AWS_BUCKET)

test:
	mkdir -p $(TEST_DIR)
	python3 cb.py $(TEST_DIR) 2 > $(TEST_DIR)/log.txt
