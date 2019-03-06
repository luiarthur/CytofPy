SHELL:=/bin/bash
.PHONY: getData test run clean retrieve retrieveAll

# EXPERIMENT NAME
EXP_NAME = sim1
# RESULTS DIRECTORY
RESULTS_DIR = results/$(EXP_NAME)
# AWS BUCKET
AWS_BUCKET = s3://cytof-vb/cb/$(subst _,-,$(EXP_NAME))
# TEST DIRECTORY
TEST_DIR = $(RESULTS_DIR)/test/

getData:
	aws s3 cp s3://cytof-cb-data/cb.txt data/cb.txt

run:
	bash run.sh $(RESULTS_DIR)/ $(AWS_BUCKET)

test:
	mkdir -p $(TEST_DIR)
	python3 cb.py $(TEST_DIR) 2 > $(TEST_DIR)/log.txt

clean:
	rm -rf results

retrieve: 
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR) --exclude "*/out.p*"

retrieveAll:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR)
